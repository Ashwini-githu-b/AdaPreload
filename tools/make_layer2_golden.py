#!/usr/bin/env python3
"""Generate Layer 2 adapter golden vectors from the original PyTorch formulation (Phase D).

The adapter steps are the notebook's own statements (nb[22] L28-32, L50-51, L68-71; identical in
nb[33] L127-130, L148-149, L173-177):

    adapter = nn.Linear(64, 88); weight and bias zeroed
    opt = torch.optim.SGD(adapter.parameters(), lr=0.001); crit = nn.CrossEntropyLoss()
    final_logits = backbone_logits + adapter(hidden)          # prediction uses the current state
    opt.zero_grad(); loss = crit(final_logits, target); loss.backward(); opt.step()

Inputs (hidden, backbone_logits) come from the frozen Layer 1 checkpoint run in float32 exactly as
nb[33] does, on SYNTHETIC app-id sequences (numpy default_rng(42)). No LSApp user's history is
used, and nothing here is shipped to or loaded into the device adapter.

Output: app/src/androidTest/assets/layer2_golden.json (deterministic for a given torch version).

Usage:
  python3 tools/make_layer2_golden.py layer1_backbone_final.pt
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_vocabulary import CHECKPOINT_SHA256, NOTEBOOK_SHA256, sha256_of  # noqa: E402
from export_layer1 import WINDOW, BackboneMorph, get_hidden  # noqa: E402  (verbatim notebook code)

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "app" / "src" / "androidTest" / "assets" / "layer2_golden.json"
SEED = 42
LR = 0.001
SEQUENCE_STEPS = 30


def backbone_outputs(backbone, context):
    """hidden [64] and backbone logits [88] for one context, as nb[33] L136-147 computes them."""
    app_t = torch.tensor([context], dtype=torch.long)
    zeros = torch.zeros(1, len(context), dtype=torch.float32)
    mask_t = torch.zeros(1, len(context), dtype=torch.bool)
    with torch.no_grad():
        hidden = get_hidden(backbone, app_t, zeros, zeros, zeros, mask_t).detach()
        backbone_logits = backbone.out(hidden)
    return hidden, backbone_logits  # shapes (1, 64), (1, 88), float32


def run_adapter(steps, dtype):
    """The notebook's Layer 2 loop over (hidden, backbone_logits, target) steps."""
    adapter = nn.Linear(64, 88).to(dtype)
    adapter.weight.data.zero_()
    adapter.bias.data.zero_()
    adapter_opt = torch.optim.SGD(adapter.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    records = []
    for hidden, backbone_logits, target in steps:
        hidden, backbone_logits = hidden.to(dtype), backbone_logits.to(dtype)
        adapter_logits = adapter(hidden)
        final_logits = backbone_logits + adapter_logits
        before = final_logits.detach().clone()
        target_t = torch.tensor([target], dtype=torch.long)
        adapter_opt.zero_grad()
        loss = crit(final_logits, target_t)
        loss.backward()
        # The gradient PyTorch applies must be softmax(final) - onehot(target) (checked, not assumed).
        expected_grad = torch.softmax(before, dim=1)[0].clone()
        expected_grad[target] -= 1
        assert torch.allclose(adapter.bias.grad, expected_grad, rtol=0, atol=1e-6 if dtype == torch.float32 else 1e-14)
        adapter_opt.step()
        records.append({"final_logits_before_update": before[0], "loss_before": loss.detach()})
    return adapter, records


def floats(t):
    return [float(x) for x in t.detach().reshape(-1).tolist()]


def fp32_sha256(t):
    return hashlib.sha256(t.detach().to(torch.float32).contiguous().numpy().astype("<f4").tobytes()).hexdigest()


def synthetic_sequence(rng, length):
    """A synthetic launch sequence over app ids 1..87 with no consecutive repeats (as after T3)."""
    seq = [int(rng.integers(1, 88))]
    while len(seq) < length:
        a = int(rng.integers(1, 88))
        if a != seq[-1]:
            seq.append(a)
    return seq


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    assert sha256_of(args.checkpoint) == CHECKPOINT_SHA256, "checkpoint hash mismatch"

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    backbone = BackboneMorph(ckpt["vocab_size"], d_model=ckpt["d_model"], morph_mode=ckpt["morph_mode"])
    backbone.load_state_dict(ckpt["model_state_dict"], strict=True)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    # Defaults the notebook relies on, checked rather than assumed.
    probe = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=LR).defaults
    assert (probe["momentum"], probe["dampening"], probe["weight_decay"], probe["nesterov"], probe["maximize"]) == (0, 0, 0, False, False)
    ce = nn.CrossEntropyLoss()
    assert (ce.reduction, ce.ignore_index, ce.label_smoothing, ce.weight) == ("mean", -100, 0.0, None)

    rng = np.random.default_rng(SEED)

    # One-step case: a full 20-launch synthetic context and a target that is not its last app.
    one_seq = synthetic_sequence(rng, WINDOW + 1)
    h1, l1 = backbone_outputs(backbone, one_seq[:WINDOW])
    one_steps = [(h1, l1, one_seq[WINDOW])]
    adapter32, rec32 = run_adapter(one_steps, torch.float32)
    adapter64, rec64 = run_adapter(one_steps, torch.float64)
    with torch.no_grad():
        after32 = l1 + adapter32(h1)
        after64 = l1.double() + adapter64(h1.double())

    # Sequence: prequential over a synthetic launch sequence, context = last <= 20 launches (T7).
    seq = synthetic_sequence(rng, SEQUENCE_STEPS + 1)
    steps = []
    for i in range(1, SEQUENCE_STEPS + 1):
        h, lg = backbone_outputs(backbone, seq[max(0, i - WINDOW):i])
        steps.append((h, lg, seq[i]))
    seq32, seq_rec32 = run_adapter(steps, torch.float32)
    seq64, seq_rec64 = run_adapter(steps, torch.float64)

    def envelope(a32, a64):
        return float((a32.detach().double() - a64.detach()).abs().max())

    rounding = {
        "one_step_weight": envelope(adapter32.weight, adapter64.weight),
        "one_step_bias": envelope(adapter32.bias, adapter64.bias),
        "one_step_logits_after": envelope(after32, after64),
        "one_step_loss": abs(float(rec32[0]["loss_before"]) - float(rec64[0]["loss_before"])),
        "sequence_weight": envelope(seq32.weight, seq64.weight),
        "sequence_bias": envelope(seq32.bias, seq64.bias),
        "sequence_logits_before_update": max(envelope(a["final_logits_before_update"], b["final_logits_before_update"])
                                             for a, b in zip(seq_rec32, seq_rec64)),
        "sequence_loss": max(abs(float(a["loss_before"]) - float(b["loss_before"])) for a, b in zip(seq_rec32, seq_rec64)),
    }

    golden = {
        "format": "adapreload-layer2-golden",
        "format_version": 1,
        "description": "Layer 2 adapter reference values from the notebook's PyTorch formulation (float32). "
                       "Inputs are frozen Layer 1 outputs on synthetic app-id sequences; no LSApp user history. "
                       "Weights are row-major [88][64] like nn.Linear.weight. Generated by tools/make_layer2_golden.py.",
        "source": {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "notebook_sha256": NOTEBOOK_SHA256,
            "formulation": "nb[22] L28-32, L50-51, L68-71 (= nb[33] L127-130, L148-149, L173-177)",
            "torch_version": torch.__version__,
            "seed": SEED,
        },
        "learning_rate": LR,
        "fp32_vs_fp64_rounding": rounding,
        "one_step": {
            "context": one_seq[:WINDOW],
            "hidden": floats(h1),
            "backbone_logits": floats(l1),
            "target": one_seq[WINDOW],
            "loss_before": float(rec32[0]["loss_before"]),
            "weight_after": floats(adapter32.weight),
            "bias_after": floats(adapter32.bias),
            "weight_after_fp32_sha256": fp32_sha256(adapter32.weight),
            "bias_after_fp32_sha256": fp32_sha256(adapter32.bias),
            "final_logits_after_same_hidden": floats(after32),
        },
        "sequence": {
            "launches": seq,
            "steps": [{"hidden": floats(h), "backbone_logits": floats(lg), "target": t,
                       "final_logits_before_update": floats(r["final_logits_before_update"]),
                       "loss_before": float(r["loss_before"])}
                      for (h, lg, t), r in zip(steps, seq_rec32)],
            "update_count": len(steps),
            "weight_after": floats(seq32.weight),
            "bias_after": floats(seq32.bias),
            "weight_after_fp32_sha256": fp32_sha256(seq32.weight),
            "bias_after_fp32_sha256": fp32_sha256(seq32.bias),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(golden, indent=1) + "\n", encoding="ascii")
    print(f"one step: target {one_seq[WINDOW]}, loss {float(rec32[0]['loss_before']):.6f}; "
          f"sequence: {len(steps)} steps, final loss {float(seq_rec32[-1]['loss_before']):.6f}")
    print("fp32 vs fp64 rounding envelope:", json.dumps({k: f"{v:.3e}" for k, v in rounding.items()}))
    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size:,} bytes, sha256 {sha256_of(OUT)})")


if __name__ == "__main__":
    main()
