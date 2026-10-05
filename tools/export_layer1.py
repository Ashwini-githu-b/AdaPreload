#!/usr/bin/env python3
"""Export the frozen AdaPreload Layer 1 backbone for Android and generate parity golden vectors.

Phase C (numerical parity). Read-only with respect to the checkpoint and the Layer 2 artifact:
weights are copied byte-for-byte, the architecture is the notebook's own class, and nothing is
retrained. Re-running with the same torch version produces byte-identical outputs.

Outputs:
  app/src/main/assets/layer1/layer1_weights.bin            float32 little-endian tensors, state_dict order
  app/src/main/assets/layer1/layer1_manifest.json          architecture, tensor table, hashes
  app/src/androidTest/assets/layer1_golden.json            golden cases: full fp32 and fp64 reference outputs
  app/src/androidTest/assets/layer1_artifact_sample.json   frozen Colab Layer 1 top-20 outputs for sampled events

Usage:
  python3 tools/export_layer1.py layer1_backbone_final.pt layer2_probability_outputs.pkl
"""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_vocabulary import (  # noqa: E402
    CHECKPOINT_SHA256, LAYER2_SHA256, NOTEBOOK_SHA256, NumpyOnlyUnpickler, sha256_of,
)

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "app" / "src" / "main" / "assets"
TEST_ASSETS = REPO / "app" / "src" / "androidTest" / "assets"
SEED = 42

# ---------------------------------------------------------------------------------------------
# Reference implementation: copied verbatim from the notebook (sha256 47540816...).
# BackboneMorph: nb[17] (CELL 21 revised) L11-53. get_hidden: nb[22] (CELL 26) L11-17.
# ---------------------------------------------------------------------------------------------
WINDOW = 20


class BackboneMorph(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, nlayers=2, maxlen=WINDOW, morph_mode='none'):
        super().__init__()
        self.morph_mode = morph_mode
        self.app_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, d_model)
        if morph_mode in ('basic', 'side_channel'):
            self.time_proj = nn.Linear(3, d_model)
        elif morph_mode == 'mlp':
            self.fuse_mlp = nn.Sequential(nn.Linear(d_model + 3, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        elif morph_mode == 'gated':
            self.gate_net = nn.Linear(d_model + 3, d_model)
            self.time_proj_g = nn.Linear(3, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128,
                                                batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.out = nn.Linear(d_model, vocab_size)
        self.maxlen = maxlen

    def pos_component(self, pos_ids, dts, hsin, hcos):
        pe = self.pos_emb(pos_ids)
        if self.morph_mode in ('none', 'side_channel'):
            return pe  # position untouched in both — the difference is handled in forward()
        tf = torch.stack([dts, hsin, hcos], dim=-1)
        if self.morph_mode == 'basic':
            return pe + self.time_proj(tf)
        elif self.morph_mode == 'mlp':
            return self.fuse_mlp(torch.cat([pe, tf], dim=-1))
        elif self.morph_mode == 'gated':
            g = torch.sigmoid(self.gate_net(torch.cat([pe, tf], dim=-1)))
            return g * pe + (1 - g) * self.time_proj_g(tf)

    def forward(self, app_ids, dts, hsin, hcos, pad_mask):
        B, L = app_ids.shape
        pos_ids = torch.arange(self.maxlen - L, self.maxlen, device=app_ids.device).unsqueeze(0).expand(B, L)
        x = self.app_emb(app_ids) + self.pos_component(pos_ids, dts, hsin, hcos)
        if self.morph_mode == 'side_channel':
            tf = torch.stack([dts, hsin, hcos], dim=-1)
            x = x + self.time_proj(tf)
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        last_idx = (~pad_mask).float().cumsum(dim=1).argmax(dim=1)
        last_hidden = h[torch.arange(B), last_idx]
        return self.out(last_hidden)


def get_hidden(backbone, app_ids, dts, hsin, hcos, pad_mask):
    B, L = app_ids.shape
    pos_ids = torch.arange(backbone.maxlen - L, backbone.maxlen).unsqueeze(0).expand(B, L)
    x = backbone.app_emb(app_ids) + backbone.pos_component(pos_ids, dts, hsin, hcos)
    h = backbone.encoder(x, src_key_padding_mask=pad_mask)
    last_idx = (~pad_mask).float().cumsum(dim=1).argmax(dim=1)
    return h[torch.arange(B), last_idx]
# ------------------------------------------------------------------------------- end verbatim


def reference_outputs(backbone, contexts, dtype, time_features=None):
    """Layer 1 logits and padding-masked probabilities exactly as nb[33] L135-158 computes them:
    the unpadded context (mask all False), get_hidden, backbone.out, logit[0] = -inf, softmax,
    padding column dropped. Contexts must share one length so they can be batched."""
    L = len(contexts[0])
    app_t = torch.tensor(contexts, dtype=torch.long)
    if time_features is None:
        zeros = torch.zeros(len(contexts), L, dtype=dtype)
        dts = hsin = hcos = zeros
    else:
        dts, hsin, hcos = (t.to(dtype) for t in time_features)
    mask_t = torch.zeros(len(contexts), L, dtype=torch.bool)
    with torch.no_grad():
        hidden = get_hidden(backbone, app_t, dts, hsin, hcos, mask_t)
        logits = backbone.out(hidden)
        masked = logits.clone()
        masked[:, 0] = float('-inf')
        probs = torch.softmax(masked, dim=1)[:, 1:]
    return logits, probs


def left_pad(context, window=WINDOW):
    return [0] * (window - len(context)) + list(context)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("layer2")
    args = parser.parse_args()

    print("1. Input hashes")
    assert sha256_of(args.checkpoint) == CHECKPOINT_SHA256, "checkpoint hash mismatch"
    assert sha256_of(args.layer2) == LAYER2_SHA256, "Layer 2 artifact hash mismatch"
    print("   checkpoint and Layer 2 artifact match the frozen manifest")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert (ckpt["vocab_size"], ckpt["window"], ckpt["d_model"], ckpt["morph_mode"]) == (88, WINDOW, 64, "none")
    vocabulary = json.loads((ASSETS / "lsapp_vocabulary.json").read_text())
    assert {a["name"]: a["id"] for a in vocabulary["apps"]} == ckpt["app2id"], "vocabulary differs from checkpoint app2id"

    backbone = BackboneMorph(ckpt["vocab_size"], d_model=ckpt["d_model"], morph_mode=ckpt["morph_mode"])
    backbone.load_state_dict(ckpt["model_state_dict"], strict=True)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone64 = copy.deepcopy(backbone).double()

    # Architecture as instantiated (read from the modules, not assumed).
    layer = backbone.encoder.layers[0]
    assert all(type(l) is type(layer) for l in backbone.encoder.layers)
    architecture = {
        "reference_class": "BackboneMorph (notebook nb[17] L11-53), inference path get_hidden (nb[22] L11-17) as used in nb[33] L135-158",
        "morph_mode": backbone.morph_mode,
        "vocab_size": backbone.out.out_features,
        "padding_id": backbone.app_emb.padding_idx,
        "window": backbone.maxlen,
        "d_model": backbone.app_emb.embedding_dim,
        "num_layers": len(backbone.encoder.layers),
        "num_heads": layer.self_attn.num_heads,
        "head_dim": layer.self_attn.head_dim,
        "dim_feedforward": layer.linear1.out_features,
        "activation": {nn.functional.relu: "relu", nn.functional.gelu: "gelu"}.get(layer.activation, str(layer.activation)),
        "norm_first": layer.norm_first,
        "layer_norm_eps": layer.norm1.eps,
        "final_encoder_norm": backbone.encoder.norm is not None,
        "positions": "right-aligned: position ids arange(window - L, window) for L real tokens",
        "readout": "encoder output at the last real token, then out: Linear(d_model, vocab_size)",
        "probabilities": "softmax over logits with logit[padding_id] = -inf; probabilities[k] is app id k + 1",
    }
    assert architecture["activation"] == "relu" and not architecture["norm_first"] and not architecture["final_encoder_norm"]
    assert layer.self_attn._qkv_same_embed_dim and layer.self_attn.batch_first
    print(f"2. Architecture: {json.dumps({k: v for k, v in architecture.items() if not isinstance(v, str) or len(v) < 20})}")

    # ---- Weights: exact float32 bytes in state_dict order ----
    state = ckpt["model_state_dict"]
    blob, tensors, offset = bytearray(), [], 0
    for name, t in state.items():
        assert t.dtype == torch.float32
        raw = t.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
        tensors.append({"name": name, "shape": list(t.shape), "offset": offset, "count": t.numel(),
                        "sha256": hashlib.sha256(raw).hexdigest()})
        blob += raw
        offset += len(raw)
    blob = bytes(blob)
    # Round trip: the bytes must reproduce every tensor bit-for-bit.
    for spec in tensors:
        back = np.frombuffer(blob, dtype="<f4", count=spec["count"], offset=spec["offset"]).reshape(spec["shape"])
        assert np.array_equal(back.view(np.uint32), state[spec["name"]].numpy().view(np.uint32)), spec["name"]
    weights_sha = hashlib.sha256(blob).hexdigest()
    print(f"3. Weights: {len(tensors)} tensors, {sum(s['count'] for s in tensors):,} parameters, "
          f"{len(blob):,} bytes, bit-exact round trip, sha256 {weights_sha}")

    manifest = {
        "format": "adapreload-layer1-weights",
        "format_version": 1,
        "description": "Frozen AdaPreload Layer 1 backbone, exported byte-for-byte from the audited checkpoint "
                       "by tools/export_layer1.py. Do not edit by hand.",
        "source": {
            "checkpoint": {"file": "layer1_backbone_final.pt", "sha256": CHECKPOINT_SHA256,
                           "bytes": Path(args.checkpoint).stat().st_size},
            "notebook": {"file": "Copy_of_adaPrealoaderneww.ipynb", "sha256": NOTEBOOK_SHA256},
            "vocabulary_apps_sha256": vocabulary["apps_sha256"],
            "exporter": "tools/export_layer1.py",
            "torch_version": torch.__version__,
        },
        "architecture": architecture,
        "weights": {"file": "layer1_weights.bin", "dtype": "float32", "byte_order": "little",
                    "bytes": len(blob), "sha256": weights_sha,
                    "parameter_count": sum(s["count"] for s in tensors)},
        "tensors": tensors,
    }

    # ---- Reproduction check against the frozen Colab outputs (all 30,040 held-out events) ----
    with open(args.layer2, "rb") as f:
        l2 = NumpyOnlyUnpickler(f).load()
    contexts = [list(map(int, c)) for c in l2["context_app_ids"]]
    art_ids, art_probs = np.asarray(l2["top_l1_ids"]), np.asarray(l2["top_l1_probs"])
    by_len = {}
    for i, c in enumerate(contexts):
        by_len.setdefault(len(c), []).append(i)
    ref32 = np.zeros((len(contexts), 87), dtype=np.float32)
    ref64 = np.zeros((len(contexts), 87), dtype=np.float64)
    logit32 = np.zeros((len(contexts), 88), dtype=np.float32)
    logit64 = np.zeros((len(contexts), 88), dtype=np.float64)
    for L, idx in sorted(by_len.items()):
        batch = [contexts[i] for i in idx]
        lo, pr = reference_outputs(backbone, batch, torch.float32)
        ref32[idx], logit32[idx] = pr.numpy(), lo.numpy()
        lo, pr = reference_outputs(backbone64, batch, torch.float64)
        ref64[idx], logit64[idx] = pr.numpy(), lo.numpy()
    mine_at_art = np.take_along_axis(ref32, art_ids - 1, axis=1)
    top20_mine = np.argsort(-ref32, axis=1, kind="stable")[:, :20] + 1
    print("4. Local PyTorch fp32 reproduction vs frozen Colab Layer 1 top-20 (Layer 2 artifact, all events):")
    print(f"   events {len(contexts):,}; max |prob diff| at Colab top-20 ids {np.abs(mine_at_art - art_probs).max():.3e}; "
          f"top-1 identical {np.mean(top20_mine[:, 0] == art_ids[:, 0]):.6f}; "
          f"top-5 sets identical {np.mean([set(a[:5]) == set(b[:5]) for a, b in zip(top20_mine, art_ids)]):.6f}; "
          f"top-20 lists identical {np.mean((top20_mine == art_ids).all(axis=1)):.6f}")
    print(f"   fp32 vs fp64 reference: max |prob diff| {np.abs(ref32 - ref64).max():.3e}, "
          f"max |logit diff| {np.abs(logit32 - logit64).max():.3e}, max |logit| {np.abs(logit64).max():.2f}")

    # ---- Semantics checks ----
    sample_len = {L: idx[0] for L, idx in by_len.items()}
    worst_pad = 0.0
    for L, i in sorted(sample_len.items()):
        ctx = contexts[i]
        padded = torch.tensor([left_pad(ctx)], dtype=torch.long)
        mask = padded == 0
        zeros = torch.zeros(1, WINDOW, dtype=torch.float64)
        with torch.no_grad():
            lo_pad = backbone64(padded, zeros, zeros, zeros, mask)[0].numpy()
        worst_pad = max(worst_pad, float(np.abs(lo_pad - logit64[i]).max()))
    g = torch.Generator().manual_seed(SEED)
    batch = [contexts[i] for i in by_len[WINDOW][:64]]
    rand_tf = [torch.randn(len(batch), WINDOW, generator=g) for _ in range(3)]
    base, _ = reference_outputs(backbone64, batch, torch.float64)
    with_time, _ = reference_outputs(backbone64, batch, torch.float64, rand_tf)
    print(f"5. Left-padded 20-token forward with key-padding mask vs unpadded reference (fp64): "
          f"max |logit diff| {worst_pad:.3e} over lengths {min(sample_len)}..{max(sample_len)}")
    print(f"   time features ignored (morph_mode 'none'): max |logit diff| with random time features "
          f"{(base - with_time).abs().max().item():.3e}")

    # ---- Golden cases ----
    rng = np.random.default_rng(SEED)
    users, event_index = np.asarray(l2["user_id"]), np.asarray(l2["event_index"])
    order = sorted(range(len(contexts)), key=lambda i: (str(users[i]), int(event_index[i])))
    cases = []
    for L in (1, 2, 5, 10, 19):
        i = next(j for j in order if len(contexts[j]) == L)
        cases.append(("artifact_len%02d" % L, i, contexts[i]))
    for n, i in enumerate(sorted(rng.choice(by_len[WINDOW], size=4, replace=False).tolist())):
        cases.append((f"artifact_full_{n}", i, contexts[i]))
    synthetic = {
        "synthetic_single_id1": [1],
        "synthetic_single_id87": [87],
        "synthetic_pair_87_1": [87, 1],
        "synthetic_full_ids_1_to_20": list(range(1, 21)),
        "synthetic_full_ids_68_to_87": list(range(68, 88)),
        "synthetic_full_repeated_id45": [45] * 20,
        "synthetic_alternating_82_33": [82, 33] * 10,
    }
    cases += [(name, None, ctx) for name, ctx in synthetic.items()]

    golden_cases = []
    for name, i, ctx in cases:
        lo32, pr32 = reference_outputs(backbone, [ctx], torch.float32)
        lo64, pr64 = reference_outputs(backbone64, [ctx], torch.float64)
        case = {
            "name": name,
            "source": "synthetic" if i is None else
                      f"layer2 artifact event (user {users[i]}, event_index {int(event_index[i])})",
            "context_length": len(ctx),
            "context": left_pad(ctx),
            "expected_top5_ids": (torch.argsort(pr32[0], descending=True, stable=True)[:5] + 1).tolist(),
            "expected_logits_fp32": [float(x) for x in lo32[0].tolist()],
            "expected_probabilities_fp32": [float(x) for x in pr32[0].tolist()],
            "expected_logits_fp64": lo64[0].tolist(),
            "expected_probabilities_fp64": pr64[0].tolist(),
        }
        if i is not None:
            case["colab_top20_l1_ids"] = art_ids[i].tolist()
            case["colab_top20_l1_probs"] = [float(x) for x in art_probs[i].tolist()]
        golden_cases.append(case)

    golden = {
        "format": "adapreload-layer1-golden",
        "format_version": 1,
        "description": "Layer 1 reference outputs from the notebook's own PyTorch code on the audited checkpoint "
                       "(nb[33] L135-158 path). Contexts are left-padded to the window with padding id 0. "
                       "probabilities[k] is app id k + 1. Generated by tools/export_layer1.py.",
        "source": {"checkpoint_sha256": CHECKPOINT_SHA256, "layer2_artifact_sha256": LAYER2_SHA256,
                   "notebook_sha256": NOTEBOOK_SHA256, "vocabulary_apps_sha256": vocabulary["apps_sha256"],
                   "weights_sha256": weights_sha, "torch_version": torch.__version__},
        "model": {k: architecture[k] for k in ("vocab_size", "padding_id", "window", "d_model", "morph_mode")},
        "cases": golden_cases,
    }

    # ---- Frozen Colab sample: per context length, every short context up to 4 per length, plus random full ones ----
    sample_idx = []
    for L in range(1, WINDOW):
        sample_idx += [j for j in order if len(contexts[j]) == L][:4]
    sample_idx += sorted(rng.choice(by_len[WINDOW], size=256 - len(sample_idx), replace=False).tolist())
    sample = {
        "format": "adapreload-layer1-colab-sample",
        "format_version": 1,
        "description": "Frozen Colab Layer 1 top-20 outputs (layer2_probability_outputs.pkl: top_l1_ids, top_l1_probs, "
                       "float32) for sampled held-out events. Contexts are left-padded to the window with padding id 0.",
        "source": {"layer2_artifact_sha256": LAYER2_SHA256, "checkpoint_sha256": CHECKPOINT_SHA256,
                   "sampling": f"per length 1..19 the first 4 events in (user, event_index) order; the rest uniform "
                               f"over full-length events, numpy default_rng({SEED})"},
        "events": [{"user_id": str(users[j]), "event_index": int(event_index[j]), "context": left_pad(contexts[j]),
                    "top20_ids": art_ids[j].tolist(), "top20_probs": [float(x) for x in art_probs[j].tolist()]}
                   for j in sample_idx],
    }

    def write(path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=1) + "\n", encoding="ascii")
        print(f"   wrote {path.relative_to(REPO)} ({path.stat().st_size:,} bytes, sha256 {sha256_of(path)})")

    print("6. Outputs")
    out_dir = ASSETS / "layer1"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "layer1_weights.bin").write_bytes(blob)
    print(f"   wrote {(out_dir / 'layer1_weights.bin').relative_to(REPO)} ({len(blob):,} bytes)")
    write(out_dir / "layer1_manifest.json", manifest)
    write(TEST_ASSETS / "layer1_golden.json", golden)
    write(TEST_ASSETS / "layer1_artifact_sample.json", sample)
    print(f"   {len(golden_cases)} golden cases, {len(sample['events'])} Colab sample events")


if __name__ == "__main__":
    main()
