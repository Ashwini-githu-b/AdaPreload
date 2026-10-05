#!/usr/bin/env python3
"""Audit the frozen AdaPreload Layer 1 checkpoint and export its vocabulary for Android.

Implements the extraction procedure in docs/ANDROID_TRACE_SPEC.md section 5.3.

Read-only with respect to the inputs: the checkpoint and the Layer 2 artifact are never
modified. Every check runs before anything is written; if any check fails, no output file
is produced.

Usage:
    python3 tools/audit_vocabulary.py LAYER1_CHECKPOINT.pt LAYER2_ARTIFACT.pkl \
        [--out app/src/main/assets/lsapp_vocabulary.json]

Requires torch and numpy >= 2 (the Layer 2 artifact was pickled with numpy 2).
"""
import argparse
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
SPEC = REPO / "docs" / "ANDROID_TRACE_SPEC.md"
DEFAULT_OUT = REPO / "app" / "src" / "main" / "assets" / "lsapp_vocabulary.json"

# Frozen manifest values (notebook nb[43] output; spec "Provenance").
CHECKPOINT_SHA256 = "fd668f160363c9b32fe5ac561bec7b46ff5492d84e10f31d0bfd6d7a0c21c092"
LAYER2_SHA256 = "bfe1e61f1c0d4a90b90dc3623db0e13887915ec05b300f5ac2128c2b8535a2c8"
NOTEBOOK_SHA256 = "47540816a02d867a032865c903ab1a2489d06d0700c86dc216feb603867fc523"

EXPECTED_KEYS = {"model_state_dict", "app2id", "vocab_size", "window", "d_model", "morph_mode"}
N_APPS, VOCAB_SIZE, WINDOW, D_MODEL, PAD_ID, PAD_TOKEN = 87, 88, 20, 64, 0, "<PAD>"
N_EVENTS, N_USERS, TOPN = 30040, 59, 20

failures = []


def check(check_id, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {check_id}" + (f" - {detail}" if detail else ""))
    if not ok:
        failures.append(check_id)
    return ok


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class NumpyOnlyUnpickler(pickle.Unpickler):
    """The Layer 2 artifact is a plain pickle. Only the numpy array classes it actually
    references are allowed, so loading it cannot execute arbitrary code."""

    ALLOWED = {
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def appendix_vocabulary():
    """The non-authoritative public-LSApp re-run listed in the spec appendix (check C7)."""
    text = SPEC.read_text(encoding="utf-8")
    appendix = text[text.index("## Appendix"):]
    return {int(i): name for i, name in re.findall(r"\| (\d+) \| `([^`]+)`", appendix)}


def vocabulary_sha256(id_to_name):
    lines = "".join(f"{i}\t{id_to_name[i]}\n" for i in sorted(id_to_name))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("layer2")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    print("H. File hashes (checked before anything is loaded)")
    ck_sha, l2_sha = sha256_of(args.checkpoint), sha256_of(args.layer2)
    check("H1 checkpoint SHA-256", ck_sha == CHECKPOINT_SHA256, ck_sha)
    check("H2 Layer 2 artifact SHA-256", l2_sha == LAYER2_SHA256, l2_sha)
    if failures:
        sys.exit("ABORT: hash mismatch; nothing loaded, nothing written.")

    print("\nC. Layer 1 checkpoint (torch.load, weights_only=True)")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    check("C1 keys", set(ckpt) == EXPECTED_KEYS, str(sorted(ckpt)))
    check(
        "C2 config",
        (ckpt.get("vocab_size"), ckpt.get("window"), ckpt.get("d_model"), ckpt.get("morph_mode"))
        == (VOCAB_SIZE, WINDOW, D_MODEL, "none"),
        f"vocab_size={ckpt.get('vocab_size')} window={ckpt.get('window')} "
        f"d_model={ckpt.get('d_model')} morph_mode={ckpt.get('morph_mode')!r}",
    )
    app2id = ckpt["app2id"]
    check(
        "C3 ids exactly 1..87, str keys, int values, padding id 0 absent",
        len(app2id) == N_APPS
        and all(type(k) is str for k in app2id)
        and all(type(v) is int for v in app2id.values())
        and sorted(app2id.values()) == list(range(1, N_APPS + 1))
        and PAD_ID not in app2id.values()
        and PAD_TOKEN not in app2id,
        f"{len(app2id)} entries, id range [{min(app2id.values())}, {max(app2id.values())}]",
    )
    check(
        "C4 ids follow Python code-point sort order (nb[14] L13-14)",
        all(app2id[name] == i + 1 for i, name in enumerate(sorted(app2id))),
    )
    check("C4b stored dict order equals id order", list(app2id.values()) == sorted(app2id.values()))

    sd = ckpt["model_state_dict"]
    shapes = {k: tuple(v.shape) for k, v in sd.items()}
    expected_shapes = {
        "app_emb.weight": (VOCAB_SIZE, D_MODEL),
        "pos_emb.weight": (WINDOW, D_MODEL),
        "out.weight": (VOCAB_SIZE, D_MODEL),
        "out.bias": (VOCAB_SIZE,),
    }
    time_keys = [k for k in sd if k.split(".")[0] in ("time_proj", "fuse_mlp", "gate_net", "time_proj_g")]
    check(
        "C5 tensor shapes; no time-feature parameters (morph_mode='none')",
        all(shapes.get(k) == s for k, s in expected_shapes.items()) and not time_keys,
        f"{len(sd)} tensors, {sum(v.numel() for v in sd.values()):,} parameters, "
        f"time-feature keys: {time_keys or 'none'}",
    )
    check("C5b all parameters float32", all(v.dtype == torch.float32 for v in sd.values()))
    check(
        "P1 padding embedding row 0 is exactly zero (nn.Embedding padding_idx=0, nb[17] L15)",
        bool(torch.all(sd["app_emb.weight"][PAD_ID] == 0)),
    )

    print("\nL. Layer 2 artifact (numpy-only unpickler)")
    with open(args.layer2, "rb") as f:
        l2 = NumpyOnlyUnpickler(f).load()
    l2_app2id, l2_id2app = l2["app2id"], l2["id2app"]
    check("C6 Layer 2 app2id equals checkpoint app2id", l2_app2id == app2id)
    check("C6b Layer 2 app2id order equals checkpoint order", list(l2_app2id.items()) == list(app2id.items()))
    check(
        "C6c Layer 2 id2app = inverse of app2id plus 0 -> '<PAD>'",
        l2_id2app == {**{v: k for k, v in app2id.items()}, PAD_ID: PAD_TOKEN} and len(l2_id2app) == VOCAB_SIZE,
    )

    target = np.asarray(l2["target_app_id"])
    event_index = np.asarray(l2["event_index"])
    top_l1, top_l2 = np.asarray(l2["top_l1_ids"]), np.asarray(l2["top_l2_ids"])
    contexts = l2["context_app_ids"]
    users = np.asarray(l2["user_id"])
    check(
        "L1 scale: events, users, top-N",
        len(target) == N_EVENTS and len(np.unique(users)) == N_USERS
        and top_l1.shape == (N_EVENTS, TOPN) and top_l2.shape == (N_EVENTS, TOPN) and len(contexts) == N_EVENTS,
        f"events={len(target)} users={len(np.unique(users))} top_l1={top_l1.shape} top_l2={top_l2.shape}",
    )
    in_range = lambda a: bool(((a >= 1) & (a <= N_APPS)).all())
    check("L2 target ids within 1..87 (padding never a target)", in_range(target),
          f"range [{target.min()}, {target.max()}]")
    check("L3 top-20 ids within 1..87 (padding never selected, T13)", in_range(top_l1) and in_range(top_l2))
    check(
        "L4 context ids within 1..87 (no padding inside online contexts, T8)",
        all(1 <= a <= N_APPS for ctx in contexts for a in ctx),
    )
    check(
        "L5 context length == min(event_index, 20) (T7)",
        all(len(ctx) == min(int(i), WINDOW) for ctx, i in zip(contexts, event_index)),
    )
    # Reported, not gating: a property of the offline sequences, not of the vocabulary.
    dup_users = [u for ctx, t, u in zip(contexts, target, users) if ctx[-1] == t]
    print(f"  [NOTE] F1 target equals the last context app in {len(dup_users)} of {len(target)} events "
          f"({100 * len(dup_users) / len(target):.2f}%, {len(set(dup_users))} users): the per-user "
          "re-sort reorders same-second ties (spec section 5.4, F1). Does not affect the vocabulary.")

    print("\nX. Cross-check with the public-LSApp re-run (spec appendix, non-authoritative)")
    id2app = {v: k for k, v in app2id.items()}
    appendix = appendix_vocabulary()
    diffs = sorted(i for i in set(id2app) | set(appendix) if id2app.get(i) != appendix.get(i))
    check("C7 checkpoint vocabulary equals spec appendix", not diffs,
          "identical" if not diffs else f"differs at ids {diffs}")

    if failures:
        sys.exit(f"\nABORT: {len(failures)} check(s) failed: {failures}. Nothing written.")

    vocabulary = {
        "format": "adapreload-lsapp-vocabulary",
        "format_version": 1,
        "description": (
            "Authoritative app vocabulary of the frozen AdaPreload Layer 1 checkpoint. "
            "Package-independent: maps checkpoint app ids to LSApp display names only. "
            "Generated by tools/audit_vocabulary.py; do not edit by hand."
        ),
        "source": {
            "checkpoint": {
                "file": "layer1_backbone_final.pt",
                "sha256": ck_sha,
                "bytes": Path(args.checkpoint).stat().st_size,
                "key": "app2id",
            },
            "cross_checked_with": {
                "file": "layer2_probability_outputs.pkl",
                "sha256": l2_sha,
                "bytes": Path(args.layer2).stat().st_size,
                "keys": ["app2id", "id2app"],
            },
            "notebook": {"file": "Copy_of_adaPrealoaderneww.ipynb", "sha256": NOTEBOOK_SHA256},
            "spec": "docs/ANDROID_TRACE_SPEC.md section 5",
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
        "model": {
            "vocab_size": ckpt["vocab_size"],
            "window": ckpt["window"],
            "d_model": ckpt["d_model"],
            "morph_mode": ckpt["morph_mode"],
        },
        "padding": {"id": PAD_ID, "token": PAD_TOKEN},
        "id_assignment": "id = 1 + index of the name in Python code-point sorted order (notebook nb[14] L13-14)",
        "app_count": len(id2app),
        "apps": [{"id": i, "name": id2app[i]} for i in sorted(id2app)],
        "apps_sha256": vocabulary_sha256(id2app),
        "apps_sha256_definition": "SHA-256 of UTF-8 text with one line per app, ascending id: '<id>\\t<name>\\n'",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(vocabulary, indent=2, ensure_ascii=True) + "\n", encoding="ascii")

    # Round trip: the written file must reproduce the checkpoint mapping exactly.
    written = json.loads(out.read_text(encoding="ascii"))
    round_trip = {a["name"]: a["id"] for a in written["apps"]}
    check("W1 written file round-trips to the checkpoint app2id", round_trip == app2id)
    check("W2 apps_sha256 recomputed from the written file",
          vocabulary_sha256({a["id"]: a["name"] for a in written["apps"]}) == written["apps_sha256"])
    if failures:
        out.unlink()
        sys.exit("ABORT: written file failed verification and was removed.")
    print(f"\nWrote {out.relative_to(REPO) if out.is_relative_to(REPO) else out} "
          f"({out.stat().st_size} bytes, sha256 {sha256_of(out)})")
    print(f"apps_sha256 {vocabulary['apps_sha256']}")


if __name__ == "__main__":
    main()
