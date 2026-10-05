# Layer 1 Android inference (Phase C: numerical parity)

Phase C establishes that the frozen Layer 1 backbone running on Android computes the same outputs
as the Colab/PyTorch checkpoint. It is **not** connected to the live UsageEvents trace. Layer 2,
online personalization and preload are not implemented.

## Frozen inputs

| Item | Value |
|---|---|
| Checkpoint `layer1_backbone_final.pt` | SHA-256 `fd668f160363c9b32fe5ac561bec7b46ff5492d84e10f31d0bfd6d7a0c21c092` (audited, unchanged) |
| Layer 2 artifact `layer2_probability_outputs.pkl` | SHA-256 `bfe1e61f1c0d4a90b90dc3623db0e13887915ec05b300f5ac2128c2b8535a2c8`; holds the Colab Layer 1 top-20 outputs used as an independent reference |
| Vocabulary | `assets/lsapp_vocabulary.json`, `apps_sha256` `d0801f3b…7d78` (87 apps, padding id 0); equals the checkpoint `app2id` |
| Notebook | SHA-256 `47540816…c523` |

## Architecture (read from the instantiated model, not assumed)

The checkpoint loads with `strict=True` into the notebook's own `BackboneMorph` class. The class
(nb[17] L11-53) and `get_hidden` (nb[22] L11-17) are copied into `tools/export_layer1.py`; a text
diff against the notebook confirmed both copies are verbatim.

| Property | Value |
|---|---|
| `morph_mode` | `none`: there are no time-feature parameters, and time inputs change no output (verified: max difference 0.0 with random time features) |
| Embeddings | app embedding 88 × 64 (`padding_idx = 0`); position embedding 20 × 64 |
| Positions | right-aligned: for L real tokens, positions `20 − L … 19` |
| Encoder | 2 × `TransformerEncoderLayer`: post-norm (`norm_first = False`), 4 heads × 16, feed-forward 128 with ReLU, LayerNorm eps 1e-5, dropout inactive (eval); no final encoder norm |
| Readout | encoder output at the last real token → `Linear(64, 88)` → 88 logits |
| Probabilities | softmax with `logit[0] = −inf`; 87 values, `probabilities[k]` = app id `k + 1` (nb[33] L152-156) |
| Size | 28 tensors, 79,576 float32 parameters |

## Runtime: plain Kotlin, no conversion of the graph

**Chosen:** `model/Layer1Model.kt` evaluates the architecture above directly:
- it uses the exact float32 weights, converted to double;
- all arithmetic is in double precision;
- it uses `StrictMath` for `exp` and `sqrt`, so JVM and Android results are bit-identical.

**Not chosen:**
- **ONNX Runtime:** adds per-ABI native libraries for a 79,576-parameter model, and parity would
  depend on how the exporter translates `MultiheadAttention`.
- **LiteRT:** a PyTorch-to-LiteRT conversion toolchain is a second translation step that would
  itself need validating.

**Why plain Kotlin:**
- no operator translation at all;
- no native code;
- the parity test runs as an ordinary JVM unit test on exactly the code that ships;
- the work per prediction is tiny (at most 20 tokens).

**Padding.** Padding is dropped before the encoder. This is exactly equivalent to PyTorch's key
padding mask, because only real tokens are attended to and positions don't change. In fp64 the
left-padded 20-token PyTorch forward and the unpadded reference agree to 5.3e-15.

## Export (`tools/export_layer1.py`)

1. Verify the SHA-256 of the checkpoint and the Layer 2 artifact.
2. Load the checkpoint with `torch.load(weights_only=True)`, then `strict` into the verbatim class.
3. Write `app/src/main/assets/layer1/layer1_weights.bin`:
   - every tensor's float32 little-endian bytes, in `state_dict` order;
   - read back and compared bit-for-bit with the checkpoint;
   - SHA-256 `1676950a543242e8d62deab29cf757f50a9ad3eeb32fbb6a7d7dcdb43a17b99c`, 318,304 bytes.
4. Write `layer1_manifest.json`: architecture, tensor table (shape, offset, per-tensor SHA-256),
   and the source hashes.
5. Write the golden vectors to `app/src/androidTest/assets/`. They ship only in the test APK.

The output is deterministic: a second run produced byte-identical files. `Layer1Model.load` refuses
weights if any of these checks fails:
- weights-file or per-tensor hash;
- tensor names or shapes;
- the checkpoint hash;
- an architecture the implementation doesn't compute.

### Reference checks (export script, all 30,040 held-out events)

| Check | Result |
|---|---|
| Local PyTorch fp32 (torch 2.14.1) vs the frozen Colab top-20 Layer 1 probabilities | max abs difference 1.0e-6; top-1, top-5 and top-20 lists identical for 100% of events |
| PyTorch fp32 vs fp64 on the same weights | max 8.6e-7 (probabilities), 6.4e-6 (logits; logits span ±12.3) |
| Left-padded 20-token forward vs unpadded reference (fp64) | max 5.3e-15 (logits) |

## Golden vectors

- **`layer1_golden.json`: 16 cases.** Each stores the left-padded 20-token input, the 88 logits
  and 87 probabilities from PyTorch fp32 and fp64, the expected top-5, and the hashes. Inputs:
  - 9 held-out contexts from the Layer 2 artifact: lengths 1, 2, 5, 10 and 19 (padded), plus 4
    full-length contexts. These also carry the Colab top-20.
  - 7 synthetic contexts: a single id 1 or 87, the pair 87 → 1, ids 1-20, ids 68-87, id 45
    repeated, and 82/33 alternating.
- **`layer1_artifact_sample.json`: 256 held-out events** with the frozen Colab top-20 ids and
  probabilities. It holds the first 4 events of every length 1-19 plus 180 random full-length
  events (seed 42).

## Tolerances (`model/Layer1Parity.kt`)

Each tolerance is set from the numerics of what is being compared, not from the observed Kotlin
error.

| Comparison | Tolerance | Basis |
|---|---|---|
| vs PyTorch fp64 | 1e-13 (probabilities), 1e-12 (logits) | same math in double; only summation order differs, so errors are around 1e-15 |
| vs PyTorch fp32 | 5e-6 (probabilities), 5e-5 (logits) | the fp32 reference's own error vs fp64 is at most 8.6e-7 / 6.4e-6 over all held-out events; about 6-8× that |
| vs Colab fp32 top-20 | 5e-6 | same basis; the local fp32 reproduction matches Colab to 1.0e-6 |
| Probability sum | \|Σp − 1\| ≤ 1e-12 | double precision |
| Ranking | top-1 and ordered top-5 must match exactly | |

**Sensitivity check:** scaling attention by √64 instead of √16 makes the fp64 check fail with
an error of 0.098, so the test detects architectural mistakes.

## Parity results (2026-10-05, JVM unit test `Layer1ParityTest`)

| Metric | Result |
|---|---|
| Output shape | 87 probabilities (app ids 1..87), 88 logits |
| Probability sum | max \|Σp − 1\| = 1.0e-15 |
| vs PyTorch fp64 | max abs error 6.7e-16, mean 7.9e-18; logits 5.3e-15 |
| vs PyTorch fp32 | max abs error 1.9e-7, mean 2.0e-9; logits 2.0e-6 |
| vs Colab top-20 (9 artifact golden cases) | max abs error 1.1e-7 |
| Top-1 / top-5 (16 golden cases) | 16/16 and 16/16 |
| Colab sample (256 events) | max abs error 3.5e-7, mean 1.1e-8; top-1 256/256, top-5 256/256, top-20 256/256 |

On a device, `Layer1ParityInstrumentedTest` loads the model through `Layer1Assets` from the app
assets and applies the same comparisons and tolerances. Run it from Android Studio. It has been
compiled here but not yet run on a device; because the arithmetic uses `StrictMath`, its results
should be bit-identical to the JVM test.

## Reproducing

```
python3 tools/export_layer1.py layer1_backbone_final.pt layer2_probability_outputs.pkl
./gradlew :app:testDebugUnitTest --tests '*Layer1ParityTest*'
./gradlew :app:connectedDebugAndroidTest   # on a device
```

## Not in Phase C

- No link from the live trace to the model.
- No Layer 2 adapter or online personalization.
- No preload, no K selection, no prediction UI.
