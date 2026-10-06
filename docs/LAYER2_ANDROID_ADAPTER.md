# Layer 2 Android adapter (Phase D: core)

Phase D adds the frozen Layer 2 adapter as a standalone Kotlin component:
- it is **not** connected to the UsageEvents trace;
- it does **not** persist its state;
- it never sees LSApp users' histories or another user's adapter state.

On the device it will start from zero and learn only from the owner's own launches. Phase C (Layer
1) is unchanged; its parity results after this change are identical (see below).

## Frozen specification

| Item | Value | Notebook |
|---|---|---|
| Adapter | `nn.Linear(64, 88)`: W 88 × 64, b 88, FP32 | nb[22] L28; nb[33] L127 |
| Initial state | W = 0, b = 0 | nb[22] L29-30; nb[33] L128 |
| Optimizer | `torch.optim.SGD(lr=0.001)`: no momentum, dampening, weight decay or Nesterov (PyTorch defaults, asserted by the fixture script) | nb[22] L31; nb[33] L120, L129 |
| Loss | `nn.CrossEntropyLoss()`: mean over a batch of 1, no class weights, no label smoothing (defaults, asserted) | nb[22] L32; nb[33] L130 |
| Output | `final_logits = backbone_logits + adapter(hidden)` | nb[22] L50-51; nb[33] L148-149 |
| Update | `zero_grad(); loss(final_logits, target).backward(); step()`, once per observed launch | nb[22] L68-71; nb[33] L174-177 |
| Order | ranking/probabilities are taken from `final_logits` *before* the update | nb[22] L54-66 before L68; nb[33] L151-171 before L173 |
| Layer 1 | frozen; `hidden` is detached, so only the adapter learns | nb[33] L146 |

There is no replay buffer, no drift trigger and no alternative algorithm.

## How the hidden vector is exposed

`Layer1Prediction` now carries `hidden`: the 64-value encoder output at the last real token. This
is the vector `predict` already computed and fed to the output layer (`last`), and it is the
notebook's `get_hidden` (nb[22] L11-17).

The change is four lines in `model/Layer1Model.kt`:
- a new constructor field;
- passing `last` into the result;
- the KDoc.

The forward pass is not duplicated and no arithmetic changed. The Phase C tests pass unchanged with
bit-identical reported errors.

## State and computation (`model/Layer2Adapter.kt`)

**State:**
- `weight: FloatArray(88 × 64)`, row-major like `nn.Linear.weight` (`W[j][k]` at `j × 64 + k`);
- `bias: FloatArray(88)`;
- `updateCount: Long`.

Everything starts at +0.0f and 0. Index j is the app id; row 0 is padding.

**Inputs:** Layer 2 runs in FP32 like the notebook. `Layer1Prediction.hiddenFp32()` and
`logitsFp32()` round Layer 1's double outputs to float once.

**Equations.** For hidden h (64), backbone logits z (88) and target y:

```
a_j  = fp32( b_j + Σ_k W_jk · h_k )            adapter logits, summed in double
f_j  = fp32( z_j + a_j )                       final logits
p    = softmax(f) over all 88 outputs          double, StrictMath.exp
loss = log Σ_j exp(f_j) − f_y                  returned, double
g_j  = fp32( p_j − 1[j = y] )                  the cross-entropy gradient wrt f_j
W_jk ← fp32( W_jk − lr · g_j · h_k )           for all j, k
b_j  ← fp32( b_j − lr · g_j )                  for all j
updateCount += 1
```

**Precision:**
- `lr` is `0.001f` (0.0010000000474974513), the value PyTorch applies to FP32 parameters.
- Each stored value is computed in double from FP32 operands and rounded to FP32 once.
- Rounding g to FP32 copies PyTorch, which holds the gradient in an FP32 tensor. Removing that
  rounding still passes the tests: its effect is below FP32 resolution.

The fixture script checks that PyTorch's actual gradient matches these equations:
`bias.grad == softmax(final) − onehot(target)` within 1e-6 (fp32) and 1e-14 (fp64). The notebook
implies no difference from the specified update.

**Ordering (`Layer2OnlineLearner`):**
- `predict(hidden, backboneLogits)` computes the final logits with the current state. It records
  the prediction together with its exact inputs, and returns a `Layer2Prediction`.
- `reveal(target)` applies exactly one `update` with those stored inputs, then clears the pending
  prediction.
- A second `predict` before `reveal`, or a `reveal` with nothing pending, throws.
- `predict` has no target parameter. So prediction t can only reflect targets 1..t − 1
  (`Layer2Prediction.updatesBefore == t − 1`), and its logits array is never changed by later
  updates.

`reset()` sets W and b back to +0.0f, `updateCount` to 0, and drops any pending prediction.

## Padding (id 0)

| Where | Treatment | Why |
|---|---|---|
| Observed target | `require(target in 1..87)`. Ids 0, −1 and 88 throw `IllegalArgumentException` before any state changes, and a pending prediction stays pending. | Padding is never a launch. Phase B only emits ids 1..87. |
| Loss and gradient | **Unmasked**: softmax over all 88 outputs, as offline. The padding output gets gradient p₀ > 0, so W[0] and b[0] move. b[0] is −4.05e-9 after the first fixture step and −4.85e-7 after 30. | The parity requirement: `CrossEntropyLoss` in nb[22]/nb[33] is unmasked. Masking it was tested as a mutation and fails D2-D4. |
| Ranking and probabilities | **Masked**: `Layer2Prediction.probabilities` is the softmax over app ids 1..87 only (87 values; `probabilities[k]` is app id k + 1). `rankedAppIds()` never contains 0, whatever the padding logit. | nb[33] L152-156 (the Layer 2 artifact's probabilities), and spec T13. |
| Layer 1 context | left padding only, as in Phase C | spec T7 |

Note: the nb[22] evaluation scored `final_logits.topk(5)` unmasked (L54-55). The device follows
nb[33] and T13 and masks padding. That matters only if padding outscored a real app.

## Golden fixture (`tools/make_layer2_golden.py` → `app/src/androidTest/assets/layer2_golden.json`)

**Method.**
1. Load the audited checkpoint (SHA-256 `fd668f16…c092`, `weights_only=True`, strict) into the
   verbatim notebook `BackboneMorph`.
2. Compute `hidden` and `backbone_logits` in float32 exactly as nb[33] L136-147 does.
3. Run the notebook's adapter statements unchanged (Linear, zeroed, `SGD(lr=0.001)`,
   `CrossEntropyLoss`, `zero_grad/backward/step`) in float32, and again in float64 to measure
   rounding.

**Inputs are synthetic.** App-id sequences come from `numpy.random.default_rng(42)`: ids 1..87,
no consecutive repeats. No LSApp user's history is used, and nothing from the fixture is loaded
into the app.

**Contents:**
- **one_step:** a 20-launch context; hidden[64]; backbone logits[88]; target 44; loss before the
  step 5.729690; W and b after the step; final logits for the same hidden after the step.
- **sequence:** 31 launches, which give 30 prequential steps (context = the last ≤ 20 launches).
  For each step: hidden, backbone logits, target, final logits *before* the update, and the loss.
  It also holds the final W and b and `update_count` 30.
- **provenance:** checkpoint and notebook hashes, the formulation citation, torch 2.14.1+cpu, the
  seed, and FP32 SHA-256s of W and b.
- **fp32_vs_fp64_rounding:** how far the FP32 reference is from the FP64 run:

  | Quantity | FP32 vs FP64 |
  |---|---|
  | W after 1 step | 2.7e-10 |
  | b after 1 step | 2.8e-11 |
  | Logits after 1 step | 2.2e-7 |
  | Loss, 1 step | 2.0e-7 |
  | W after 30 steps | 1.7e-9 |
  | b after 30 steps | 2.8e-10 |
  | Logits before update (sequence) | 3.8e-7 |
  | Loss (sequence) | 4.4e-7 |

The file is 501,999 bytes with SHA-256 `d298ffdb6f3a6624314c0d04f4751a48e8044d4a297dbb542853ce8d0ad98902`.
A second run produced byte-identical output. Every stored input is checked to be exactly
representable in FP32.

## Tests (`Layer2AdapterTest`, JVM)

**Tolerances.** Each is set from FP32 resolution and the reference's own FP32-vs-FP64 rounding
(about 6-10× it), not from the observed Kotlin error.

| Quantity | Tolerance | Basis |
|---|---|---|
| Loss, final logits | 5e-6 | values below 16 (FP32 ulp ≤ 9.5e-7); reference rounding ≤ 4.4e-7 |
| W, b after 1 step | 2e-9 | values below 3.9e-3 (ulp 2.3e-10); reference rounding 2.7e-10 |
| W, b after 30 steps | 1e-8 | values below 7.8e-3 (ulp 4.7e-10); reference rounding 1.7e-9 |
| Fixture inputs vs Android Layer 1 | 5e-5 | Phase C `FP32_LOGIT_TOLERANCE` (unchanged) |
| Zero-state probabilities vs Layer 1 | 5e-6 | Phase C `FP32_PROBABILITY_TOLERANCE` (unchanged) |

**Results (2026-10-06):** 7/7 tests pass.

| Test | What it asserts | Measured |
|---|---|---|
| Fixture provenance | checkpoint hash, lr 0.001, 30 steps, targets in 1..87 with none repeating the previous launch; the fixture's hidden and logits equal Android Layer 1 on the same 31 contexts | hidden 1.3e-6, logits 1.8e-6 |
| **D1** zero state | new adapter: W = b = +0.0f and `updateCount` 0. On 47 contexts (16 Phase C golden + 31 fixture), adapter logits = 0, final logits bit-equal to Layer 1's FP32 logits, full 87-app ranking identical to Layer 1, `updatesBefore` 0 | probabilities vs Layer 1: 7.1e-8 |
| **D2** one step | loss, W, b, and final logits for the same hidden after the step; `updateCount` 1; padding bias moves (b[0] < 0) | loss 2.0e-7, W 2.3e-10, b 7.3e-12, logits 6.0e-8 |
| **D3** ordering | for each of 30 steps: prediction t made with `updatesBefore` = t, matching PyTorch's final logits *before* update t; a second predict or reveal is refused; the recorded prediction is unchanged after later updates. At t = 0, 1, 14 and 28: two learners with the same history give a bit-identical prediction t whatever target t is revealed, and different predictions at t + 1. | logits 1.2e-7, loss 3.7e-7 |
| **D4** sequence | 30 updates against the final PyTorch W and b and every step's loss; `updateCount` 30; the learner's state is bit-equal to direct updates (no extra or missing updates) | W 4.7e-10, b 1.2e-10, loss 3.7e-7 |
| **D5** reset | after 30 steps plus a pending prediction, `reset()` gives W = b = +0.0f, `updateCount` 0 and nothing pending; `reveal` is refused; the next update is bit-identical to a new adapter's | exact |
| Padding | targets 0, −1 and 88 refused with no state change; probabilities and ranking unaffected by the padding logit (set to 100); wrong input sizes refused | exact |

**Sensitivity:**
- Each mutation below makes D2-D4 (or D3-D4 and the padding test) fail:
  - lr = 0.0011;
  - padding masked in the loss;
  - two updates per reveal;
  - bias not updated.
- Removing the FP32 rounding of the gradient passes, as expected: it is below FP32 resolution.

## Phase C regression

`Layer1ParityTest` passes 5/5 with exactly the values recorded in `LAYER1_ANDROID_INFERENCE.md`:

| Comparison | Result |
|---|---|
| vs fp64 | 6.661e-16 / logits 5.329e-15 |
| vs fp32 | 1.927e-7 / logits 1.966e-6 |
| vs Colab golden cases | 1.122e-7 |
| Golden cases top-1 / top-5 | 16/16, 16/16 |
| Colab sample (256 events) | 3.523e-7; top-1, top-5 and top-20 all 256/256 |

These are unchanged:
- the Layer 1 weights, manifest and checkpoint hash;
- the vocabulary and mapping;
- the Layer 1 golden files;
- the Phase C tolerances.

## Reproducing

```
python3 tools/make_layer2_golden.py layer1_backbone_final.pt
./gradlew :app:testDebugUnitTest --tests '*Layer2AdapterTest*' --tests '*Layer1ParityTest*'
```

## Not in this iteration (open before live integration)

1. **Persistence.**
   - W, b, `updateCount` and the pending prediction (its hidden and backbone logits) are in memory
     only.
   - They need to be stored atomically with the trace cursor and the appended records. Otherwise
     a crash between "record appended" and "adapter updated" would skip an update or apply it
     twice.
   - That also needs a format (raw FP32 little-endian plus a hash) and a version and checkpoint
     check on load.
2. **Live hook.** The planned point is `TraceRuntime` after `commitBatch`. For each newly APPENDED
   launch, in order:
   1. `reveal(appId)` for the pending prediction;
   2. Layer 1 on the last ≤ 20 retained ids;
   3. `predict` → pending.

   Still to decide:
   - COLLAPSED events cause no update (T3);
   - the first launch after the start has nothing to reveal;
   - what to do with several launches in one poll batch (process strictly in order);
   - the threading (the trace executor).
3. **Gaps and restarts.**
   - A7 says the state continues across gaps and service restarts, with no reset and no backfill.
   - Whether a prediction left pending across a gap is revealed by the next launch (as the
     continuous sequence implies) should be confirmed explicitly.
4. **On-device run of these tests.**
   - The adapter is pure Kotlin with `StrictMath`, so ART should give bit-identical results.
   - There is no instrumented Layer 2 test yet. The fixture is already in `androidTest/assets` for
     one.
5. **Logging and evaluation fields**: the prediction records and losses to export. This is Phase E
   scope, and not started.
