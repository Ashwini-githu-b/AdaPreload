# ANDROID_TRACE_SPEC

The rules a real-device AdaPreload trace must follow so that it stays comparable with the
offline LSApp experiment. The offline notebook is the source of truth; this file only
restates it and marks where Android cannot reproduce it without a methodological decision.

## Provenance

| Item | Value |
|---|---|
| Notebook | `Copy_of_adaPrealoaderneww.ipynb`, 45 cells |
| Notebook SHA-256 | `47540816a02d867a032865c903ab1a2489d06d0700c86dc216feb603867fc523` |
| Layer 1 checkpoint (`layer1_backbone_final.pt`) | SHA-256 `fd668f160363c9b32fe5ac561bec7b46ff5492d84e10f31d0bfd6d7a0c21c092` (nb[43] output) |
| Layer 2 artifact (`layer2_probability_outputs.pkl`) | SHA-256 `bfe1e61f1c0d4a90b90dc3623db0e13887915ec05b300f5ac2128c2b8535a2c8` |
| Layer 3 artifact (`layer3_cached_distributions.pkl`) | SHA-256 `1580e82d48d1b4eed3ba06996ae0757982b4f2587054401fd33596d449e566fd` |
| Reproduction check | Re-running nb[3], nb[8] and nb[14] L13–15 unchanged on the public LSApp file (nb[1]) reproduces the notebook's printed counts: 1,673,261 launches, 213,496 retained launches, `VOCAB_SIZE` 88, and 19,121 same-second pairs. Vocabulary examples below come from this re-run; the checkpoint remains authoritative. |

**Citation key.** `nb[i] Lx–y` = notebook cell at 0-based position `i`, source lines `x–y`
(line 1 is the cell's `# @title` line). Cells cited:

| nb[i] | Header label | Colab cell id |
|---|---|---|
| nb[2] | CELL 3 — Load and verify schema | `QcwApx0rFZ20` |
| nb[3] | CELL 4 — Extract real launch events | `x2BElb2gGUBK` |
| nb[8] | CELL 9 — Collapse same-app repeat-launch noise | `PL3tVbdvNtlK` |
| nb[9] | CELL 10 — Re-check the rhythm signal on clean data | `69fBsiYdS75v` |
| nb[14] | CELL 15 — vocab/features | `QFUzXnWU03g_` |
| nb[15] | CELL 16 — User-level split | `VCjEHzI248Rk` |
| nb[16] | CELL 17 — Dataset + collate | `DWGukmx94-p7` |
| nb[17] | CELL 21 (revised) — BackboneMorph | `wJLoTeUS5H7J` |
| nb[20] | CELL 24 — Train and save the FINAL Layer 1 backbone | `lmc8b8D7MppR` |
| nb[21] | CELL 25 — Load the frozen Layer 1 backbone | `k4ruD_DG7lL6` |
| nb[22] | CELL 26 — Layer 2: per-user online adapter | `eJFtRNnRnHmj` |
| nb[33] | CELL 37 — L2 probability-output extraction | `SFckgPBisez9` |
| nb[34] | CELL 38 — LAYER 3 setup | `W4bS8Vbw9k0P` |
| nb[35] | CELL 39 — LAYER 3 policy evaluation | `ckTefY0G9mbT` |
| nb[39] | CELL 43 — LAYER 3 artifacts + research summary (duplicated as nb[40]) | `xlSR2Naq9u4f` |
| nb[41] | CELL 44 — FREEZE step 1 | `vM_VUEi290uQ` |
| nb[42] | CELL 45 — FREEZE step 2 | `miDoxNKjSyys` |
| nb[43] | CELL 46 — FREEZE step 3 | `QO-QPPEeS1xV` |

**Terms.** A *launch* is a raw `Opened` record. A *retained launch* is a launch that survives
the duplicate collapse (T3). An *event* is a retained launch that serves as a prediction
target (T9).

---

## Part A — Exactly reproducible rules

These follow from the notebook alone. Android must implement them as written.

**T1. Launch definition.** A launch is a record whose `event_type == 'Opened'`. Records of
type `Closed`, `User Interaction` and `Broken` are never launches.
Fields used afterwards: `user_id`, `timestamp`, `app_name`. `session_id` is never used.
*Source:* nb[3] L9 (filter), L4–8 (rationale); nb[2] L5–11 (schema).

**T2. Ordering.** Each user's launches are processed in ascending `timestamp` order.
*Source:* nb[3] L11; nb[8] L15; nb[22] L20; nb[33] L121.

**T3. Consecutive-duplicate collapse.** In a user's ordered launch sequence, every run of
consecutive launches with the same `app_name` becomes a single retained launch. It keeps the
run's `app_name` and the `timestamp` of the run's first launch.
- No time threshold: runs merge regardless of the gap between launches.
- No session boundary: runs merge across `session_id` changes.
- Applies across the user's whole history.

Online equivalent: drop a launch if its app equals the app of the most recent retained
launch. A dropped launch creates no event, no score and no update, and does not change the
retained launch's timestamp.
*Source:* nb[8] L8–16. Reproduced on public LSApp with this exact code: 1,673,261 → 213,496
launches; 24,133 retained launches span more than one `session_id`; the longest merged gap is
8,284,861 s (about 96 days).

**T4. No session segmentation.** Each user is one continuous sequence, from their first launch
to their last. It is never split or reset by sessions, time gaps, days, screen-off or reboots.
*Source:* nb[8] L8–16; nb[15] L19–29; nb[22] L20–35.

**T5. Vocabulary.**
- The vocabulary is the 87 LSApp `app_name` display strings, sorted in Python
  code-point order. Ids are 1..87; id 0 is padding; `VOCAB_SIZE = 88`.
- **Authoritative mapping:** `ckpt['app2id']` in `layer1_backbone_final.pt`. Android loads it
  and never rebuilds it.
- Lookup is an exact string match, Unicode included. For example, `S’more` uses U+2019, and
  `eBay` and `imo` sort last because they start with a lowercase letter.

*Source:* nb[14] L13–15 (construction), L18 (lookup); nb[20] L28–29 (persisted);
nb[33] L29–31 (`id2app`, `id2app[0] = '<PAD>'`).

**T6. No OOV path offline.** Every offline launch is in the vocabulary by construction, and
this is asserted. The embedding has exactly 88 rows (padding + 87 apps) and no unknown-app
token. Android handling of unknown apps is decision A3.
*Source:* nb[33] L45; nb[17] L15.

**T7. Context window.** `WINDOW = 20` (also stored as `ckpt['window']`). For event index
`i ≥ 1`, the context is `app_ids[max(0, i−20) : i]`: the most recent ≤20 retained launches
before `i`, oldest first. Length `L = min(i, 20)`.
*Source:* nb[15] L15, L27–29; nb[20] L30; nb[22] L35–37; nb[33] L28, L135–137.

**T8. No padding at inference; right-aligned positions.**
- The online context holds exactly `L` real ids, and the key-padding mask is all `False`.
- Position ids are `arange(20 − L, 20)`, so the most recent launch is always position 19.
- The hidden state is the encoder output at the last position.
- Left-padding plus a mask is used only to batch training data.

*Source:* nb[22] L42 (mask), L13–17 (positions, last index); nb[17] L45; nb[16] L14–30
(training-only padding).

**T9. Target.**
- The target for event `i` is `app_ids[i]`, the next retained launch. Every retained launch
  except the user's first is a target.
- The first retained launch is context only. No event exists until a user has ≥2 retained
  launches.
- The prediction for `i` uses only launches before `i`, so it can be computed as soon as
  launch `i−1` is retained.
- Because of T3, a target is never the same app as the last context launch.

*Source:* nb[15] L25–29; nb[22] L25–26, L35, L43.

**T10. Time features exist but do not affect the model.**
- The notebook computes:
  - `Δt` = seconds since the previous *retained* launch (0 for the first);
  - `log_dt = log1p(max(Δt, 0))`;
  - `hour = timestamp.hour`;
  - `hour_sin`/`hour_cos = sin`/`cos(2π·hour/24)`.
- The frozen backbone uses `morph_mode='none'`. In that mode, `pos_component` returns only
  the position embedding, and `forward` adds a time projection only in `side_channel` mode.
  The Layer 2 path (`get_hidden`) uses the same `pos_component`.
- Therefore `log_dt`, `hour_sin` and `hour_cos` have **no effect** on Layer 1 or Layer 2
  outputs. The model input is the app-id window only, and timestamps matter only for
  ordering (T2).
- There are no other context features: no day of week, location or device state.

*Source:* nb[14] L19–24 (features); nb[20] L8, L32 (`morph_mode='none'`); nb[17] L32–33,
L45–49; nb[22] L14.

**T11. Layer 1 inference.**
- The only preprocessing is T5 → T7 → T8: id lookup, window slice, tensor.
- The backbone is frozen, runs in eval mode (dropout off), and runs without gradients.

*Source:* nb[21] L8–10; nb[22] L46–48.

**T12. Prequential order and Layer 2 update** (per event, per user):
1. Build the context (T7).
2. Compute the Layer 1 hidden state and logits.
3. Compute `adapter_logits = W·h + b`.
4. `final = L1_logits + adapter_logits`.
5. Select the top-K set and score it against the target.
6. Compute the cross-entropy loss of `final` against the target.
7. Take **one** SGD step.

Details:
- The adapter is `Linear(64 → 88)`.
- `W` and `b` are **zero** at the start of each user's sequence.
- Plain SGD with `lr = 0.001`: no momentum and no weight decay.
- The loss covers all 88 logits. **The padding logit is not masked in the loss.**

*Source:* nb[22] L28–32, L35–71; nb[33] L127–130, L145–149, L173–177; nb[39] L39.

**T13. Padding excluded from decisions.** The preload ranking uses
`softmax(final with logit[0] = −∞)` over ids 1..87, and the returned indices are shifted
by +1. Verified offline: padding never appears in a preload set.
*Source:* nb[33] L152–158, L163–166; nb[41] L48–54.

**T14. Decision set.**
- The preload set at event `i` is the top-K of the T13 distribution.
- The offline pipeline stores the top-20 ids and probabilities per event, and derives every
  K ∈ {1, 2, 4, 5, 8, 16} as a prefix. Coverage = `1[target ∈ top-K]`.
- The same masked top-20 is also stored for Layer 1 alone, for comparison.
- K = 5 is the largest *observed* Δcoverage. The notebook calls it descriptive, not optimal
  (the K=4 and K=5 confidence intervals overlap).

*Source:* nb[33] L132, L153–166; nb[34] L14; nb[35] L35–36; nb[42] L196–209; nb[43] L9–11.

**T15. Per-event record.** To run the offline Layer 3 analysis unchanged on device data,
each event record mirrors the Layer 2 artifact row:
`user_id, event_index, context_app_ids, target_app_id, top_l1_ids[20], top_l1_probs[20],
top_l2_ids[20], top_l2_probs[20]`.
Ids are in vocabulary space (padding excluded); probabilities are float32.
*Source:* nb[33] L160–167, L192–203.

**T16. Aggregation.** Macro = per-user mean, then the mean over users. One device is one
user, so macro equals micro for that device.
*Source:* nb[35] L14–18.

---

## Part B — Android adaptations that require a methodological decision

UsageEvents cannot reproduce these from the notebook alone. Each one stays **open** until a
decision is recorded here. No Android equivalent is assumed.

**A1. What counts as `Opened`.**
- The notebook treats `Opened` as an LSApp label only; how LSApp's collector produced it is
  not in the notebook (nb[3] L4–9).
- Raw LSApp contains bursts of `Opened` for the same app about 1 s apart (nb[2] output).
- Because T3 merges same-app runs with no time limit, the *granularity* of same-app repeats
  cannot change the retained sequence. What changes it is which *other* packages are counted
  between them (A4) and how they map (A2).
- *Decision:* which UsageEvents type(s) define a launch.

**A2. Package name → LSApp `app_name`.**
- The model's identifiers are LSApp display names (T5); UsageEvents give package names.
  No mapping exists in the notebook.
- Hard cases in the vocabulary:
  - `Phone` vs `Android In Call UI` (often the same package);
  - `Messages` / `Messaging` / `Verizon Messages`;
  - `Google` / `Google Chrome` / `Microsoft Bing Search`;
  - `Telegram` / `Telegram X`;
  - Samsung-only entries (`Samsung Email`, `Samsung Gallery`, `Samsung Internet Browser`,
    `Samsung Notes`, `Samsung Pay`).
- T3 runs on the *mapped* identifier (nb[8] L8–9), so two packages mapped to one name collapse
  together.
- *Decision:* the mapping table, including many-to-one and one-to-many cases.

**A3. Apps outside the vocabulary (OOV).**
- The notebook has no OOV handling (T6).
- Any OOV treatment changes the retained sequence. For example, `A, OOV, A` becomes `A` only
  if OOV launches are removed before T3. It also changes the event count.
- The model has no unknown-app token.
- *Decision:* the OOV policy, and whether it applies before or after T3.

**A4. Non-app foreground packages:** home launcher, SystemUI, keyboard, permission dialogs,
share sheet, and AdaPreload itself.
- The notebook does not address them.
- Evidence: the 87-name vocabulary has no launcher, SystemUI or keyboard entry, but it does
  include `Settings` and `Android In Call UI`.
- Excluding the launcher makes `A → home → A` collapse to `A` under T3.
- *Decision:* the exclusion set, and whether exclusion happens before T3.

**A5. Timestamp source, precision and timezone.**
- LSApp timestamps are naive strings at 1 s resolution with no stated timezone (nb[2] output;
  nb[3] L10). UsageEvents give epoch milliseconds.
- By T10 this affects only ordering and any exported time features.
- 9.0% of retained LSApp launches share their second with the previous launch (nb[9] output).
- *Decision:* the precision and timezone for exported timestamps and `hour`.

**A6. Sequence start (cold start).**
- Offline, each user starts at their first record with a zero adapter (nb[22] L20, L28–30).
- *Decision:* whether the device sequence starts at logging start, or is seeded with the
  pre-install history that UsageEvents still retains. Seeding changes how comparable the
  cold start is.

**A7. Observation gaps.**
- The offline data has no notion of a missing period.
- Launches that happen while the service is down can be recovered from UsageEvents later.
- *Decision:* run them through T12 (flagged as backfilled), add them as context only, or drop
  them. Each option changes the event count or the adapter's trajectory.

**A8. User unit.**
- Offline, the unit is `user_id`, with one adapter per user (nb[22] L19–31).
- *Decision:* treatment of multiple Android users or a work profile on one device.

**A9. Live K.**
- The notebook does not fix a deployment K (T14).
- Logging the top-20 per event (T15) keeps every K computable after the fact.
- *Decision:* only needed if a single live K must be acted on.

**A10. Preload action.**
- The notebook's planned Android stage was soft preload via `startActivity → moveTaskToBack`,
  and it explicitly says this is *not* equivalent to AppFlow-style scheduling
  (nb[39] L107–111).
- Locked project decision: **shadow mode**. Android logs decisions (T14/T15) and performs no
  preload action.
