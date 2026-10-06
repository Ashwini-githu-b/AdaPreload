# Layer 2 live integration (Phase D2)

Phase D2 connects the Phase D Layer 2 learner (`docs/LAYER2_ANDROID_ADAPTER.md`) to the Phase B
trace, and persists its state in the same SQLite transaction as the trace.

Unchanged:
- the Layer 2 equations, lr, initialization and padding behavior;
- Layer 1, the vocabulary and the mapping;
- the Phase B trace semantics. `TraceSession`, `TracePipeline` and the trace models are not
  modified.

Shadow mode as before: no preload, no `startActivity`, no measurement, no K selection.

## Prequential order per launch

Only an **APPENDED** record is a launch of the sequence. `LivePersonalizer.prepare` handles each one
in sequence order, as follows.

| Step | What happens | Code |
|---|---|---|
| 1 | Phase B decides the record is a new supported launch (APPENDED). COLLAPSED, DISCARDED (OOV, ambiguous, unsupported, excluded) and non-launch records are skipped. | `TracePipeline` (unchanged); `prepare` filters on `Outcome.APPENDED` |
| 2 | If a prediction is pending (made after the previous launch): score it against this launch (L1 and L2 rank, loss), then apply **one** SGD update with the prediction's stored inputs. The first launch has nothing pending, so there is no update. | `Layer2OnlineLearner.reveal` |
| 3 | Append the launch to the context: the last ≤ 20 launches (T7). | `context = (context + appId).takeLast(20)` |
| 4 | Run frozen Layer 1 on that context, left-padded. | `Layer1Model.predict` |
| 5 | Feed the FP32 hidden vector and logits to Layer 2. | `hiddenFp32()`, `logitsFp32()` |
| 6 | Record the Layer 2 prediction before any later update: its final logits, the L1 and L2 top-1, and `updateCount`. | `Layer2OnlineLearner.predict`, `Layer2LogEntry` |
| 7 | It becomes the pending prediction, together with its context, hidden vector, backbone logits and final logits. | `PendingPrediction` |
| 8 | The batch's records, the new cursor, the Layer 2 state and the log rows are committed in **one** SQLite transaction. | `PersonalizedTraceStore` → `SqliteTraceStore.commitBatch(…, layer2)` |

`predict` has no target parameter. A launch only ever reveals the prediction made *before* it
(position − 1). The prediction made from a context ending with launch p waits for launch p + 1.

## Transaction and persistence design

**One transaction:**
- `TraceSession.poll` calls `store.commitBatch(records, cursor, polledAt)` exactly as in Phase B.
- In D2 the store is `PersonalizedTraceStore`, a decorator over `SqliteTraceStore`. It:
  1. computes the batch's Layer 2 work on a **copy** of the learner (`prepare`);
  2. commits the events, cursor, last-poll time, Layer 2 state and log rows in the existing
     Phase B transaction (`SqliteTraceStore.inTransaction`);
  3. only then makes the copy current (`accept`).
- A failed commit rolls everything back and leaves the in-memory learner as it was. The retry
  queries the same range again, because the cursor did not move, and does the same work exactly
  once.
- A batch without an APPENDED launch commits exactly as in Phase B, and Layer 2 is not written.
  So the model is written once per batch that contains launches, not on every UsageEvent.

**Schema version 2.** Two tables are added. The Phase B tables are unchanged, and upgrading from
version 1 only creates the new tables.

| Table | Content |
|---|---|
| `layer2_state` | A single row (`id = 1`) holding the `Layer2StateCodec` blob. The blob contains: magic and format version 1; identity (checkpoint SHA-256, Layer 1 weights SHA-256, vocabulary `apps_sha256`, the raw bits of lr); W 88 × 64 and b 88 as raw FP32 bits; `updateCount`; and the pending prediction (position, context ids, hidden[64], backbone logits[88], final logits[88]). It ends with a SHA-256 over all of it. |
| `layer2_log` | One row per processed launch. The key is `sequence_position`, so a second prediction for the same launch is a constraint violation that rolls back the whole batch. Columns: app id; the revealed prediction's position, L1 rank, L2 rank and loss; `update_count`; and the L1 and L2 top-1 of the new prediction. |

**Checks on open (`LivePersonalizer.open`).** Any failure throws. The state is never repaired or
reset silently.
1. The blob's digest, magic and version are valid, with no trailing bytes.
2. The identity equals the running model: checkpoint, Layer 1 weights, vocabulary, lr.
3. The launches Layer 2 has processed (`pending.position + 1`) equal the trace's APPENDED count.
4. `updateCount` equals `pending.position`: every launch after the first revealed exactly one
   prediction.
5. The pending context equals the last ≤ 20 APPENDED ids in the events table.
6. Layer 1 recomputed on that context gives bit-identical hidden vector and logits.
7. The restored adapter gives bit-identical final logits for them.

**Fail closed.** If Layer 2 cannot be opened (the checks above, or missing or broken Layer 1
assets), `TraceRuntime` does not observe at all and shows `Not observing: Layer 2 cannot start: …`.
Unobserved time is an ordinary gap (A7), so the trace and the adapter can never diverge.

## Restart and gap semantics

| Case | Behavior |
|---|---|
| A. Continuous | Each 5 s poll commits its batch together with its Layer 2 work. Several launches in one batch are processed in sequence order. The result is bit-identical to one launch per batch (D7). |
| B. Service stop/start | The stop does a final poll (with Layer 2) and closes the window (`SERVICE_STOPPED`). The pending prediction stays pending. The next start reuses the in-process learner, which equals the committed state. |
| C. Process death | The new process re-opens Layer 2 from SQLite with the checks above. The old window is closed at its cursor (`PROCESS_ENDED`) and a new one opens now, so events after the last commit fall into the gap and are never read. Committed work is neither repeated nor lost; uncommitted work never existed. |
| D. Usage Access loss/recovery | The window closes (`USAGE_ACCESS_LOST`) and nothing is committed while access is off. On recovery a new window opens, and the next supported launch reveals the pending prediction. |
| E. Gap, then a launch | No reset and no backfill (A7). The first APPENDED launch after the gap reveals the pre-gap prediction. If it is the same app as the last pre-gap launch it collapses (T3 across gaps): no reveal, and the next different app reveals instead. |

## Spec conflict: an existing Phase B sequence

A6 and T12 say the adapter starts at zero **at the start of the sequence**. A database written by
Phase B already holds a sequence recorded without Layer 2. Taking it over would need a rule the
spec doesn't have: either replay those launches, or start the adapter mid-sequence.

Implemented behavior:
- The schema is upgraded and the trace is kept unchanged.
- If the trace already holds launches and there is no Layer 2 state, Layer 2 refuses to start and
  nothing is observed. The message reads "… launches recorded without Layer 2; Layer 2 must start
  at zero with the sequence (A6, T12)".
- A version 1 database with no launches starts normally.

To run D2 on the device that holds the Phase B trace:
1. export that trace;
2. clear the app's data;
3. start a new experiment.

Alternatively, a different rule for existing data would have to be decided first.

## Diagnostics (minimal)

- **logcat, tag `AdaPreloadLayer2`, one line per launch:**
  `launch #p app a: revealed #p−1 (L1 rank r1, L2 rank r2, loss x) | nothing to reveal; updates n; next L1 top-1 i, L2 top-1 j`.
  It also logs an open line with the update count and the pending position.
- **`layer2_log` table:** the same fields.
- **Trace card:** Layer 2 updates, the last launch processed, the revealed prediction's L1/L2
  rank, and the next app's L1 / L2 top-1.
- No per-weight logging. The JSON trace export is unchanged and does not include `layer2_log`.

## Tests

**JVM: `live/LivePersonalizationTest`, 10/10 pass.** The tests run the real `TraceSession`,
`TracePipeline` and Layer 1 model. `InMemoryLayer2Store` commits all-or-nothing like a SQLite
transaction, stores the state in codec form, and can fail a commit on demand. "Process death" is
a new learner and session over the same store. The reference is the prequential definition
computed directly with `Layer2OnlineLearner`.

| Test | Asserts |
|---|---|
| D6 | The first launch creates a pending prediction (position 0, context [a₀], equal to Layer 1's logits) and no update: W = b = +0.0f, `updateCount` 0. |
| D7 | The second launch reveals prediction 0, with exactly one update: bit-identical to `Layer2Adapter.update` with the stored inputs, the same loss, and both ranks. The one-batch run gives byte-identical state and log. |
| D8 | Collapsed duplicates (within a batch, across batches, and a collapse-only batch) cause no prediction, no update and no Layer 2 write. |
| D9 | OOV, home, IME, self, ambiguous, unsupported, system, non-launchable and non-launch events, and batches holding only those, leave the pending prediction unrevealed and the state byte-identical. The next supported launch reveals it. |
| D10 | Across a `SERVICE_STOPPED` or `USAGE_ACCESS_LOST` gap, gap events are never queried, and the pre-gap prediction is revealed by the first post-gap launch. A collapse across the gap does not reveal. Final W and b equal the reference. |
| D11 | A new process restores W, b, `updateCount`, the pending prediction and the last log entry bit-exactly. The codec round-trips byte-exactly. A damaged blob, or a foreign weights hash, is refused. |
| D12 | (a) A failed commit changes neither the trace nor the learner, and the retry processes each launch once. (b) Death after a commit resumes exactly. (c) Death after a failed commit: the uncommitted launches are neither in the trace nor applied, and they fall into the gap. (d) A Phase B-only trace, or a launch committed around Layer 2, is refused. (e) Re-processing a launch, or accepting a stale step, is refused. |
| D13 | For 25 launches in batches of 1-10: each prediction p used exactly p updates, and its L2 top-1 equals the reference's pre-update prediction. Each reveal is scored with that prediction's logits and loss. Every persisted pending prediction is the pre-update one. Final W and b are bit-identical to the reference. |
| D14 | Each launch reveals position − 1, never its own position. After each commit, the newest prediction (context ending with the newest launch) is still pending, with `updateCount` = its position. The wrong ordering (a launch updating the prediction made from itself) gives different W. |
| extra | `Layer2Adapter.copy` and `restore` are exact and independent; a wrong shape is refused. |

**Sensitivity:**
- A launch also updating its own prediction fails D6-D14 (9 tests).
- Accepting before the commit fails D12.
- Treating collapsed duplicates as launches fails D8-D10.

**SQLite: `Layer2SqliteStoreTest` (instrumented; compiled, not run here — needs a device).**
1. A new connection restores the state byte-exactly and continues.
2. A batch whose `layer2_log` insert violates the key rolls back its events, cursor and Layer 2
   state together.
3. A version 1 database with a recorded launch is upgraded with the trace untouched, and Layer 2
   refuses to take it over.

**Boundary:**
- The JVM tests prove the ordering, retry and restart logic, given an all-or-nothing commit.
- That SQLite provides the all-or-nothing commit is shown by test 2, and it will only be confirmed
  once that test runs on a device.
- An actual kill in the middle of a transaction is not simulated. Its outcome (committed or not)
  is exactly the D12 (b) or (c) case.

## Not in D2

- No on-device live run yet.
- The T15 per-event record (Layer 1 and Layer 2 top-20 ids and probabilities) is not logged; only
  top-1 and ranks are. It belongs to the Phase E evaluation.
- The JSON export does not include the Layer 2 log.
- A rule for databases that already hold a Phase B sequence (see the conflict above).
