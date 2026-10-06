# Phase E1: shadow preload policy

E1 adds a **decision step** after the live Layer 2 prediction. It decides which app AdaPreload
*would* preload, records that decision, and **preloads nothing**.
- No process, activity or service of another app is started, bound or warmed.
- No cached-process state is touched.
- No hidden API, root or ADB mechanism is used.

E1 makes **no claim** about launch latency, memory, CPU or battery.

## Phase D final status

**Implementation.** Phase D/D2 is complete (`bd2685d`, `f9edd0a`):
- frozen Layer 1;
- zero-initialized FP32 `Linear(64 → 88)` adapter, SGD lr 0.001;
- predict before update, one update per canonical launch;
- state persisted atomically with the trace.

**Manual validation on the Motorola edge 50 fusion,** as reported by the project lead:
- the live pipeline works;
- Layer 2 changes rankings;
- state and the pending prediction survive restarts;
- gaps reset nothing, and nothing is backfilled.

The Phase B trace was exported before the fresh D2 sequence started:
`adapreload_trace_20261006_144709.json`, SHA-256 `8d26988f…7f18`.

**Open closeout item.** No record shows that the instrumented tests (`Layer2SqliteStoreTest`) were
run on the device against `f9edd0a`. Command:
`./gradlew :app:connectedDebugAndroidTest -Pandroid.injected.androidTest.leaveApksInstalledAfterRun=true`.
- Without that flag, AGP uninstalls the app after the run, which deletes its data.
- Run it before installing an E1 build. E1 upgrades the database to version 3, and a D2 build cannot
  open a version 3 database (SQLite refuses to downgrade).

## Four separate things

| | Concept | Phase | Where |
|---|---|---|---|
| A | **Prediction:** the Layer 2 ranking of the next app, made after launch p | D2 | `LivePersonalizer` → `layer2_log` |
| B | **Preload decision:** whether AdaPreload *would* preload a ranked app now | **E1** | `ShadowPreloadPolicy` → `shadow_decisions` |
| C | **Preload execution:** actually warming the app | E2 (not implemented) | — |
| D | **User launch:** the next canonical launch (the prediction's target) | B/D2 | `events`; `shadow_decisions.actual_app_id` |

E1 implements A → B only. A row with `decision = PRELOAD` means the policy chose that app. It does
**not** mean the app was warmed, started, or made any faster.

## Data flow

```
UsageEvents → Phase B pipeline → APPENDED launch p
  → LivePersonalizer.prepare: reveal/update for p−1, Layer 1, Layer 2 prediction p   (A, unchanged)
  → ShadowPreloadPolicy.decide(prediction p)                                           (B, new)
  → one SQLite transaction: events, cursor, Layer 2 state + log, shadow decisions
  → learner.accept; logcat; status card
```

**Insertion point.** `PersonalizedTraceStore.commitBatch`, after `prepare` and before the commit.
The policy reads each finished prediction of the batch (`Step.predictions`) and returns data only.
- It never feeds back into Layer 2: the D2 state, log, trace records and cursor are byte-identical
  with or without the policy (test 11).
- If the policy throws, that batch's decisions are dropped and reported, and the Layer 2 work
  commits as before.

## Policy

**Input:**
- the prediction made after launch p (its position, and the current app = launch p's app);
- the Layer 2 probabilities: the padding-masked softmax over the 87 apps (T13). These are the
  values the offline Layer 3 analysis ranked.

**Ranking:** by descending probability; ties rank the lower id first (the Layer 2 rule).

**Examined:** the top `max(maxRank, maxCandidates)` apps. Each gets one decision; the first
matching rule wins:

| Order | Condition | Decision / reason |
|---|---|---|
| 1 | policy disabled | SKIP `policy_disabled` |
| 2 | rank > `maxRank` | SKIP `rank_out_of_range` |
| 3 | not eligible (below) | SKIP `excluded:<eligibility>` |
| 4 | probability < `minProbability` | SKIP `below_threshold` |
| 5 | `maxCandidates` already chosen | SKIP `candidate_limit` |
| 6 | otherwise | **PRELOAD** `selected` |

**Eligibility** (`MappingCandidateResolver`). It uses the existing mapping table and the Phase B
classifier, with no package vocabulary of its own:

| Eligibility | Meaning |
|---|---|
| `invalid_app_id` | not in 1..87 (padding is never a candidate) |
| `current_app` | the app of launch p (with `excludeCurrentApp`). By T3 the next canonical launch is never the same app, so preloading it is pointless. |
| `no_mapped_package` | no MAPPED row for the id. AMBIGUOUS and UNSUPPORTED rows carry no app id, so such apps (e.g. Google Chrome, Phone, Messages, Maps) are never candidates. |
| `not_installed` | none of its MAPPED packages has a launcher entry on this device |
| `multiple_packages` | more than one installed MAPPED package (e.g. both Amazon Shopping packages): the target is ambiguous |
| `excluded_self` / `excluded_home` / `excluded_ime` / `excluded_system` | the package classifies as AdaPreload, the default home app, an enabled input method, or the explicit system list (A4). The environment is the open observation window's snapshot, the same one the trace classifies with. |
| `not_supported` | any other non-SUPPORTED classification |
| `eligible` | exactly one installed MAPPED package that classifies as SUPPORTED |

The checks only query package metadata (`getLaunchIntentForPackage`, the home and IME lookups);
they never start anything.

## Configuration and defaults (`ShadowPolicyConfig`, version `e1-shadow-1`)

| Parameter | Default | Note |
|---|---|---|
| `enabled` | `true` | When false, every examined app is SKIP `policy_disabled` |
| `maxCandidates` | **1** | Preload count is a parameter of its own, separate from prediction K |
| `maxRank` | **1** | Only the Layer 2 top-1 is considered; an excluded top-1 is not replaced by rank 2 |
| `minProbability` | **0.5** | Engineering starting point, **not** a tuned research result (see below) |
| `excludeCurrentApp` | `true` | See `current_app` |

The defaults are set in `TraceRuntime.SHADOW_POLICY` and changing them needs a rebuild. Every
decision row stores the configuration it was made with.

**Why 0.5.**
- The notebook defines no probability threshold. Its Layer 3 analysis uses fixed capacities K and
  states that the probability mass is uncalibrated (nb[39] L90-91).
- K = 4-5 being the strongest prediction region offline is **not** a reason to preload 4-5 apps,
  so preload count stays at 1.

For context only, here is a descriptive read-only summary of the frozen offline Layer 2 outputs
(`layer2_probability_outputs.pkl`; 30,040 held-out events, 59 LSApp users). "Precision" is how
often the top-1 app was the next launch when the gate passed:

| Top-1 p ≥ | Gate passes | Precision (micro) | Precision (macro) | Top-1 hits / all events |
|---|---|---|---|---|
| 0.0 | 100.0 % | 0.514 | 0.466 | 0.514 |
| 0.3 | 91.8 % | 0.539 | 0.490 | 0.495 |
| 0.4 | 77.9 % | 0.580 | 0.524 | 0.451 |
| **0.5** | **61.6 %** | **0.631** | **0.558** | **0.389** |
| 0.6 | 46.4 % | 0.687 | 0.603 | 0.318 |
| 0.7 | 31.8 % | 0.753 | 0.654 | 0.240 |
| 0.8 | 18.2 % | 0.824 | 0.715 | 0.150 |

**What the table shows:** the probability is informative, because precision rises with the
threshold. 0.5 is a round value where a single preload is right more often than wrong offline,
while still firing on most events.

**What it does not show:**
- It is not an optimum, and it is not utility: the offline lr was tuned on these same users, the
  probabilities are uncalibrated, and the device user is a different person.
- It assigns no cost to a wrong preload.

## Storage, logs and status

**SQLite schema version 3** adds the `shadow_decisions` table. The migration from version 2 (or 1)
only creates tables; existing rows are untouched.
- Key: `(prediction_position, rank)`, so a decision can never be made twice. A duplicate rolls back
  the whole batch.
- Columns:
  - `app_id`, `package`, `probability`;
  - `eligibility`, `decision`, `reason`;
  - `current_app_id`;
  - `decided_at_ms`: the poll that made the prediction;
  - `actual_app_id`;
  - `policy_version`, `policy_enabled`, `min_probability`, `max_candidates`, `max_rank`,
    `exclude_current_app`.
- `actual_app_id` is set in the transaction of the launch that reveals the prediction. It is the
  user's actual next canonical launch. It stays null while the prediction is pending.
- `layer2_log` and every D2 table are unchanged.

**logcat, tag `AdaPreloadShadow`, one line per decision** (format only; the values are illustrative, not device output):
```
prediction #12 (after app 82): rank 1 app 33 Instagram [com.instagram.android] p=0.623 -> PRELOAD; actual preload: NONE (shadow mode)
prediction #13 (after app 33): rank 1 app 25 Google Chrome [-] p=0.710 -> SKIP reason=excluded:no_mapped_package; actual preload: NONE (shadow mode)
prediction #14 (after app 85): rank 1 app 82 WhatsApp Messenger [com.whatsapp] p=0.311 -> SKIP reason=below_threshold; actual preload: NONE (shadow mode)
```
The `AdaPreloadLayer2` lines are unchanged: `launch #N … revealed #N−1 … updates N`.

**Trace card:** two rows, "Shadow preload policy" (on/off with its parameters) and "Last shadow
decision" (`#p app, rank r, p=…: PRELOAD (not executed)` or `SKIP (reason)`).

## Tests

**JVM: 16 new tests, plus all earlier tests unchanged.**
- `shadow/ShadowPreloadPolicyTest` (12):
  1. disabled → SKIP;
  2. rank-1 above the threshold → PRELOAD;
  3. below the threshold → SKIP, and the threshold is inclusive;
  4. rank outside `maxRank` → SKIP `rank_out_of_range`, and an excluded top-1 is not replaced;
  5. `maxCandidates` is respected, and an excluded app uses no slot;
  6. AdaPreload is excluded: the mapping can't map it, and its package classifies as SELF;
  7. home and IME are excluded;
  8. unmapped (Google Chrome), invalid (0, 88) and not-installed apps are excluded;
  9. ambiguous: no AMBIGUOUS package in the real table is ever chosen, and both Amazon packages
     installed → `multiple_packages`;
  10. output is deterministic, and ties go to the lower id;

  plus tests for the defaults and validation, and for the current app.
- `live/ShadowLivePolicyTest` (4):
  - **11:** Layer 2 state bytes, log, trace records and cursor are identical with the policy off,
    on (several configurations, permissive and real resolver), disabled, and throwing;
  - **12:** one decision per prediction and examined rank, persisted with its batch, made on the
    pre-update prediction, and resolved by exactly the next launch;
  - a failed commit or a restart never duplicates or loses a decision;
  - **zero side effects:** the whole app source contains no launch API other than AdaPreload's own
    settings screens, notification and foreground service, all of which predate E1. The decision
    path has no Android import.
- **Sensitivity:**
  - a non-inclusive threshold fails test 3;
  - ignoring exclusions fails 7 tests.

**Instrumented: `ShadowSqliteStoreTest`** (compiled; needs a device). It checks that:
- decisions persist and are resolved across batches and connections;
- a duplicate decision rolls back the whole batch;
- a version 2 database is upgraded with its Layer 2 state, log and trace unchanged, and resumes.

## Limitations

- **No device validation of E1 yet.** It needs the Motorola (see below).
- **Decisions start with the first prediction made by an E1 build.** The prediction pending at the
  upgrade has none.
- **Only apps with a MAPPED package can be chosen** (30 of 87 ids). The unresolved AMBIGUOUS rows
  (Chrome, Phone, Messages, Maps, Settings, …) are always SKIP. Any E1/E3 evaluation is therefore
  restricted to mapped apps, and must report the skip share.
- **Eligibility depends on device state at decision time:** installed packages, the default home
  app and enabled IMEs, as of the window's start.
- **The threshold and defaults are untuned starting points,** and the probabilities are uncalibrated.
- **The JSON trace export is unchanged** and does not include `shadow_decisions`. Read the table
  from the database.
- **A pre-E1 build cannot open the version 3 database.**

## What E2 must add (not started)

1. A preload *executor* behind the PRELOAD decision. It must be an explicitly reviewed mechanism,
   with every attempt and its outcome recorded separately from the decision (concept C).
   - nb[39] L109-111 names soft preload via `startActivity` → `moveTaskToBack(true)`. That is not
     equivalent to kernel-level page scheduling and must not be described as such.
2. Safety limits: rate limiting, not acting while the screen is off, and not acting on the current
   or home app.
3. Updating the side-effect guard test: E2 is the first phase allowed to add a launch API.

Measurement (latency, PSS, battery) and the C → D benefit belong to E3.

## Device validation steps (shadow mode)

1. Install the E1 build. The data is kept: `adb install -r` upgrades the database from version 2
   to 3.
2. Start the service and use the phone normally.
3. Watch `adb logcat -s AdaPreloadLayer2 AdaPreloadShadow`.

What to check:
- each supported launch gives `launch #N … revealed #N−1 … updates N`, followed by one
  `prediction #N … -> PRELOAD|SKIP …; actual preload: NONE` line;
- no other app ever opens or comes to the foreground because of a decision;
- the trace card shows the last decision, marked "not executed".
