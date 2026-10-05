# ANDROID_TRACE_SPEC

The rules a real-device AdaPreload trace must follow so that it stays comparable with the
offline LSApp experiment. The offline notebook is the source of truth. This file restates it
and records every place where Android departs from it.

| Section | Content |
|---|---|
| 1 | Exact LSApp/Colab rules (`T*`), reproduced on Android unchanged |
| 2 | Android-specific adaptations (`A*`), decided 2026-10-05; not equivalent to LSApp |
| 3 | Package mapping (`M*`): representation and unresolved cases |
| 4 | Methodological limitations (`L*`) |
| 5 | Authoritative `app2id`: artifact audit and extraction procedure |
| 6 | Phase B implementation: where each rule lives in the code |

## Provenance

| Item | Value |
|---|---|
| Notebook | `Copy_of_adaPrealoaderneww.ipynb`, 45 cells |
| Notebook SHA-256 | `47540816a02d867a032865c903ab1a2489d06d0700c86dc216feb603867fc523` |
| Layer 1 checkpoint (`layer1_backbone_final.pt`) | SHA-256 `fd668f160363c9b32fe5ac561bec7b46ff5492d84e10f31d0bfd6d7a0c21c092` (nb[43] output) |
| Layer 2 artifact (`layer2_probability_outputs.pkl`) | SHA-256 `bfe1e61f1c0d4a90b90dc3623db0e13887915ec05b300f5ac2128c2b8535a2c8` |
| Layer 3 artifact (`layer3_cached_distributions.pkl`) | SHA-256 `1580e82d48d1b4eed3ba06996ae0757982b4f2587054401fd33596d449e566fd` |
| Authoritative vocabulary | `app/src/main/assets/lsapp_vocabulary.json`, extracted from the checkpoint above (§5.4) |
| Reproduction check | Re-running nb[3], nb[8] and nb[14] L13–15 unchanged on the public LSApp file (nb[1]) reproduces the notebook's printed counts: 1,673,261 launches, 213,496 retained launches, `VOCAB_SIZE` 88, and 19,121 same-second pairs. Vocabulary examples below come from this re-run; the checkpoint remains authoritative. |

**Citation key.** `nb[i] Lx–y` = notebook cell at 0-based position `i`, source lines `x–y`
(line 1 is the cell's `# @title` line). Cells cited:

| nb[i] | Header label | Colab cell id |
|---|---|---|
| nb[1] | CELL 2 — Download LSApp | `0H_ohA2yFYVZ` |
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

**Terms.**
- *Launch*: offline, a raw `Opened` record; on Android, a `SUPPORTED` launch candidate (A3).
- *Retained launch*: a launch that survives the duplicate collapse (T3).
- *Event*: a retained launch that serves as a prediction target (T9).
- *Observation window*: an interval during which AdaPreload is actually observing (A7).

---

## 1. Exact LSApp/Colab rules

These follow from the notebook alone. Android must implement them as written, applied to the
sequence produced by Section 2.

**T1. Launch definition.** A launch is a record whose `event_type == 'Opened'`. Records of
type `Closed`, `User Interaction` and `Broken` are never launches.
Fields used afterwards: `user_id`, `timestamp`, `app_name`. `session_id` is never used.
*Source:* nb[3] L9 (filter), L4–8 (rationale); nb[2] L5–11 (schema).

**T2. Ordering.** Each user's launches are processed in ascending `timestamp` order.
The two-key sorts (nb[3] L11, nb[8] L15) keep file order for same-second ties. The per-user
re-sorts that feed training and evaluation (nb[15] L20, nb[22] L20, nb[33] L121) use a
single-key, non-stable sort that reorders same-second ties (§5.4, F1).
*Source:* nb[3] L11; nb[8] L15; nb[15] L20; nb[22] L20; nb[33] L121.

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

The collapse holds in the stored `deduped` table. In the sequences actually used for training
and evaluation, the T2 re-sort reintroduces adjacent duplicates for about 2% of examples
(§5.4, F1). Android applies T3 as specified and has no such residue.

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
token. Android's handling of unknown apps is an adaptation (A3).
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
- In the stored `deduped` order, a target is never the same app as the last context launch
  (T3). In the frozen Layer 2/3 events this does not hold for 599 of 30,040 events (§5.4, F1).
  On Android it always holds.

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

## 2. Android-specific adaptations

Decided 2026-10-05. None of these is claimed to be equivalent to LSApp; each is a documented
departure. Items marked *Validate* must be checked on the target device and API level before
field use.

### 2.1 Observation pipeline (normative order)

```
UsageEvents (calling user only)
 → keep events inside an observation window             (A6, A7, A8)
 → write every event to the raw trace                    (A5, AU1)
 → launch candidates: eventType == ACTIVITY_RESUMED      (A1)
 → classify package → SUPPORTED(app id) | discard class  (A4, Section 3)
 → discard everything that is not SUPPORTED              (A3)
 → collapse consecutive identical app ids                (T3)
 → append to the prediction sequence                     (T7–T9)
```

**Consequence.** Discarded candidates create no boundary: `A, X, A`, where `X` is unsupported
or excluded, yields one retained `A`. The prediction sequence therefore depends only on three
things:
1. the observation windows;
2. which packages have status `MAPPED`;
3. T3.

The exclusion rules (A4) never add or split launches. They only label discards for audit and
keep excluded packages from ever being mapped.

**A1. Launch candidate = `ACTIVITY_RESUMED`.**
- `UsageEvents.Event.ACTIVITY_RESUMED` has value 1. On API 26–28 the same value is named
  `MOVE_TO_FOREGROUND`, so one constant covers minSdk 26+.
- Not assumed equivalent to LSApp `Opened`, whose collector semantics are not in the notebook
  (nb[3] L4–9). Because T3 collapses repeated same-app events, burst granularity does not
  matter; cross-app interleavings may.
- *Validate on the target device/API:*
  - **V1** multi-activity apps;
  - **V2** rotation / configuration change;
  - **V3** screen off, then unlock with the same app on top;
  - **V4** Home and Recents transitions;
  - **V5** split-screen, picture-in-picture and freeform (several apps resumed at once);
  - **V6** dialogs from other packages over an app (permission, share sheet, in-call);
  - **V7** Custom Tabs and other cross-package activities;
  - **V8** notification shade and Quick Settings (expected: no `ACTIVITY_RESUMED`);
  - **V9** notification-tap trampolines;
  - **V10** delay between a launch and its appearance in `queryEvents`.

**A3. Discard before collapse (OOV, excluded, ambiguous).**
- Order: map → discard non-`SUPPORTED` → collapse (T3) → append.
- LSApp has no OOV case (T6), so this rule has no offline counterpart. It is an Android
  adaptation.
- Every discarded candidate stays in the audit trace with its class (AU2).

**A4. Exclusions.** Classification precedence for each launch candidate (first match wins):

| # | Class | Rule | Mechanism |
|---|---|---|---|
| 1 | `SELF` | AdaPreload's own package | `Context.getPackageName()` |
| 2 | `EXCLUDED_HOME` | the current default home app | `resolveActivity(MAIN + CATEGORY_HOME, MATCH_DEFAULT_ONLY)`, re-resolved at each window start. Not "every HOME handler": Settings declares a fallback HOME activity (AOSP Settings `FallbackHome`), and `Settings` is a vocabulary app. |
| 3 | `EXCLUDED_IME` | enabled input methods | `InputMethodManager.getEnabledInputMethodList()` package names |
| 4 | `EXCLUDED_SYSTEM` | small explicit list (below) | exact package match |
| 5 | `SUPPORTED` / `AMBIGUOUS` / `UNSUPPORTED` | package listed in mapping table 1 (Section 3) | mapping lookup |
| 6 | `EXCLUDED_NON_LAUNCHABLE` | no launcher entry | `getLaunchIntentForPackage(pkg) == null` |
| 7 | `OOV` | everything else | — |

- **Rows 1–4 come before the mapping**, so an excluded package can never be mapped. A mapping
  entry for such a package is a configuration error.
- **Row 6 comes after the mapping**, so a metadata heuristic never drops a vocabulary app that
  is a system package without a launcher entry (for example `Android In Call UI`).
- **No blanket `FLAG_SYSTEM` rule.** Many vocabulary apps are preinstalled system apps
  (`Settings`, `Phone`, `Camera`, `Clock`, `Contacts`, `Messages`, …).
- **Initial explicit list (row 4); validate on the target device:**
  - `com.android.systemui`;
  - permission dialogs: `com.google.android.permissioncontroller` and
    `com.android.permissioncontroller` (API 29+), `com.google.android.packageinstaller` and
    `com.android.packageinstaller` (API 26–28);
  - system chooser/resolver: `android`, `com.android.intentresolver`.
- **Package visibility:** rows 2, 3 and 6 need `<queries>` entries on API 30+ for the HOME,
  LAUNCHER and `android.view.InputMethod` intents. These are declared in the manifest.

**A5. Time.**
- The raw `getTimeStamp()` (epoch ms) is stored unmodified in the raw trace. The device time
  zone is recorded per observation window.
- No time features are computed for, or fed to, the model. T10 (`morph_mode='none'`) means the
  effective model input is the ordered app-id sequence.
- **Ordering:** events are processed in the order `queryEvents` returns them and are not
  re-sorted. A timestamp lower than its predecessor's (e.g. a clock change) is logged as an
  anomaly, not reordered.

**A6. Sequence start.**
- The experiment's sequence starts when the observation service first runs with Usage Access
  granted. Events before that instant are never read, and no history is reconstructed.
- The first retained launch after the start is context only (T9). The Layer 2 adapter, once
  implemented, starts at zero at that point (T12).
- The Phase A service can run without Usage Access, but no observation window opens until
  access is granted.

**A7. Observation windows and gaps.**
- A window opens when observation starts: the service starts with Usage Access granted, or
  access is regained.
- A window closes at the upper bound of the last successful `queryEvents` call before the
  service stops, is killed, or loses Usage Access.
- Events whose timestamps fall in a gap between windows are never processed, even though
  UsageEvents may still return them. No backfill.
- Each window's start, end and close reason is recorded (AU3).
- **Locked 2026-10-05:** a gap does not reset the experiment. The sequence, context window,
  collapse state and (later) adapter all continue across gaps, consistent with T4. The last
  retained app before a gap and the first one after it are subject to T3.

**A8. Profiles.**
- Observation runs only when AdaPreload runs in the primary user's personal profile
  (`UserManager.isSystemUser()`). Otherwise no window opens, and the reason is logged.
- `queryEvents` only returns the calling user's events (AOSP
  `UsageStatsService.queryEvents` → `queryEventsHelper(UserHandle.getCallingUserId(), …)`).
  Work-profile and other-user launches therefore never reach AdaPreload. They are
  *unobservable*, not discarded, and cannot be logged per event.
- For audit, the profiles present on the device are logged at each window start
  (`UserManager.getUserProfiles()`).

**A9. K.** No live K is fixed. Each event logs the Layer 1 and Layer 2 top-20 (T15), so every
K ∈ {1, 2, 4, 5, 8, 16} can be computed afterwards, exactly as offline.

**A10. Preload action: shadow mode.** Decisions are logged only. No `startActivity`,
`moveTaskToBack` or other action is ever taken on another app. The notebook's planned soft
preload (nb[39] L107–111) is not implemented.

### 2.2 Audit records

| ID | Record | Content |
|---|---|---|
| AU1 | Raw event | every event returned inside a window: type, package, class, raw timestamp |
| AU2 | Classification | per launch candidate: class (A4), the mapped app id if `SUPPORTED`, and whether T3 collapsed it |
| AU3 | Observation window | start, end, close reason, time zone, home package, enabled IMEs, profiles present, mapping-table version |
| AU4 | Anomaly | non-monotonic timestamps, `queryEvents` failures, Usage Access lost |

---

## 3. Package mapping (A2)

### 3.1 Representation

```
Android package ─(table 1)→ canonical app identity ─(table 2)→ LSApp display name ─(app2id)→ checkpoint app id
```

| Stage | Source | Rules |
|---|---|---|
| Table 1: package → canonical identity | researcher-curated, per device | one row per package; status `MAPPED`, `AMBIGUOUS` or `UNSUPPORTED`; the evidence used to verify the identity is recorded |
| Table 2: canonical identity → LSApp name | researcher-curated | at most one LSApp name per identity; the name must be an exact key of `app2id` |
| `app2id`: LSApp name → id | extracted from the checkpoint (Section 5) | never hand-written or edited |

- Only `MAPPED` rows produce `SUPPORTED` candidates. `AMBIGUOUS` and `UNSUPPORTED` rows, and
  packages missing from table 1, are discarded under A3.
- Several packages may share one canonical identity. They then collapse together under T3, as
  LSApp collapses on `app_name` (nb[8] L8–9).
- The tables are frozen for the duration of an experiment, and their version is logged per
  window (AU3). Changing them mid-experiment changes the sequence.
- **Storage:** tables 1 and 2 are kept as one tab-separated file,
  `app/src/main/assets/package_mapping.tsv`, with columns `package, canonical_app, lsapp_name,
  lsapp_id, status, confidence, source, notes`. It is validated against the vocabulary at load,
  and any invalid row stops collection:
  - a MAPPED name must be a vocabulary key, and its id must match;
  - one canonical identity maps to one LSApp name;
  - packages are unique;
  - always-excluded packages cannot be mapped.
- **Current state (Phase B, 2026-10-05):**
  - 31 `MAPPED` rows:
    - 30 apps whose LSApp label is the app's own brand name, under their official package id
      (confidence `high`, not yet verified on the target device);
    - `in.amazon.mShop.android.shopping` → `Amazon Shopping` (id 2), the regional package
      observed on the Motorola edge 50 fusion (Android 16) trace of 2026-10-06, where it had
      been classified OOV;
  - 39 `AMBIGUOUS` rows recording M1–M6. These are counted separately and never mapped.

### 3.2 Unresolved cases (flagged, not guessed)

The names come from the non-authoritative re-run (Appendix) and must be re-confirmed against
the checkpoint.

| ID | LSApp names | Why ambiguous |
|---|---|---|
| M1 | `Phone`, `Android In Call UI` | The dialer and the in-call screen are often one package, so a package-level mapping cannot separate them. Separating them would need activity class names, which is a further adaptation. |
| M2 | `Messages`, `Messaging`, `Verizon Messages` | Three SMS-client labels; which device app matches which label cannot be decided from the names. |
| M2b | `Facebook Messenger` / `Messenger Lite`, `Text One` / `TextNow`, `Telegram` / `Telegram X`, `Hangouts` | Related or similarly named apps; each must be verified as a distinct identity. `Hangouts` is discontinued. |
| M3 | `Samsung Email`, `Samsung Gallery`, `Samsung Internet Browser`, `Samsung Notes`, `Samsung Pay`, `Flipboard Briefing` | Samsung-specific, with no counterpart on other vendors. On Samsung devices, current app names have changed (e.g. Samsung Pay is now Samsung Wallet). |
| M4 | `Calculator`, `Calendar`, `Camera`, `Clock`, `Contacts`, `Phone`, `Messages`, `Settings`, `Maps` | Vendor-neutral role labels. Whether a device's own app for that role maps to the label is a decision. |
| M5 | `Google Play Music`, `Hangouts`, `Twitter`, `PayPal Mobile Cash`, `Microsoft Bing Search` | Discontinued apps, or labels that differ from today's store names. Mapping by identity vs. by label, and whether a successor app maps, are decisions. |
| M6 | `Google`, `Google Chrome`, `Microsoft Bing Search` | The Google app also hosts assistant and feed surfaces. Custom Tabs appear as the browser package (L5). |

---

## 4. Methodological limitations

| ID | Limitation |
|---|---|
| L1 | `ACTIVITY_RESUMED` is not shown to be equivalent to LSApp `Opened`. Only the A1 validation can bound the difference. |
| L2 | Discard-before-collapse has no offline counterpart. It merges runs separated by unsupported apps (`A, X, A` → `A`), which changes the transition structure relative to real usage. LSApp has only 87 apps across 292 users (nb[2] output), which suggests it was restricted to a fixed app set before release. How that restriction treated other apps is not documented in the notebook. |
| L3 | The prediction space is the 87 LSApp apps, collected in 2017–2018. Decisions can never select an app outside the vocabulary. Results must report coverage over retained events **and** the share of launch candidates discarded, by class (AU2). |
| L4 | Ambiguous and unmapped apps are excluded until resolved, so the device sequence covers only part of real usage. |
| L5 | Unobservable: other profiles and users (A8); launches during gaps (A7); the true task owner of cross-package activities such as Custom Tabs and share targets. Task-root fields are system-only for third-party apps. |
| L6 | One device is one user (n = 1). Offline results are 59-user macro statistics, so per-device results are case studies, not cohort estimates. |
| L7 | The device user is a cold-start user, like the offline held-out users (zero adapter; first launch is context only). Early-sequence results are noisy. |
| L8 | Gaps are not backfilled (A7). How the offline data handled its own gaps is unknown. |
| L9 | Ordering uses Android stream order at millisecond resolution, so the device trace has no adjacent duplicates. Offline, the unstable re-sort of same-second ties left adjacent duplicates in 1.99% of the frozen evaluation events and about 2.06% of training examples (§5.4, F1). These events are almost never hit at rank 1 (L2 Hit@1 0.015 vs 0.524 for all other events), so offline metrics are slightly lower than they would be on duplicate-free sequences. Device-vs-offline comparisons must state this. The offline results are not changed. |
| L10 | The offline `lr = 0.001` and K = 5 were selected on the evaluation cohort (nb[43] L7–11). This caveat carries over to any device comparison. |
| L11 | Shadow mode supports no claims about latency, memory, battery or energy (unsafe-claims list in nb[43] output). |

---

## 5. Authoritative `app2id`: artifact audit

### 5.1 Where `app2id` exists

| Artifact | Contains `app2id`? | Evidence |
|---|---|---|
| `layer1_backbone_final.pt` | **Yes.** Key `app2id` (dict `str → int`), next to `model_state_dict`, `vocab_size`, `window`, `d_model`, `morph_mode` | nb[20] L26–33 |
| `layer2_probability_outputs.pkl` | **Yes, as a copy.** `app2id` = `ckpt['app2id']`, plus `id2app` with `0 → '<PAD>'` | nb[33] L29–31, L201 |
| `layer3_cached_distributions.pkl` | No (only `n_apps`) | nb[34] L105–108 |
| `layer3_config.json`, `layer3_final_config.json` | No | nb[39] L24–42; nb[41] L111–122; nb[43] L20–33 |
| Notebook outputs | No: `app2id` is never printed | full notebook |

### 5.2 Inputs

Both files were provided on 2026-10-05, and their SHA-256 values match Provenance:
- `layer1_backbone_final.pt` (331,499 bytes);
- `layer2_probability_outputs.pkl` (11,365,614 bytes).

They are not committed to the repository. No retraining and no notebook change was involved.

### 5.3 Extraction procedure (read-only)

1. Verify that the file's SHA-256 equals the Provenance value. Stop if it does not.
2. Load it with `torch.load(path, map_location='cpu', weights_only=True)`.
   - The notebook loads with `weights_only=False` (nb[21] L5); extraction doesn't need that.
   - `weights_only=True` only rebuilds tensors and plain containers, so loading the file
     cannot run arbitrary code.
   - Tested 2026-10-05 with torch 2.14.1 on a stand-in checkpoint with the same structure (the
     nb[17] model with `morph_mode='none'`, and `app2id` from the Appendix). It loads.
   - The stand-in is 331,017 bytes and holds 79,576 parameters. That is consistent with the
     real 331,499-byte file containing nothing else of significant size.
3. All acceptance checks must pass:

| ID | Check | Source |
|---|---|---|
| C1 | keys are exactly `model_state_dict, app2id, vocab_size, window, d_model, morph_mode` | nb[20] L26–33 |
| C2 | `vocab_size == 88`, `window == 20`, `d_model == 64`, `morph_mode == 'none'` | nb[20] L8, L29–32 |
| C3 | 87 entries; ids exactly 1..87; keys are `str` and values are `int`; id 0 absent | nb[14] L13–15 |
| C4 | `app2id[name] == i + 1` for each `i, name in enumerate(sorted(app2id))` | nb[14] L13–14 |
| C5 | `app_emb.weight` is (88, 64), `pos_emb.weight` (20, 64), `out.weight` (88, 64), `out.bias` (88,); no `time_proj`, `fuse_mlp` or `gate_net` keys | nb[17] L15–27; T10 |
| C6 | if provided, the Layer 2 artifact's `app2id` equals the checkpoint's, and `id2app` is its inverse plus `0 → '<PAD>'` | nb[33] L29–31 |
| C7 | equals the Appendix list. A difference stops the process: the checkpoint still wins, but the difference must be explained. | Appendix |

4. Export, never hand-edited: a JSON file with provenance, model configuration, padding
   semantics and the id → name list. Strings must be byte-exact (e.g. U+2019 in `S’more`).
   Implemented by `tools/audit_vocabulary.py`. The script verifies the hashes before loading
   anything, writes nothing if a gating check fails, and loads the Layer 2 pickle with an
   unpickler that only allows the three numpy classes the file references.

### 5.4 Audit result (2026-10-05)

```
python3 tools/audit_vocabulary.py layer1_backbone_final.pt layer2_probability_outputs.pkl
```

Run with torch 2.14.1 and numpy 2.4.6. Both inputs hash identically before and after the run.

| ID | Check | Result |
|---|---|---|
| H1, H2 | SHA-256 of both inputs | PASS: match Provenance |
| C1 | checkpoint keys | PASS: exactly the 6 expected keys |
| C2 | config | PASS: `vocab_size` 88, `window` 20, `d_model` 64, `morph_mode` `'none'` |
| C3 | ids 1..87, `str → int`, padding id 0 absent | PASS |
| C4 | ids follow Python code-point sort order; stored order = id order | PASS |
| C5 | tensor shapes; no time-feature parameters; float32 | PASS: 28 tensors, 79,576 parameters, no `time_proj`/`fuse_mlp`/`gate_net` |
| P1 | padding embedding row 0 is exactly zero (`padding_idx=0`) | PASS |
| C6 | Layer 2 `app2id` equals the checkpoint (content and order); `id2app` = inverse + `0 → '<PAD>'` | PASS |
| L1 | Layer 2 scale | PASS: 30,040 events, 59 users, top-20 for L1 and L2 |
| L2–L4 | target, top-20 and context ids all within 1..87 (padding never a target, never selected, never inside an online context) | PASS: targets span [1, 86] |
| L5 | context length = min(event_index, 20) | PASS |
| C7 | equals the Appendix re-run | PASS: identical |
| W1, W2 | written file round-trips to `app2id`; `apps_sha256` recomputes | PASS |

**Vocabulary mismatches: none.**

**F1. Adjacent duplicates in the frozen evaluation sequences.** This is reported, not gating,
and it is not a vocabulary issue.
- In 599 of 30,040 Layer 2 events (1.99%, 21 of 59 users), the target equals the last
  context app.
- Cause: the per-user `sort_values('timestamp')` (nb[22] L20, nb[33] L121) is a single-key,
  non-stable sort, and it reorders launches that share a 1-second timestamp.
- Evidence:
  - every held-out user's evaluated sequence has the same length and the same multiset of app
    ids as the stored `deduped` order;
  - all 1,078 positions where the order differs lie inside same-second ties;
  - the same sort reproduces the artifact's order for 59 of 59 users (pandas 3.0.6,
    numpy 2.4.6);
  - the stored order has no adjacent duplicates.
- Training (nb[15] L20, same sort; reproduced, not verified against an artifact): 3,770 of
  183,165 examples (2.06%, 97 users).
- Effect on these events: L1 Hit@1 0.018 / Hit@5 0.431 and L2 Hit@1 0.015 / Hit@5 0.648,
  against 0.511 / 0.818 and 0.524 / 0.855 on all other events.
- Consequence: see T3, T9 and L9. The offline methodology and results are unchanged; Android
  applies T3 as specified.

**Output:** `app/src/main/assets/lsapp_vocabulary.json`

| Field | Value |
|---|---|
| File SHA-256 | `e039f4c50d8c25c52f7c9f047bba3a389d73dbad4468d9a85851be412604a7df` (6,464 bytes, ASCII; deterministic: a re-run is byte-identical) |
| `apps_sha256` | `d0801f3b2e1ece7558989fa2fc70c86560aebb36a2e441f2ffa9b865a2147d78`: SHA-256 of the UTF-8 lines `<id>\t<name>\n`, ascending id |
| Content | provenance (both input hashes, notebook hash, tool versions), model config, padding `{id: 0, token: "<PAD>"}`, 87 `{id, name}` entries. No package names. |
| Consumers | none yet. Package mapping and Phase B have not started. |

---

## 6. Phase B implementation

Code in `app/src/main/java/com/adapreload/instrumentation/`. The `trace/` package has no
Android imports and is unit-tested on the JVM (`app/src/test/.../trace/`).

| Rule | Implementation |
|---|---|
| A1 launch candidate, A3 discard before collapse, T3 collapse (first timestamp kept), A5 stream order | `trace/TracePipeline.kt` |
| A4 classification precedence and explicit system list | `trace/LaunchClassifier.kt` |
| A4 default home only, enabled IMEs, launcher entry; A8 primary user, unlocked | `collect/AndroidTraceSources.kt` (`AndroidEnvironment`) |
| Raw events from `queryEvents` | `collect/AndroidTraceSources.kt` (`UsageStatsEventSource`) |
| A6, A7 windows, incremental polling, no backfill, recovery after process death | `trace/TraceSession.kt` |
| Exactly-once persistence; sequence state derived from stored events | `collect/SqliteTraceStore.kt` |
| Section 3 mapping table | `trace/PackageMapping.kt`, `assets/package_mapping.tsv` |
| T5 vocabulary, verified by `apps_sha256` at load | `trace/Vocabulary.kt`, `assets/lsapp_vocabulary.json` |
| AU1–AU4 audit export | `trace/TraceExporter.kt` |
| Polling loop (every 5 s on one background thread), status, export | `collect/TraceRuntime.kt` |

**Incremental polling.**
- Each poll queries `[cursor, now − 2 s)` and commits the records and the new cursor in one
  SQLite transaction, so every event is processed exactly once.
- The 2 s settle delay covers events the platform has stamped but not yet inserted: AOSP sets
  `mTimeStamp` when the event is reported and inserts it later on a handler thread.
- If the wall clock is behind the cursor, nothing is queried until it catches up.

**Restart.** On service start, a window left open by a dead process is closed at its cursor
(`PROCESS_ENDED`). A new window then opens at the current time, so gap events are never read.
The sequence state is re-derived from the stored records.

**Not implemented in Phase B:** Layer 1 inference, the Layer 2 adapter, any model runtime,
preload actions, K selection and prediction UI.

**Phase C:** frozen Layer 1 inference with numerical parity to the checkpoint is in
`docs/LAYER1_ANDROID_INFERENCE.md`. It is not yet connected to this trace.

---

## Appendix: vocabulary from the public-LSApp re-run (NON-AUTHORITATIVE)

Produced by nb[14] L13–14 on the public LSApp file. Use it only for planning Section 3 and for
check C7. The expected id is the position in the list.

| id | name | id | name | id | name |
|---|---|---|---|---|---|
| 1 | `AOL` | 30 | `Hangouts` | 59 | `Robinhood` |
| 2 | `Amazon Shopping` | 31 | `Hulu` | 60 | `Samsung Email` |
| 3 | `Android In Call UI` | 32 | `Ibotta` | 61 | `Samsung Gallery` |
| 4 | `Army Men Strike` | 33 | `Instagram` | 62 | `Samsung Internet Browser` |
| 5 | `Badoo` | 34 | `Kik` | 63 | `Samsung Notes` |
| 6 | `Baseball Boy!` | 35 | `Lucktastic` | 64 | `Samsung Pay` |
| 7 | `Brave Browser` | 36 | `MAX Cleaner` | 65 | `Settings` |
| 8 | `Calculator` | 37 | `MUIQ Survey App` | 66 | `Slidejoy` |
| 9 | `Calendar` | 38 | `Maps` | 67 | `Snapchat` |
| 10 | `Calorie Counter` | 39 | `Messages` | 68 | `Spotify Music` |
| 11 | `Camera` | 40 | `Messaging` | 69 | `SurveyCow` |
| 12 | `Clean Master` | 41 | `Messenger Lite` | 70 | `Swagbucks` |
| 13 | `Clock` | 42 | `MetroZone` | 71 | `Swagbucks Watch (TV)` |
| 14 | `Contacts` | 43 | `Microsoft Bing Search` | 72 | `S’more` |
| 15 | `DigiHUD Pro Speedometer` | 44 | `Microsoft Outlook` | 73 | `Telegram` |
| 16 | `Discord` | 45 | `Minesweeper Classic (Mines)` | 74 | `Telegram X` |
| 17 | `EntertaiNow` | 46 | `Movie Play Box` | 75 | `Text One` |
| 18 | `Facebook` | 47 | `Netflix` | 76 | `TextNow` |
| 19 | `Facebook Messenger` | 48 | `OfferUp` | 77 | `The PCH App` |
| 20 | `Faceu` | 49 | `Pandora Music` | 78 | `Twitter` |
| 21 | `Flickr` | 50 | `PayPal Mobile Cash` | 79 | `Verizon Messages` |
| 22 | `Flipboard Briefing` | 51 | `Phone` | 80 | `Walmart` |
| 23 | `Gmail` | 52 | `Pinterest` | 81 | `WeChat` |
| 24 | `Google` | 53 | `Pixlr` | 82 | `WhatsApp Messenger` |
| 25 | `Google Chrome` | 54 | `Podcast Addict` | 83 | `Words With Friends 2` |
| 26 | `Google Drive` | 55 | `Quora` | 84 | `Yahoo Mail` |
| 27 | `Google Photos` | 56 | `Receipt Hog` | 85 | `YouTube` |
| 28 | `Google Play Music` | 57 | `Reddit` | 86 | `eBay` |
| 29 | `Google Play Store` | 58 | `Reward Stash` | 87 | `imo` |
