# Phase E2a: read-only service warmability inventory

E2a answers one question for researcher review: **which installed, mapped apps declare a service an
ordinary app could, in principle, bind to?** It reads PackageManager metadata only. It binds
nothing, starts nothing, and invokes no Binder. It is a feasibility measurement, **not** an
allowlist: nothing here decides a service *should* be used.

E2b (the preload executor) is not implemented, and no `preload_attempts` table exists.

## What it inspects

For each row of the existing mapping table (`assets/package_mapping.tsv`) whose status is
**MAPPED** (so it carries a canonical app id), the inventory reads the package's declared services
and classifies each one. AMBIGUOUS and UNSUPPORTED rows carry no canonical id and are never
inventoried; a package with no mapping row at all is never queried. There is no second package
vocabulary.

**Classification** (`ServiceWarmabilityInventory.classify`). A service is **POTENTIALLY_BINDABLE**
only when all of these hold; otherwise it is **NOT_BINDABLE** with the first failing reason, in this
order:

| Check | Field | Disqualifier if it fails |
|---|---|---|
| Exported | `ServiceInfo.exported` | `NOT_EXPORTED` |
| Component enabled | `ServiceInfo.enabled` | `COMPONENT_DISABLED` |
| Application enabled | `ApplicationInfo.enabled` | `APPLICATION_DISABLED` |
| No permission | `ServiceInfo.permission == null` | `PERMISSION_REQUIRED` |

The label is **POTENTIALLY_BINDABLE**, never GUARANTEED_BINDABLE: restrictions that only appear at
bind time (process signature, `BIND_EXTERNAL_SERVICE`, runtime policy) are not checked, because E2a
never binds. `exported` is read as the platform resolved it, never inferred from the package name or
from the presence of intent filters.

## Android APIs used

- `PackageManager.getPackageInfo(pkg, GET_SERVICES | MATCH_DISABLED_COMPONENTS)` — the only call.
  `GET_SERVICES` returns `PackageInfo.services`; `MATCH_DISABLED_COMPONENTS` (API 24+) includes
  disabled services so they can be reported as `COMPONENT_DISABLED` rather than omitted.
- On API 33+ the `PackageManager.PackageInfoFlags` overload is used; below it, the int-flags
  overload. Both return the same `ServiceInfo` fields. minSdk 26, targetSdk 37.
- A package that is not installed, or not visible under AdaPreload's `<queries>` declaration, throws
  `NameNotFoundException`, reported as not installed (never as an error). Mapped apps are ordinary
  launchable apps, so they are visible via the existing LAUNCHER query that E1 already relies on.

The inventory does **not** enumerate all installed packages (which would need broad package
visibility). It is scoped to the mapped packages, which are the only ones E2 could ever target.

## Persistence

**None added.** The inventory is held in memory for the current process and rendered to:
- logcat (`AdaPreloadInventory`): the full report, including the candidate table;
- the status card: a one-line "warmable mapped apps: N / M installed";
- an exported JSON report (`adapreload_service_inventory_*.json`) via the system file picker.

A database migration is deliberately avoided: the inventory is a point-in-time scan for review, not
part of the live pipeline, and nothing downstream reads it. Schema version stays at 3 (Phase E1).

## Output

The JSON report (`InventoryReport.json`) carries, for researcher review:
- `counts`: mapped rows, mapped apps, installed packages and apps, warmable apps, candidate
  services, the per-app-count breakdown, and coverage as a percentage string (the trace JSON writer
  emits only integers, and `warmable_apps / installed_mapped_apps` reproduces it exactly);
- `ambiguous_apps`: each canonical id reached by more than one MAPPED package;
- `packages`: every mapped package, whether installed, and every declared service with its fields,
  `potentially_bindable` and `disqualifier`.

## Safety and the E1 invariant

The E1 guard test `ShadowLivePolicyTest.theShadowPolicyHasNoWayToLaunchAnything` scans all of
`src/main/java` and asserts the only launch-type APIs in the whole app are AdaPreload's own
pre-existing three (its settings screens, its notification, its foreground service). E2a adds none,
so that test still passes — it is the standing proof that E2a introduced no way to bind, start or
warm anything.

## Tests (`inventory.ServiceWarmabilityInventoryTest`, 12)

Over fake metadata and the real mapping asset:

1. exported + enabled + no permission → POTENTIALLY_BINDABLE;
2. exported + component-disabled → `COMPONENT_DISABLED`; exported + app-disabled → `APPLICATION_DISABLED`;
3. not exported → `NOT_EXPORTED`;
4. exported + permission → `PERMISSION_REQUIRED`;
5. several candidate services → all reported, none selected;
6. an unmapped or non-MAPPED package → not eligible, not queried;
7. two packages mapping to one canonical id → reported as ambiguous;
8. a package installed with no services, and a not-installed package → handled safely;

plus the fixed disqualifier order, zero-coverage when nothing is installed, the report renderer, and
a run over the shipped mapping table (31 MAPPED rows, 30 canonical ids, Amazon Shopping the one
ambiguous id).

## Running it on the device

1. Build and install: `./gradlew :app:installDebug` (keeps existing data).
2. Open AdaPreload, tap **Scan warmable services (read-only)**. This runs the scan on the
   background thread; it binds nothing.
3. Read the result with `adb logcat -s AdaPreloadInventory`, or tap **Export service inventory
   (JSON)** to save the report through the file picker.

The scan is independent of the trace service: it does not need the service running or Usage Access.

## Not in E2a

- No binding, starting, stopping, broadcasting or Binder call.
- No preload executor and no `preload_attempts` table (E2b).
- No allowlist: the inventory lists what *could* theoretically be bound, for the researcher to
  review which, if any, *should* be, given that an exported service may still have side effects.
