package com.adapreload.instrumentation.inventory

import com.adapreload.instrumentation.trace.Json
import java.util.Locale

/**
 * Renders a [ServiceInventoryResult] for review: a human-readable summary (logcat and UI) and a
 * machine-readable JSON report (export for researcher review). Both are read-only descriptions of
 * what was found; neither selects or acts on anything.
 */
object InventoryReport {

    private fun pct(x: Double) = String.format(Locale.ROOT, "%.1f%%", x * 100)

    /** A compact summary (A, B, C, E) for logcat and the status card. */
    fun summary(r: ServiceInventoryResult): String = buildString {
        append("E2a service warmability inventory (${ServiceWarmabilityInventory.VERSION})\n")
        append("Mapped rows: ${r.mappedRowCount}; mapped apps (distinct ids): ${r.mappedAppCount}\n")
        append("Installed mapped packages: ${r.installedPackages.size} of ${r.packages.size}; ")
        append("installed mapped apps: ${r.installedAppIds.size}\n")
        append("Warmable apps: ${r.warmableAppIds.size} of ${r.installedAppIds.size} installed")
        if (r.installedAppIds.isNotEmpty()) append(" (${pct(r.coverageOfInstalled)})")
        append("\n")
        append("Apps by candidate count - 0: ${r.appsWithNoCandidate.size}, ")
        append("1: ${r.appsWithOneCandidate.size}, >1: ${r.appsWithMultipleCandidates.size}\n")
        append("Candidate services: ${r.candidates.size}; ambiguous mapped ids: ${r.ambiguousApps.size}")
    }

    /** The full human-readable report, including the candidate table (D) and ambiguous cases (F). */
    fun text(r: ServiceInventoryResult): String = buildString {
        append(summary(r))
        append("\n\nD. Potentially bindable services:\n")
        if (r.candidates.isEmpty()) {
            append("  (none)\n")
        } else {
            append(String.format(Locale.ROOT, "  %-26s %-40s %-28s\n", "App [id]", "Package", "Service"))
            for ((pkg, row) in r.candidates.sortedWith(compareBy({ it.first.appId }, { it.first.packageName }, { it.second.service.className }))) {
                append(String.format(Locale.ROOT, "  %-26s %-40s %-28s\n",
                    "${pkg.lsappName} [${pkg.appId}]", pkg.packageName, shortClass(row.service.className)))
            }
        }
        append("\nF. Ambiguous mapped ids (more than one MAPPED package):\n")
        if (r.ambiguousApps.isEmpty()) {
            append("  (none)\n")
        } else {
            for (a in r.ambiguousApps) {
                val installed = a.packages.filter { p -> r.installedPackages.any { it.packageName == p } }
                append("  ${a.lsappName} [${a.appId}]: ${a.packages.joinToString(", ")}")
                append(" (installed: ${if (installed.isEmpty()) "none" else installed.joinToString(", ")})\n")
            }
        }
        append("\nInstalled mapped apps with no candidate service:\n")
        val names = r.installedPackages.associate { it.appId to it.lsappName }
        append(if (r.appsWithNoCandidate.isEmpty()) "  (none)\n"
        else r.appsWithNoCandidate.joinToString("\n") { "  ${names[it]} [$it]" } + "\n")
    }

    /** The machine-readable report for export, as pure-ASCII JSON (reuses the trace JSON writer). */
    fun json(r: ServiceInventoryResult): String = Json.write(
        linkedMapOf(
            "format" to "adapreload-e2a-service-inventory",
            "format_version" to 1L,
            "inventory_version" to ServiceWarmabilityInventory.VERSION,
            "note" to "Read-only PackageManager metadata. No service was bound, started or invoked. " +
                "potentially_bindable means exported, enabled and permission-free; bind-time restrictions are not checked.",
            "counts" to linkedMapOf(
                "mapped_rows" to r.mappedRowCount.toLong(),
                "mapped_apps" to r.mappedAppCount.toLong(),
                "mapped_packages" to r.packages.size.toLong(),
                "installed_mapped_packages" to r.installedPackages.size.toLong(),
                "installed_mapped_apps" to r.installedAppIds.size.toLong(),
                "warmable_apps" to r.warmableAppIds.size.toLong(),
                "candidate_services" to r.candidates.size.toLong(),
                "apps_zero_candidates" to r.appsWithNoCandidate.size.toLong(),
                "apps_one_candidate" to r.appsWithOneCandidate.size.toLong(),
                "apps_multiple_candidates" to r.appsWithMultipleCandidates.size.toLong(),
                // Coverage = warmable_apps / installed_mapped_apps; given as a string because the
                // trace JSON writer emits only integers (the two counts above reproduce it exactly).
                "coverage_of_installed_pct" to pct(r.coverageOfInstalled),
            ),
            "ambiguous_apps" to r.ambiguousApps.map {
                linkedMapOf<String, Any?>("app_id" to it.appId.toLong(), "lsapp_name" to it.lsappName, "packages" to it.packages)
            },
            "packages" to r.packages.map { pkg ->
                linkedMapOf<String, Any?>(
                    "package" to pkg.packageName,
                    "app_id" to pkg.appId.toLong(),
                    "lsapp_name" to pkg.lsappName,
                    "installed" to pkg.installed,
                    "services" to pkg.services.map { row ->
                        linkedMapOf<String, Any?>(
                            "class" to row.service.className,
                            "exported" to row.service.exported,
                            "component_enabled" to row.service.componentEnabled,
                            "application_enabled" to row.service.applicationEnabled,
                            "permission" to row.service.permission,
                            "potentially_bindable" to row.potentiallyBindable,
                            "disqualifier" to row.disqualifier?.name,
                        )
                    },
                )
            },
        )
    )

    private fun shortClass(className: String): String = className.substringAfterLast('.').take(28)
}
