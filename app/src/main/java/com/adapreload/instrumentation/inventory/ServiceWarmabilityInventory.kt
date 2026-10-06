package com.adapreload.instrumentation.inventory

import com.adapreload.instrumentation.trace.MappingStatus
import com.adapreload.instrumentation.trace.PackageMapping

/**
 * Phase E2a: a READ-ONLY inventory of which installed, MAPPED apps declare services that an
 * ordinary third-party app could, in principle, bind to.
 *
 * This file is pure metadata classification: it reads [DeclaredService] records (taken from
 * `PackageManager` on the device) and the existing [PackageMapping], and produces a report. It
 * binds nothing, starts nothing, and calls no Binder. The result labels a service
 * POTENTIALLY_BINDABLE, never GUARANTEED_BINDABLE: further restrictions (process signature,
 * `BIND_EXTERNAL_SERVICE`, runtime policy) can only be seen at bind time, which E2a never reaches.
 *
 * The inventory is a feasibility measurement for researcher review, NOT an allowlist. Nothing here
 * decides that any service SHOULD be used; E2b, if it happens, needs an explicit reviewed allowlist
 * because an exported service may still have side effects.
 */

/** One declared service component of an installed package, as read from PackageManager metadata. */
data class DeclaredService(
    /** Fully-qualified service class name (`ServiceInfo.name`). */
    val className: String,
    /** `ServiceInfo.exported`: already resolved by the platform, never inferred from intent filters. */
    val exported: Boolean,
    /** `ServiceInfo.enabled`: the component's own enabled state (manifest `android:enabled` plus any override). */
    val componentEnabled: Boolean,
    /** `ApplicationInfo.enabled`: the whole app's enabled state; a disabled app's components are unreachable. */
    val applicationEnabled: Boolean,
    /** `ServiceInfo.permission`: the permission a caller must hold to bind; null means none is required. */
    val permission: String?,
) {
    /** Reachable only if the component and its application are both enabled (`ComponentInfo.isEnabled()`). */
    val enabled: Boolean get() = componentEnabled && applicationEnabled
}

enum class ServiceStatus { POTENTIALLY_BINDABLE, NOT_BINDABLE }

/** Why a service is not potentially bindable. The first failing check in [classify]'s fixed order wins. */
enum class Disqualifier { NOT_EXPORTED, COMPONENT_DISABLED, APPLICATION_DISABLED, PERMISSION_REQUIRED }

/** A declared service with its classification. */
class ServiceInventoryRow(val service: DeclaredService, val status: ServiceStatus, val disqualifier: Disqualifier?) {
    val potentiallyBindable: Boolean get() = status == ServiceStatus.POTENTIALLY_BINDABLE
}

/** The services of one MAPPED package (empty when the package is not installed, not visible, or has no services). */
class PackageInventory(
    val packageName: String,
    val appId: Int,
    val lsappName: String,
    /** True when PackageManager returned the package (installed and visible to AdaPreload). */
    val installed: Boolean,
    val services: List<ServiceInventoryRow>,
) {
    val candidates: List<ServiceInventoryRow> get() = services.filter { it.potentiallyBindable }
}

/** A canonical app id reached by more than one MAPPED package: the bind target would be ambiguous. */
class AmbiguousMappedApp(val appId: Int, val lsappName: String, val packages: List<String>)

/** Everything the E2a inventory found, with the counts the report is built from. */
class ServiceInventoryResult(
    /** MAPPED rows in the mapping table (Amazon's two packages are two rows). */
    val mappedRowCount: Int,
    /** Distinct canonical app ids among MAPPED rows. */
    val mappedAppCount: Int,
    val packages: List<PackageInventory>,
    val ambiguousApps: List<AmbiguousMappedApp>,
) {
    val installedPackages: List<PackageInventory> get() = packages.filter { it.installed }

    /** Distinct canonical app ids that have at least one installed, visible MAPPED package. */
    val installedAppIds: Set<Int> get() = installedPackages.map { it.appId }.toSet()

    /** Candidate count per installed canonical app id (a POTENTIALLY_BINDABLE service of any of its packages). */
    val candidatesPerApp: Map<Int, Int>
        get() = installedAppIds.associateWith { id -> installedPackages.filter { it.appId == id }.sumOf { it.candidates.size } }

    val appsWithNoCandidate: List<Int> get() = candidatesPerApp.filterValues { it == 0 }.keys.sorted()
    val appsWithOneCandidate: List<Int> get() = candidatesPerApp.filterValues { it == 1 }.keys.sorted()
    val appsWithMultipleCandidates: List<Int> get() = candidatesPerApp.filterValues { it > 1 }.keys.sorted()

    /** Installed canonical app ids with at least one candidate service. */
    val warmableAppIds: Set<Int> get() = candidatesPerApp.filterValues { it > 0 }.keys

    /** Every candidate service across all installed packages, for the researcher-review table. */
    val candidates: List<Pair<PackageInventory, ServiceInventoryRow>>
        get() = installedPackages.flatMap { pkg -> pkg.candidates.map { pkg to it } }

    /** Fraction of installed mapped apps that are potentially warmable; 0.0 when none are installed. */
    val coverageOfInstalled: Double
        get() = if (installedAppIds.isEmpty()) 0.0 else warmableAppIds.size.toDouble() / installedAppIds.size
}

/** Supplies the declared services of a package, or null if it is not installed / not visible to AdaPreload. */
fun interface PackageServiceSource {
    fun servicesOf(packageName: String): List<DeclaredService>?
}

object ServiceWarmabilityInventory {

    const val VERSION = "e2a-inventory-1"

    /**
     * Classifies one service. A service is POTENTIALLY_BINDABLE only when it is exported, both the
     * component and its application are enabled, and it requires no permission. Any other case is
     * NOT_BINDABLE with the first failing reason, checked in this order: exported, component
     * enabled, application enabled, permission.
     */
    fun classify(service: DeclaredService): ServiceInventoryRow {
        val disqualifier = when {
            !service.exported -> Disqualifier.NOT_EXPORTED
            !service.componentEnabled -> Disqualifier.COMPONENT_DISABLED
            !service.applicationEnabled -> Disqualifier.APPLICATION_DISABLED
            service.permission != null -> Disqualifier.PERMISSION_REQUIRED
            else -> null
        }
        val status = if (disqualifier == null) ServiceStatus.POTENTIALLY_BINDABLE else ServiceStatus.NOT_BINDABLE
        return ServiceInventoryRow(service, status, disqualifier)
    }

    /**
     * Builds the inventory from the existing [mapping] and a [source] of declared services. Only
     * MAPPED rows are considered (AMBIGUOUS and UNSUPPORTED rows carry no canonical app id and are
     * never eligible). Reuses the mapping's canonical app ids and names; it introduces no second
     * package vocabulary.
     */
    fun build(mapping: PackageMapping, source: PackageServiceSource): ServiceInventoryResult {
        val mapped = mapping.entries.filter { it.status == MappingStatus.MAPPED }
        val packages = mapped.map { entry ->
            val declared = source.servicesOf(entry.packageName)
            PackageInventory(
                packageName = entry.packageName,
                appId = entry.lsappId!!,
                lsappName = entry.lsappName!!,
                installed = declared != null,
                services = declared.orEmpty().map(::classify),
            )
        }
        val ambiguous = mapped.groupBy { it.lsappId!! }
            .filter { it.value.size > 1 }
            .map { (id, rows) -> AmbiguousMappedApp(id, rows.first().lsappName!!, rows.map { it.packageName }) }
            .sortedBy { it.appId }
        return ServiceInventoryResult(
            mappedRowCount = mapped.size,
            mappedAppCount = mapped.map { it.lsappId }.distinct().size,
            packages = packages,
            ambiguousApps = ambiguous,
        )
    }
}
