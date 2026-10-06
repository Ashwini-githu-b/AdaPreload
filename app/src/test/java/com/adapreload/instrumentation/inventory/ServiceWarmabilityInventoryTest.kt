package com.adapreload.instrumentation.inventory

import com.adapreload.instrumentation.trace.LaunchClassifier
import com.adapreload.instrumentation.trace.MappingStatus
import com.adapreload.instrumentation.trace.PackageMapping
import com.adapreload.instrumentation.trace.TraceTestSupport
import com.adapreload.instrumentation.trace.TraceTestSupport.SELF
import com.adapreload.instrumentation.trace.TraceTestSupport.id
import com.adapreload.instrumentation.trace.TraceTestSupport.vocabulary
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Phase E2a: the read-only service warmability inventory over fake PackageManager metadata. */
class ServiceWarmabilityInventoryTest {

    private val wa = "com.whatsapp"
    private val ig = "com.instagram.android"
    private val yt = "com.google.android.youtube"
    private val st = "com.android.settings"

    private fun service(
        className: String,
        exported: Boolean = true,
        componentEnabled: Boolean = true,
        applicationEnabled: Boolean = true,
        permission: String? = null,
    ) = DeclaredService(className, exported, componentEnabled, applicationEnabled, permission)

    /** A source over a fixed map; a package absent from the map is "not installed" (null). */
    private fun source(map: Map<String, List<DeclaredService>>) = PackageServiceSource { map[it] }

    private fun build(map: Map<String, List<DeclaredService>>, mapping: PackageMapping = TraceTestSupport.mapping) =
        ServiceWarmabilityInventory.build(mapping, source(map))

    private fun rowFor(result: ServiceInventoryResult, pkg: String, className: String) =
        result.packages.single { it.packageName == pkg }.services.single { it.service.className == className }

    /** 1 */
    @Test
    fun exportedEnabledNoPermissionIsPotentiallyBindable() {
        val r = build(mapOf(wa to listOf(service("com.whatsapp.SyncService"))))
        val row = rowFor(r, wa, "com.whatsapp.SyncService")
        assertEquals(ServiceStatus.POTENTIALLY_BINDABLE, row.status)
        assertNull(row.disqualifier)
        assertTrue(row.potentiallyBindable)
        assertEquals(listOf(id("WhatsApp Messenger")), r.warmableAppIds.toList())
    }

    /** 2 */
    @Test
    fun exportedButDisabledIsNotBindable() {
        val component = build(mapOf(wa to listOf(service("A", componentEnabled = false))))
        assertEquals(Disqualifier.COMPONENT_DISABLED, rowFor(component, wa, "A").disqualifier)
        assertFalse(rowFor(component, wa, "A").potentiallyBindable)

        val application = build(mapOf(wa to listOf(service("B", applicationEnabled = false))))
        assertEquals(Disqualifier.APPLICATION_DISABLED, rowFor(application, wa, "B").disqualifier)
        assertTrue(application.warmableAppIds.isEmpty())
    }

    /** 3 */
    @Test
    fun notExportedIsNotBindable() {
        val r = build(mapOf(wa to listOf(service("A", exported = false))))
        assertEquals(Disqualifier.NOT_EXPORTED, rowFor(r, wa, "A").disqualifier)
        assertEquals(ServiceStatus.NOT_BINDABLE, rowFor(r, wa, "A").status)
    }

    /** 4 */
    @Test
    fun exportedWithPermissionIsNotBindable() {
        val r = build(mapOf(wa to listOf(service("A", permission = "com.whatsapp.permission.BIND"))))
        assertEquals(Disqualifier.PERMISSION_REQUIRED, rowFor(r, wa, "A").disqualifier)
        assertFalse(rowFor(r, wa, "A").potentiallyBindable)
        assertTrue(r.warmableAppIds.isEmpty())
    }

    /** The first failing check wins, in the order exported, component, application, permission. */
    @Test
    fun disqualifierOrderIsFixed() {
        // Not exported AND disabled AND has a permission: NOT_EXPORTED is reported.
        assertEquals(
            Disqualifier.NOT_EXPORTED,
            ServiceWarmabilityInventory.classify(service("A", exported = false, componentEnabled = false, permission = "p")).disqualifier,
        )
        assertEquals(
            Disqualifier.COMPONENT_DISABLED,
            ServiceWarmabilityInventory.classify(service("A", componentEnabled = false, permission = "p")).disqualifier,
        )
    }

    /** 5 */
    @Test
    fun multipleCandidateServicesAreAllReported() {
        val r = build(mapOf(ig to listOf(
            service("com.instagram.A"),
            service("com.instagram.B"),
            service("com.instagram.C", exported = false), // not a candidate
            service("com.instagram.D"),
        )))
        val candidates = r.packages.single { it.packageName == ig }.candidates.map { it.service.className }
        assertEquals(listOf("com.instagram.A", "com.instagram.B", "com.instagram.D"), candidates)
        assertEquals(mapOf(id("Instagram") to 3), r.candidatesPerApp.filterValues { it > 0 })
        assertEquals(listOf(id("Instagram")), r.appsWithMultipleCandidates)
        // The inventory lists every candidate; it does not select one.
        assertEquals(3, r.candidates.size)
    }

    /** 6 */
    @Test
    fun unmappedPackageIsNotEligible() {
        // An AMBIGUOUS-status row (google_phone) carries no canonical id and is never inventoried,
        // and a package with no mapping row at all is not queried.
        val r = build(mapOf(
            "com.google.android.dialer" to listOf(service("Dialer")),   // AMBIGUOUS row
            "com.example.unknown" to listOf(service("Unknown")),        // no row at all
            wa to listOf(service("A")),
        ))
        assertTrue(r.packages.none { it.packageName == "com.google.android.dialer" })
        assertTrue(r.packages.none { it.packageName == "com.example.unknown" })
        assertEquals(setOf(wa), r.packages.filter { it.installed }.map { it.packageName }.toSet())
    }

    /** 7 */
    @Test
    fun ambiguousMappingIsReported() {
        // Two packages mapped to the same canonical app (as Amazon Shopping is in the real table).
        val mapping = TraceTestSupport.mappingOf(
            TraceTestSupport.row("com.a.youtube", "yt_a", "YouTube"),
            TraceTestSupport.row("com.b.youtube", "yt_b", "YouTube"),
            TraceTestSupport.row(wa, "whatsapp", "WhatsApp Messenger"),
        )
        val r = build(mapOf("com.a.youtube" to listOf(service("A")), wa to listOf(service("B"))), mapping)
        assertEquals(1, r.ambiguousApps.size)
        val amb = r.ambiguousApps.single()
        assertEquals(id("YouTube"), amb.appId)
        assertEquals(listOf("com.a.youtube", "com.b.youtube"), amb.packages)
        // Both packages are inventoried under the same app id; only the installed one contributes.
        assertEquals(2, r.packages.count { it.appId == id("YouTube") })
        assertEquals(1, r.packages.count { it.appId == id("YouTube") && it.installed })
    }

    /** 8 */
    @Test
    fun missingPackageOrServiceMetadataIsHandledSafely() {
        // wa installed with no services; yt not installed at all (absent from the source map).
        val r = build(mapOf(wa to emptyList(), st to listOf(service("S"))))
        val waPkg = r.packages.single { it.packageName == wa }
        assertTrue(waPkg.installed)
        assertTrue(waPkg.services.isEmpty())
        assertTrue(waPkg.candidates.isEmpty())

        val ytPkg = r.packages.single { it.packageName == yt }
        assertFalse(ytPkg.installed)
        assertTrue(ytPkg.services.isEmpty())

        // Only installed packages count toward coverage; a not-installed package is excluded.
        assertEquals(setOf(id("WhatsApp Messenger"), id("Settings")), r.installedAppIds)
        assertEquals(setOf(id("Settings")), r.warmableAppIds)
        assertEquals(0.5, r.coverageOfInstalled, 1e-9)
        assertEquals(listOf(id("WhatsApp Messenger")), r.appsWithNoCandidate)
    }

    @Test
    fun coverageIsZeroWhenNothingIsInstalled() {
        val r = build(emptyMap())
        assertTrue(r.installedAppIds.isEmpty())
        assertEquals(0.0, r.coverageOfInstalled, 0.0)
        assertEquals(4, r.mappedAppCount) // the four MAPPED apps in the test mapping
    }

    /** The inventory runs over the shipped mapping table and reuses its canonical ids and names. */
    @Test
    fun runsOverTheRealMappingAsset() {
        val mapping = PackageMapping.parse(
            TraceTestSupport.asset("package_mapping.tsv").readBytes(), vocabulary, LaunchClassifier.SYSTEM_EXCLUSIONS + SELF,
        )
        val mappedRows = mapping.entries.count { it.status == MappingStatus.MAPPED }
        // Nothing "installed": the source reports every package as absent.
        val r = ServiceWarmabilityInventory.build(mapping) { null }
        assertEquals(mappedRows, r.mappedRowCount)
        assertEquals(mappedRows, r.packages.size)
        assertTrue(r.installedPackages.isEmpty())
        // Amazon Shopping is the one canonical id with two MAPPED packages.
        assertEquals(1, r.ambiguousApps.size)
        assertEquals(id("Amazon Shopping"), r.ambiguousApps.single().appId)
        assertEquals(mappedRows - 1, r.mappedAppCount) // one id carries two rows
        // Every MAPPED row carries a canonical id and name (used instead of a second vocabulary).
        assertTrue(r.packages.all { it.lsappName.isNotEmpty() && it.appId in 1..vocabulary.appCount })
    }

    @Test
    fun reportRendersWithoutCandidatesAndWithThem() {
        val empty = InventoryReport.text(build(mapOf(wa to listOf(service("A", exported = false)))))
        assertTrue(empty.contains("D. Potentially bindable services:"))
        assertTrue(empty.contains("(none)"))

        val r = build(mapOf(wa to listOf(service("com.whatsapp.Sync")), ig to listOf(service("com.insta.Sync"))))
        val text = InventoryReport.text(r)
        assertTrue(text.contains("com.whatsapp"))
        assertTrue(text.contains("Sync"))
        val json = InventoryReport.json(r)
        assertTrue(json.contains("adapreload-e2a-service-inventory"))
        assertTrue(json.contains("\"potentially_bindable\": true"))
        assertTrue(json.contains("\"coverage_of_installed_pct\""))
    }
}
