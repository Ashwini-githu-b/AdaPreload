package com.adapreload.instrumentation.prewarm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Unit tests for the DEBUG-ONLY E2b prewarm harness (pure parts only; the bind itself needs a device).
 * They pin the allowlist, the single approved candidate, the verified Custom Tabs action, the log
 * format, and — by scanning the harness source — that it only ever binds/unbinds/warms up and never
 * navigates or launches.
 */
class PrewarmLabTest {

    @Test
    fun onlyBraveIsApprovedAndCarriesTheVerifiedCustomTabsAction() {
        val brave = PrewarmCandidate.BRAVE_CUSTOMTABS
        assertTrue(brave.approved)
        assertEquals("com.brave.browser", brave.pkg)
        assertEquals("org.chromium.chrome.browser.customtabs.CustomTabsConnectionService", brave.service)
        assertEquals("android.support.customtabs.action.CustomTabsService", brave.action)
        assertEquals(CUSTOM_TABS_CONNECTION_ACTION, brave.action)
        assertEquals(listOf(PrewarmCandidate.BRAVE_CUSTOMTABS), PrewarmCandidate.approved)
    }

    @Test
    fun theOtherCandidatesAreListedButNotApprovedAndAssertNoAction() {
        val others = PrewarmCandidate.entries.filterNot { it.approved }
        assertEquals(5, others.size) // Snapchat, Drive, Play prewarm, Spotify, Play keep-alive
        // No action is asserted for an unapproved candidate without evidence.
        for (c in others) assertNull("${c.display} must not assert an action", c.action)
        // The six are exactly the E2a priority list.
        assertEquals(
            listOf("com.brave.browser", "com.snapchat.android", "com.google.android.apps.docs",
                "com.android.vending", "com.spotify.music", "com.android.vending"),
            PrewarmCandidate.entries.map { it.pkg },
        )
    }

    @Test
    fun logLineIsStructuredAndStable() {
        val line = prewarmLogLine(1234L, PrewarmCandidate.BRAVE_CUSTOMTABS, "bind_attempt=true bind_success=true warmup=false")
        assertEquals(
            "E2B_VALIDATE trial=1234 candidate=Brave pkg=com.brave.browser " +
                "service=org.chromium.chrome.browser.customtabs.CustomTabsConnectionService " +
                "action=android.support.customtabs.action.CustomTabsService bind_attempt=true bind_success=true warmup=false",
            line,
        )
        // Null candidate (e.g. an unbind with nothing bound) degrades cleanly.
        assertTrue(prewarmLogLine(0L, null, "unbind_skipped=not_bound").startsWith("E2B_VALIDATE trial=0 candidate=- pkg=-"))
    }

    @Test
    fun theHarnessOnlyBindsAndWarmsUpAndNeverNavigatesOrLaunches() {
        // Scan code only, not comments (the KDoc legitimately names the methods it must not call).
        val code = File("src/debug/java/com/adapreload/instrumentation/prewarm/PrewarmLab.kt")
            .readLines()
            .map { it.substringBefore("//") } // drop trailing inline comments
            .filterNot { val t = it.trim(); t.startsWith("*") || t.startsWith("/*") } // drop block-comment lines
            .joinToString("\n")
        assertTrue(code.contains("bindService"))
        assertTrue(code.contains("unbindService"))
        assertTrue(code.contains("warmup(0L)"))
        for (forbidden in listOf("mayLaunchUrl", "launchUrl", "startActivity", "sendBroadcast", "\\.navigate", "newSession")) {
            assertFalse("harness must not call $forbidden", Regex(forbidden).containsMatchIn(code))
        }
    }
}
