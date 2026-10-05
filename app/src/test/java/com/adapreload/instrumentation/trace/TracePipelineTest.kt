package com.adapreload.instrumentation.trace

import com.adapreload.instrumentation.trace.TraceTestSupport.HOME
import com.adapreload.instrumentation.trace.TraceTestSupport.IME
import com.adapreload.instrumentation.trace.TraceTestSupport.SELF
import com.adapreload.instrumentation.trace.TraceTestSupport.env
import com.adapreload.instrumentation.trace.TraceTestSupport.hasLauncherEntry
import com.adapreload.instrumentation.trace.TraceTestSupport.id
import com.adapreload.instrumentation.trace.TraceTestSupport.other
import com.adapreload.instrumentation.trace.TraceTestSupport.pipeline
import com.adapreload.instrumentation.trace.TraceTestSupport.resumed
import com.adapreload.instrumentation.trace.TraceTestSupport.sequenceOf
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class TracePipelineTest {

    private fun run(vararg events: RawUsageEvent, state: TraceState = TraceState()) =
        pipeline().process(state, events.toList(), windowId = 1, observedAtMs = 99_000, env = env, hasLauncherEntry = hasLauncherEntry)

    private val wa = "com.whatsapp"
    private val ig = "com.instagram.android"

    @Test
    fun consecutiveDuplicatesCollapseKeepingFirstTimestamp() {
        val r = run(resumed(1000, wa), resumed(1500, wa), resumed(2000, ig), resumed(2100, ig), resumed(3000, wa))

        assertEquals(listOf(id("WhatsApp Messenger"), id("Instagram"), id("WhatsApp Messenger")), sequenceOf(r.records))
        assertEquals(
            listOf(Outcome.APPENDED, Outcome.COLLAPSED, Outcome.APPENDED, Outcome.COLLAPSED, Outcome.APPENDED),
            r.records.map { it.outcome },
        )
        val first = r.records.first { it.outcome == Outcome.APPENDED }
        assertEquals(1000L, first.event.timestampMs)
        // A collapsed launch points at the retained launch it merged into.
        assertEquals(listOf(0, 0, 1, 1, 2), r.records.map { it.sequencePosition })
        assertEquals(TraceState(nextPosition = 3, lastRetainedAppId = id("WhatsApp Messenger"), lastEventTimestampMs = 3000), r.state)
    }

    @Test
    fun oovAndExcludedEventsAreRemovedBeforeCollapseAndCreateNoBoundary() {
        val r = run(
            resumed(1000, wa),
            resumed(1100, "com.example.unknown.app"), // OOV
            resumed(1200, HOME),                       // excluded home
            resumed(1300, "com.google.android.dialer"), // ambiguous
            resumed(1400, wa),
        )
        assertEquals(listOf(id("WhatsApp Messenger")), sequenceOf(r.records))
        assertEquals(
            listOf(null, Classification.OOV, Classification.EXCLUDED_HOME, Classification.AMBIGUOUS, null),
            r.records.map { if (it.outcome == Outcome.DISCARDED) it.classification else null },
        )
        assertEquals(Outcome.COLLAPSED, r.records.last().outcome)
    }

    @Test
    fun onlyActivityResumedEventsAreLaunchCandidates() {
        val r = run(other(1000, wa, type = 2), other(1001, wa, type = 23), resumed(1002, wa))
        assertEquals(listOf(Outcome.NOT_LAUNCH_EVENT, Outcome.NOT_LAUNCH_EVENT, Outcome.APPENDED), r.records.map { it.outcome })
        assertNull(r.records[0].classification)
    }

    @Test
    fun onlyTheDefaultHomeAppIsExcludedAsHome() {
        // Settings declares a fallback HOME activity, but it is not the default home: if mapped, it stays eligible.
        val r = run(resumed(1000, HOME), resumed(1100, "com.android.settings"))
        assertEquals(Classification.EXCLUDED_HOME, r.records[0].classification)
        assertEquals(Classification.SUPPORTED, r.records[1].classification)
        assertEquals(listOf(id("Settings")), sequenceOf(r.records))
    }

    @Test
    fun mappedAppWithoutLauncherEntryIsKeptButUnmappedOneIsExcluded() {
        // com.android.settings has no launcher entry in the test environment but is mapped (mapping precedes the launcher test).
        val r = run(resumed(1000, "com.android.settings"), resumed(1100, "com.example.service.only"))
        assertEquals(Classification.SUPPORTED, r.records[0].classification)
        assertEquals(Classification.EXCLUDED_NON_LAUNCHABLE, r.records[1].classification)
    }

    @Test
    fun adaPreloadItselfIsExcluded() {
        val r = run(resumed(1000, wa), resumed(1100, SELF), resumed(1200, wa))
        assertEquals(Classification.SELF, r.records[1].classification)
        assertEquals(listOf(id("WhatsApp Messenger")), sequenceOf(r.records))
    }

    @Test
    fun selfTakesPrecedenceOverEverythingAndCannotBeMapped() {
        val selfAsHome = env.copy(defaultHomePackage = SELF)
        val r = pipeline().process(TraceState(), listOf(resumed(1, SELF)), 1, 2, selfAsHome, hasLauncherEntry)
        assertEquals(Classification.SELF, r.records.single().classification)
        val error = runCatching { TraceTestSupport.mappingOf(TraceTestSupport.row(SELF, "self", "YouTube")) }.exceptionOrNull()
        assertTrue(error?.message.orEmpty().contains("cannot be mapped"))
    }

    @Test
    fun imeAndExplicitSystemPackagesAreExcluded() {
        val r = run(resumed(1, IME), resumed(2, "com.android.systemui"), resumed(3, "com.google.android.permissioncontroller"))
        assertEquals(
            listOf(Classification.EXCLUDED_IME, Classification.EXCLUDED_SYSTEM, Classification.EXCLUDED_SYSTEM),
            r.records.map { it.classification },
        )
    }

    @Test
    fun eventsKeepStreamOrderAndClockAnomaliesAreFlaggedNotReordered() {
        val r = run(resumed(5000, wa), resumed(4000, ig), resumed(6000, wa))
        assertEquals(listOf(5000L, 4000L, 6000L), r.records.map { it.event.timestampMs })
        assertEquals(listOf(false, true, false), r.records.map { it.timestampAnomaly })
        assertEquals(listOf(id("WhatsApp Messenger"), id("Instagram"), id("WhatsApp Messenger")), sequenceOf(r.records))
    }

    @Test
    fun stateCarriesAcrossBatches() {
        val first = run(resumed(1000, wa))
        val second = run(resumed(2000, wa), resumed(3000, ig), state = first.state)
        assertEquals(Outcome.COLLAPSED, second.records[0].outcome)
        assertEquals(1, second.records[1].sequencePosition)
        assertFalse(second.records.any { it.timestampAnomaly })
    }

    @Test
    fun contextWindowIsTheLast20RetainedLaunchesBeforeTheEvent() {
        val seq = (1..30).toList()
        assertEquals(listOf(1, 2, 3), contextWindow(seq, 3))
        assertEquals((6..25).toList(), contextWindow(seq, 25))
        assertEquals(emptyList<Int>(), contextWindow(seq, 0))
    }
}
