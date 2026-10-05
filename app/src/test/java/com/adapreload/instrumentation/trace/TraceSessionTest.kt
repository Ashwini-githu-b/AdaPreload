package com.adapreload.instrumentation.trace

import com.adapreload.instrumentation.trace.TraceTestSupport.hasLauncherEntry
import com.adapreload.instrumentation.trace.TraceTestSupport.id
import com.adapreload.instrumentation.trace.TraceTestSupport.pipeline
import com.adapreload.instrumentation.trace.TraceTestSupport.resumed
import com.adapreload.instrumentation.trace.TraceTestSupport.sequenceOf
import com.adapreload.instrumentation.trace.TraceTestSupport.windowInfo
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TraceSessionTest {

    private val wa = "com.whatsapp"
    private val ig = "com.instagram.android"
    private val yt = "com.google.android.youtube"

    private var now = 0L
    private fun session(store: TraceStore, source: UsageEventSource) =
        TraceSession(store, source, pipeline(), clock = { now }, settleMs = 2_000)

    @Test
    fun pollingIsIncrementalHalfOpenAndExactlyOnce() {
        val source = FakeEventSource(listOf(resumed(10_500, wa), resumed(12_000, ig), resumed(13_999, yt), resumed(14_000, wa)))
        val store = InMemoryTraceStore()
        val s = session(store, source)
        now = 10_000; s.openWindow(windowInfo(10_000))

        now = 11_000; assertTrue(s.poll(hasLauncherEntry) is TraceSession.PollResult.NotYet)
        now = 14_000; s.poll(hasLauncherEntry) // [10000, 12000)
        now = 16_000; s.poll(hasLauncherEntry) // [12000, 14000)
        now = 30_000; s.poll(hasLauncherEntry) // [14000, 28000)

        assertEquals(listOf(10_000L to 12_000L, 12_000L to 14_000L, 14_000L to 28_000L), source.queried)
        assertEquals(listOf(10_500L, 12_000L, 13_999L, 14_000L), store.records.map { it.record.event.timestampMs })
    }

    @Test
    fun gapDoesNotResetSequenceOrContextAndIsNotBackfilled() {
        val timeline = listOf(
            resumed(1_000, wa), resumed(2_000, ig),   // window 1
            resumed(6_000, yt), resumed(7_000, ig),   // during the gap: never read
            resumed(11_000, ig), resumed(12_000, wa), // window 2; ig collapses with the last pre-gap launch
        )
        val source = FakeEventSource(timeline)
        val store = InMemoryTraceStore()
        val s = session(store, source)

        now = 500; s.openWindow(windowInfo(500))
        now = 5_000; s.poll(hasLauncherEntry) // [500, 3000)
        s.closeWindow(CloseReason.SERVICE_STOPPED)

        now = 10_000; s.openWindow(windowInfo(10_000))
        now = 15_000; s.poll(hasLauncherEntry) // [10000, 13000)
        s.closeWindow(CloseReason.SERVICE_STOPPED)

        assertTrue(source.queried.none { (b, e) -> 6_000L in b until e || 7_000L in b until e })
        val seq = sequenceOf(store.records.map { it.record })
        assertEquals(listOf(id("WhatsApp Messenger"), id("Instagram"), id("WhatsApp Messenger")), seq)
        assertEquals(Outcome.COLLAPSED, store.records.first { it.record.event.timestampMs == 11_000L }.record.outcome)
        // The context of the next event starts with the two pre-gap launches.
        assertEquals(listOf(id("WhatsApp Messenger"), id("Instagram")), contextWindow(seq, 3).take(2))
        assertEquals(listOf(3_000L, 10_000L), store.windows.let { listOf(it[0].endMs!!, it[1].info.startMs) })
    }

    @Test
    fun restartAfterProcessDeathRecoversWindowAndContinuesSequence() {
        val source = FakeEventSource(listOf(resumed(1_000, wa), resumed(2_000, ig), resumed(8_000, yt), resumed(21_000, ig), resumed(22_000, yt)))
        val store = InMemoryTraceStore()

        val first = session(store, source)
        now = 500; first.openWindow(windowInfo(500))
        now = 5_000; first.poll(hasLauncherEntry) // cursor 3000; process then dies without closing

        val second = session(store, source)       // new process, same store
        assertEquals(TraceState(2, id("Instagram"), 2_000), second.state)
        assertEquals(1L, second.recoverUnclosedWindow())
        now = 20_000; second.openWindow(windowInfo(20_000))
        now = 25_000; second.poll(hasLauncherEntry)

        assertEquals(CloseReason.PROCESS_ENDED, store.windows[0].closeReason)
        assertEquals(3_000L, store.windows[0].endMs)
        // 8000 fell in the gap; 21000 Instagram collapses into the pre-gap Instagram.
        assertEquals(listOf(id("WhatsApp Messenger"), id("Instagram"), id("YouTube")), sequenceOf(store.records.map { it.record }))
        assertEquals(listOf(0, 1, 1, 2), store.records.map { it.record.sequencePosition })
    }

    @Test
    fun clockBehindCursorQueriesNothing() {
        val source = FakeEventSource(emptyList())
        val s = session(InMemoryTraceStore(), source)
        now = 100_000; s.openWindow(windowInfo(100_000))
        now = 50_000
        assertTrue(s.poll(hasLauncherEntry) is TraceSession.PollResult.NotYet)
        assertTrue(source.queried.isEmpty())
    }
}
