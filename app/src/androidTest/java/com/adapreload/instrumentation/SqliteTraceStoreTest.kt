package com.adapreload.instrumentation

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.adapreload.instrumentation.collect.SqliteTraceStore
import com.adapreload.instrumentation.trace.CloseReason
import com.adapreload.instrumentation.trace.EnvironmentSnapshot
import com.adapreload.instrumentation.trace.LaunchClassifier
import com.adapreload.instrumentation.trace.Outcome
import com.adapreload.instrumentation.trace.PackageMapping
import com.adapreload.instrumentation.trace.RawUsageEvent
import com.adapreload.instrumentation.trace.TracePipeline
import com.adapreload.instrumentation.trace.TraceSession
import com.adapreload.instrumentation.trace.TraceState
import com.adapreload.instrumentation.trace.UsageEventSource
import com.adapreload.instrumentation.trace.UsageEventTypes
import com.adapreload.instrumentation.trace.Vocabulary
import com.adapreload.instrumentation.trace.WindowInfo
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The SQLite store must behave like the in-memory store used by the JVM tests: the sequence
 * survives a process restart and a service gap, and gap events are never read.
 * Uses its own database file, never the real trace.
 */
@RunWith(AndroidJUnit4::class)
class SqliteTraceStoreTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val dbName = "trace_store_test.db"
    private lateinit var vocabulary: Vocabulary
    private lateinit var pipeline: TracePipeline

    private val timeline = listOf(
        resumed(1_000, "com.whatsapp"), resumed(2_000, "com.instagram.android"),
        resumed(8_000, "com.google.android.youtube"), // falls in the gap
        resumed(21_000, "com.instagram.android"), resumed(22_000, "com.google.android.youtube"),
    )
    private val source = UsageEventSource { begin, end -> timeline.filter { it.timestampMs in begin until end } }
    private var now = 0L

    @Before
    fun setUp() {
        context.deleteDatabase(dbName)
        vocabulary = Vocabulary.parse(context.assets.open("lsapp_vocabulary.json").use { String(it.readBytes()) })
        val mapping = PackageMapping.parse(
            context.assets.open("package_mapping.tsv").use { it.readBytes() },
            vocabulary,
            LaunchClassifier.SYSTEM_EXCLUSIONS + context.packageName,
        )
        pipeline = TracePipeline(LaunchClassifier(mapping))
    }

    @After
    fun tearDown() {
        context.deleteDatabase(dbName)
    }

    @Test
    fun sequenceSurvivesRestartAndGapWithoutBackfill() {
        val first = SqliteTraceStore(context, dbName)
        TraceSession(first, source, pipeline, { now }, 2_000).apply {
            now = 500; openWindow(window(500))
            now = 5_000; poll { true }
        }
        first.close() // the process dies without closing the window

        val second = SqliteTraceStore(context, dbName)
        val session = TraceSession(second, source, pipeline, { now }, 2_000)
        assertEquals(TraceState(2, vocabulary.idOf("Instagram"), 2_000), session.state)
        session.recoverUnclosedWindow()
        now = 20_000; session.openWindow(window(20_000))
        now = 25_000; session.poll { true }

        val snapshot = second.snapshot()
        assertEquals(CloseReason.PROCESS_ENDED, snapshot.windows[0].closeReason)
        assertEquals(3_000L, snapshot.windows[0].endMs)
        assertEquals(listOf(1_000L, 2_000L, 21_000L, 22_000L), snapshot.records.map { it.record.event.timestampMs })
        assertEquals(
            listOf("WhatsApp Messenger", "Instagram", "YouTube"),
            snapshot.records.filter { it.record.outcome == Outcome.APPENDED }.map { it.record.lsappName },
        )
        assertEquals(listOf(0, 1, 1, 2), snapshot.records.map { it.record.sequencePosition })
        assertEquals(3, second.counts().sequenceLength)
        second.close()
    }

    private fun resumed(ts: Long, pkg: String) = RawUsageEvent(ts, UsageEventTypes.ACTIVITY_RESUMED, pkg, "$pkg.Main")

    private fun window(start: Long) = WindowInfo(
        start, "UTC", EnvironmentSnapshot(context.packageName, "com.example.launcher", emptySet()),
        1, "test", vocabulary.appsSha256, 36, "test",
    )
}
