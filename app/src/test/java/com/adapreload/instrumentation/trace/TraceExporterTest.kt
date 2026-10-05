package com.adapreload.instrumentation.trace

import com.adapreload.instrumentation.trace.TraceTestSupport.hasLauncherEntry
import com.adapreload.instrumentation.trace.TraceTestSupport.mapping
import com.adapreload.instrumentation.trace.TraceTestSupport.pipeline
import com.adapreload.instrumentation.trace.TraceTestSupport.resumed
import com.adapreload.instrumentation.trace.TraceTestSupport.vocabulary
import com.adapreload.instrumentation.trace.TraceTestSupport.windowInfo
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TraceExporterTest {

    private fun recordedTrace(): TraceSnapshot {
        val timeline = listOf(
            resumed(1_000, "com.whatsapp"), resumed(1_500, "com.example.unknown"), resumed(2_000, "com.whatsapp"),
            RawUsageEvent(2_500, 2, "com.whatsapp", null), resumed(11_000, "com.instagram.android"),
        )
        var now = 0L
        val store = InMemoryTraceStore()
        val s = TraceSession(store, FakeEventSource(timeline), pipeline(), { now }, 2_000)
        now = 500; s.openWindow(windowInfo(500))
        now = 5_000; s.poll(hasLauncherEntry)
        s.closeWindow(CloseReason.SERVICE_STOPPED)
        now = 10_000; s.openWindow(windowInfo(10_000))
        now = 15_000; s.poll(hasLauncherEntry)
        return store.snapshot()
    }

    @Test
    fun exportIsDeterministicAndIndependentOfInputOrder() {
        val snapshot = recordedTrace()
        val a = TraceExporter.export(snapshot, vocabulary, mapping, exportedAtMs = 42)
        val b = TraceExporter.export(snapshot, vocabulary, mapping, exportedAtMs = 42)
        val shuffled = snapshot.copy(windows = snapshot.windows.reversed(), records = snapshot.records.reversed())
        assertEquals(a, b)
        assertEquals(a, TraceExporter.export(shuffled, vocabulary, mapping, exportedAtMs = 42))
        assertTrue(a.all { it.code < 128 })
    }

    @Test
    fun exportContainsEverySection() {
        val root = Json.parse(TraceExporter.export(recordedTrace(), vocabulary, mapping, 42)) as Map<*, *>
        assertEquals(
            listOf(
                "format", "format_version", "spec", "exported_at_ms", "exported_at_utc", "vocabulary", "experiment",
                "counts", "mapping", "windows", "gaps", "mapping_decisions", "discarded_summary", "sequence", "events",
            ),
            root.keys.toList(),
        )
        assertEquals(5, (root["events"] as List<*>).size)
        assertEquals(
            listOf("WhatsApp Messenger", "Instagram"),
            (root["sequence"] as List<*>).map { (it as Map<*, *>)["lsapp_name"] },
        )
        val gap = (root["gaps"] as List<*>).single() as Map<*, *>
        assertEquals(listOf(3_000L, 10_000L, 7_000L, "SERVICE_STOPPED"), listOf(gap["start_ms"], gap["end_ms"], gap["duration_ms"], gap["cause"]))
        assertEquals(1L, (root["discarded_summary"] as Map<*, *>)["OOV"])
        val decision = (root["mapping_decisions"] as List<*>).map { it as Map<*, *> }.first { it["package"] == "com.example.unknown" }
        assertEquals("OOV", decision["classification"])
        assertEquals(mapping.sha256, (root["mapping"] as Map<*, *>)["sha256"])
    }
}
