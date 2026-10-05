package com.adapreload.instrumentation.trace

import java.time.Instant

/**
 * Deterministic JSON export of the trace: the same snapshot, vocabulary, mapping and export
 * time always produce byte-identical output, whatever order the snapshot lists come in.
 */
object TraceExporter {
    const val FORMAT = "adapreload-trace-export"
    const val FORMAT_VERSION = 1

    fun export(
        snapshot: TraceSnapshot,
        vocabulary: Vocabulary,
        mapping: PackageMapping,
        exportedAtMs: Long,
    ): String {
        val windows = snapshot.windows.sortedBy { it.id }
        val records = snapshot.records.sortedBy { it.id }
        val candidates = records.filter { it.record.classification != null }

        val root = linkedMapOf<String, Any?>(
            "format" to FORMAT,
            "format_version" to FORMAT_VERSION,
            "spec" to "docs/ANDROID_TRACE_SPEC.md",
            "exported_at_ms" to exportedAtMs,
            "exported_at_utc" to utc(exportedAtMs),
            "vocabulary" to linkedMapOf("apps_sha256" to vocabulary.appsSha256, "app_count" to vocabulary.appCount),
            "experiment" to linkedMapOf(
                "start_ms" to snapshot.experimentStartMs,
                "start_utc" to snapshot.experimentStartMs?.let(::utc),
                "cursor_ms" to snapshot.cursorMs,
                "last_poll_ms" to snapshot.lastPollMs,
            ),
            "counts" to counts(TraceCounts.of(records.map { it.record }, windows.size, snapshot.lastPollMs)),
            "mapping" to linkedMapOf(
                "sha256" to mapping.sha256,
                "entries" to mapping.entries.map(::mappingEntry),
            ),
            "windows" to windows.map(::window),
            "gaps" to gaps(windows),
            "mapping_decisions" to mappingDecisions(candidates),
            "discarded_summary" to discardedSummary(candidates),
            "sequence" to records.filter { it.record.outcome == Outcome.APPENDED }
                .sortedBy { it.record.sequencePosition }
                .map(::sequenceEntry),
            "events" to records.map(::event),
        )
        return Json.write(root)
    }

    private fun utc(ms: Long): String = Instant.ofEpochMilli(ms).toString()

    private fun counts(c: TraceCounts) = linkedMapOf<String, Any?>(
        "observed_events" to c.observedEvents,
        "launch_candidates" to c.launchCandidates,
        "supported_launches" to c.supportedLaunches,
        "sequence_length" to c.sequenceLength,
        "collapsed" to c.collapsed,
        "unmapped" to c.unmapped,
        "excluded" to c.excluded,
        "windows" to c.windows,
    )

    private fun mappingEntry(e: MappingEntry) = linkedMapOf<String, Any?>(
        "package" to e.packageName,
        "canonical_app" to e.canonicalApp,
        "lsapp_name" to e.lsappName,
        "lsapp_id" to e.lsappId,
        "status" to e.status.name,
        "confidence" to e.confidence,
        "source" to e.source,
        "notes" to e.notes,
    )

    private fun window(w: ObservationWindow) = linkedMapOf<String, Any?>(
        "id" to w.id,
        "start_ms" to w.info.startMs,
        "start_utc" to utc(w.info.startMs),
        "end_ms" to w.endMs,
        "end_utc" to w.endMs?.let(::utc),
        "close_reason" to w.closeReason?.name,
        "time_zone" to w.info.timeZoneId,
        "default_home_package" to w.info.environment.defaultHomePackage,
        "ime_packages" to w.info.environment.imePackages.sorted(),
        "self_package" to w.info.environment.selfPackage,
        "profile_count" to w.info.profileCount,
        "mapping_sha256" to w.info.mappingSha256,
        "vocabulary_sha256" to w.info.vocabularySha256,
        "api_level" to w.info.apiLevel,
        "device" to w.info.device,
    )

    /** Intervals between consecutive windows, during which nothing was observed (A7). */
    private fun gaps(windows: List<ObservationWindow>) = windows.zipWithNext().mapNotNull { (a, b) ->
        val end = a.endMs ?: return@mapNotNull null
        linkedMapOf<String, Any?>(
            "after_window" to a.id,
            "before_window" to b.id,
            "start_ms" to end,
            "end_ms" to b.info.startMs,
            "duration_ms" to b.info.startMs - end,
            "cause" to a.closeReason?.name,
        )
    }

    /** How each observed package was classified, with counts; drives mapping work. */
    private fun mappingDecisions(candidates: List<StoredRecord>) =
        candidates.groupBy { it.record.event.packageName to it.record.classification!! }
            .toSortedMap(compareBy<Pair<String, Classification>> { it.first }.thenBy { it.second.ordinal })
            .map { (key, group) ->
                val r = group.first().record
                linkedMapOf<String, Any?>(
                    "package" to key.first,
                    "classification" to key.second.name,
                    "app_id" to r.appId,
                    "lsapp_name" to r.lsappName,
                    "launch_candidates" to group.size,
                    "first_seen_ms" to group.first().record.event.timestampMs,
                    "last_seen_ms" to group.last().record.event.timestampMs,
                )
            }

    private fun discardedSummary(candidates: List<StoredRecord>) =
        Classification.entries.filter { it != Classification.SUPPORTED }.associateTo(linkedMapOf<String, Any?>()) { c ->
            c.name to candidates.count { it.record.classification == c }
        }

    private fun sequenceEntry(s: StoredRecord) = linkedMapOf<String, Any?>(
        "position" to s.record.sequencePosition,
        "app_id" to s.record.appId,
        "lsapp_name" to s.record.lsappName,
        "package" to s.record.event.packageName,
        "timestamp_ms" to s.record.event.timestampMs,
        "timestamp_utc" to utc(s.record.event.timestampMs),
        "event_id" to s.id,
        "window_id" to s.record.windowId,
    )

    private fun event(s: StoredRecord): Map<String, Any?> {
        val r = s.record
        return linkedMapOf(
            "id" to s.id,
            "window_id" to r.windowId,
            "timestamp_ms" to r.event.timestampMs,
            "timestamp_utc" to utc(r.event.timestampMs),
            "observed_at_ms" to r.observedAtMs,
            "event_type" to r.event.eventType,
            "event_type_name" to UsageEventTypes.name(r.event.eventType),
            "package" to r.event.packageName,
            "class" to r.event.className,
            "outcome" to r.outcome.name,
            "classification" to r.classification?.name,
            "app_id" to r.appId,
            "lsapp_name" to r.lsappName,
            "sequence_position" to r.sequencePosition,
            "timestamp_anomaly" to r.timestampAnomaly,
        )
    }
}
