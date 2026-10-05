package com.adapreload.instrumentation.trace

/**
 * Turns raw usage events into audit records and the canonical launch sequence:
 *
 * raw event → ACTIVITY_RESUMED? → classify package (A4) → discard non-SUPPORTED (A3, no
 * boundary) → collapse consecutive identical app ids (T3, first timestamp kept) → append.
 *
 * Events are processed in the order given (stream order, A5) and never re-sorted.
 */
class TracePipeline(private val classifier: LaunchClassifier) {

    class Result(val records: List<TraceRecord>, val state: TraceState)

    fun process(
        state: TraceState,
        events: List<RawUsageEvent>,
        windowId: Long,
        observedAtMs: Long,
        env: EnvironmentSnapshot,
        hasLauncherEntry: (String) -> Boolean,
    ): Result {
        var s = state
        val records = ArrayList<TraceRecord>(events.size)
        for (e in events) {
            val anomaly = s.lastEventTimestampMs?.let { e.timestampMs < it } ?: false
            s = s.copy(lastEventTimestampMs = e.timestampMs)

            fun record(outcome: Outcome, c: Classification? = null, entry: MappingEntry? = null, position: Int? = null) =
                TraceRecord(windowId, observedAtMs, e, outcome, c, entry?.lsappId, entry?.lsappName, position, anomaly)

            if (e.eventType != UsageEventTypes.ACTIVITY_RESUMED) {
                records += record(Outcome.NOT_LAUNCH_EVENT)
                continue
            }
            val (classification, entry) = classifier.classify(e.packageName, env, hasLauncherEntry)
            if (classification != Classification.SUPPORTED) {
                records += record(Outcome.DISCARDED, classification)
                continue
            }
            val appId = entry!!.lsappId!!
            if (appId == s.lastRetainedAppId) {
                records += record(Outcome.COLLAPSED, classification, entry, s.nextPosition - 1)
            } else {
                records += record(Outcome.APPENDED, classification, entry, s.nextPosition)
                s = s.copy(nextPosition = s.nextPosition + 1, lastRetainedAppId = appId)
            }
        }
        return Result(records, s)
    }
}
