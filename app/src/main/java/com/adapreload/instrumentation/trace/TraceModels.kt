package com.adapreload.instrumentation.trace

/** One event exactly as returned by UsageStatsManager.queryEvents, kept in stream order. */
data class RawUsageEvent(
    val timestampMs: Long,
    val eventType: Int,
    val packageName: String,
    val className: String?,
)

/** UsageEvents.Event type values (AOSP UsageEvents.java), kept here so the core has no Android import. */
object UsageEventTypes {
    /** ACTIVITY_RESUMED; the same value is named MOVE_TO_FOREGROUND on API 26-28 (spec A1). */
    const val ACTIVITY_RESUMED = 1

    private val NAMES = mapOf(
        1 to "ACTIVITY_RESUMED", 2 to "ACTIVITY_PAUSED", 5 to "CONFIGURATION_CHANGE",
        7 to "USER_INTERACTION", 8 to "SHORTCUT_INVOCATION", 11 to "STANDBY_BUCKET_CHANGED",
        15 to "SCREEN_INTERACTIVE", 16 to "SCREEN_NON_INTERACTIVE", 17 to "KEYGUARD_SHOWN",
        18 to "KEYGUARD_HIDDEN", 19 to "FOREGROUND_SERVICE_START", 20 to "FOREGROUND_SERVICE_STOP",
        23 to "ACTIVITY_STOPPED", 26 to "DEVICE_SHUTDOWN", 27 to "DEVICE_STARTUP",
    )

    fun name(type: Int): String = NAMES[type] ?: "TYPE_$type"
}

/** Classification of an ACTIVITY_RESUMED launch candidate (spec A4, precedence in [LaunchClassifier]). */
enum class Classification(val group: Group) {
    SUPPORTED(Group.SUPPORTED),
    SELF(Group.EXCLUDED),
    EXCLUDED_HOME(Group.EXCLUDED),
    EXCLUDED_IME(Group.EXCLUDED),
    EXCLUDED_SYSTEM(Group.EXCLUDED),
    AMBIGUOUS(Group.UNMAPPED),
    UNSUPPORTED(Group.UNMAPPED),
    EXCLUDED_NON_LAUNCHABLE(Group.EXCLUDED),
    OOV(Group.UNMAPPED);

    enum class Group { SUPPORTED, EXCLUDED, UNMAPPED }
}

/** What the pipeline did with one raw event. */
enum class Outcome {
    /** Not an ACTIVITY_RESUMED event: raw trace only. */
    NOT_LAUNCH_EVENT,

    /** Launch candidate that is not SUPPORTED; dropped without creating a boundary (A3). */
    DISCARDED,

    /** SUPPORTED, but the same app as the last retained launch (T3). */
    COLLAPSED,

    /** SUPPORTED and appended to the canonical launch sequence. */
    APPENDED,
}

/** One processed raw event: the audit record (AU1, AU2). */
data class TraceRecord(
    val windowId: Long,
    /** Wall-clock time of the poll that observed the event. */
    val observedAtMs: Long,
    val event: RawUsageEvent,
    val outcome: Outcome,
    /** Set for launch candidates only. */
    val classification: Classification?,
    /** Set when SUPPORTED. */
    val appId: Int?,
    val lsappName: String?,
    /** APPENDED: this launch's position. COLLAPSED: position of the retained launch it merged into. */
    val sequencePosition: Int?,
    /** Timestamp lower than the previous event's (e.g. a clock change); never reordered (A5). */
    val timestampAnomaly: Boolean,
)

/** Sequence state carried across polls, service gaps and restarts (A7). */
data class TraceState(
    val nextPosition: Int = 0,
    val lastRetainedAppId: Int? = null,
    val lastEventTimestampMs: Long? = null,
)

/** Device state used to classify launches, resolved at each observation-window start (A4). */
data class EnvironmentSnapshot(
    val selfPackage: String,
    /** The current default home app only, not every HOME-capable app. */
    val defaultHomePackage: String?,
    val imePackages: Set<String>,
)

enum class CloseReason {
    SERVICE_STOPPED,
    USAGE_ACCESS_LOST,

    /** The process ended without closing the window (killed or crashed); closed at the next start. */
    PROCESS_ENDED,
}

/** Metadata recorded when an observation window opens (AU3). */
data class WindowInfo(
    val startMs: Long,
    val timeZoneId: String,
    val environment: EnvironmentSnapshot,
    val profileCount: Int,
    val mappingSha256: String,
    val vocabularySha256: String,
    val apiLevel: Int,
    val device: String,
)

data class ObservationWindow(
    val id: Long,
    val info: WindowInfo,
    /** Upper bound of the last successful query; null while the window is open. */
    val endMs: Long?,
    val closeReason: CloseReason?,
)

data class StoredRecord(val id: Long, val record: TraceRecord)

/** Persistent observation state; the sequence state is derived from the stored records. */
data class PersistedState(
    val trace: TraceState,
    val cursorMs: Long?,
    val openWindowId: Long?,
    val experimentStartMs: Long?,
    val lastPollMs: Long?,
)

/** Everything needed to export or audit the trace. */
data class TraceSnapshot(
    val experimentStartMs: Long?,
    val cursorMs: Long?,
    val lastPollMs: Long?,
    val windows: List<ObservationWindow>,
    val records: List<StoredRecord>,
)

data class TraceCounts(
    val observedEvents: Int = 0,
    val launchCandidates: Int = 0,
    val supportedLaunches: Int = 0,
    val sequenceLength: Int = 0,
    val collapsed: Int = 0,
    val unmapped: Int = 0,
    val excluded: Int = 0,
    val windows: Int = 0,
    val lastCandidate: TraceRecord? = null,
    val lastPollMs: Long? = null,
) {
    companion object {
        fun of(records: List<TraceRecord>, windows: Int, lastPollMs: Long?): TraceCounts {
            val candidates = records.filter { it.classification != null }
            return TraceCounts(
                observedEvents = records.size,
                launchCandidates = candidates.size,
                supportedLaunches = candidates.count { it.classification == Classification.SUPPORTED },
                sequenceLength = records.count { it.outcome == Outcome.APPENDED },
                collapsed = records.count { it.outcome == Outcome.COLLAPSED },
                unmapped = candidates.count { it.classification!!.group == Classification.Group.UNMAPPED },
                excluded = candidates.count { it.classification!!.group == Classification.Group.EXCLUDED },
                windows = windows,
                lastCandidate = candidates.lastOrNull(),
                lastPollMs = lastPollMs,
            )
        }
    }
}

/** Source of raw usage events for a half-open time range [beginMs, endMs). */
fun interface UsageEventSource {
    fun query(beginMs: Long, endMs: Long): List<RawUsageEvent>
}

/** Persistence used by [TraceSession]. Each call is atomic. */
interface TraceStore {
    fun loadState(): PersistedState

    /** Records the window, makes it the open window and sets the cursor to its start (no backfill). */
    fun openWindow(info: WindowInfo): Long

    fun closeWindow(windowId: Long, endMs: Long, reason: CloseReason)

    /** Appends the records and advances the cursor in one transaction (exactly-once processing). */
    fun commitBatch(records: List<TraceRecord>, cursorMs: Long, polledAtMs: Long)

    fun snapshot(): TraceSnapshot

    fun counts(): TraceCounts
}

/** T7: the context for event index [eventIndex] is ids[max(0, eventIndex - window) : eventIndex]. */
fun contextWindow(sequence: List<Int>, eventIndex: Int, window: Int = 20): List<Int> =
    sequence.subList(maxOf(0, eventIndex - window), eventIndex)
