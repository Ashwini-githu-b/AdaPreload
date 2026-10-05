package com.adapreload.instrumentation.trace

/**
 * Observation windows and incremental polling (ANDROID_TRACE_SPEC A6, A7).
 *
 * - Each poll queries the half-open range [cursor, now - settle) and commits the processed
 *   events together with the new cursor, so every event is processed exactly once.
 * - The settle delay exists because the platform stamps an event when it is reported but
 *   inserts it later on a handler thread; a range ending at "now" could miss such events.
 * - Opening a window sets the cursor to the window start: events from a gap are never read
 *   (no backfill). The sequence state is not reset, so the sequence, the 20-event context and
 *   the collapse state continue across gaps and restarts.
 */
class TraceSession(
    private val store: TraceStore,
    private val source: UsageEventSource,
    private val pipeline: TracePipeline,
    private val clock: () -> Long,
    private val settleMs: Long = DEFAULT_SETTLE_MS,
) {
    sealed interface PollResult {
        data object NoOpenWindow : PollResult

        /** Nothing to query yet, or the wall clock is behind the cursor. */
        data class NotYet(val cursorMs: Long, val endMs: Long) : PollResult

        data class Polled(val beginMs: Long, val endMs: Long, val events: Int) : PollResult
    }

    private var trace: TraceState
    private var cursorMs: Long?
    private var environment: EnvironmentSnapshot? = null

    var openWindowId: Long?
        private set

    init {
        val persisted = store.loadState()
        trace = persisted.trace
        cursorMs = persisted.cursorMs
        openWindowId = persisted.openWindowId
    }

    val state: TraceState get() = trace

    /**
     * Closes a window left open by a process that ended without closing it, at the upper bound
     * of its last successful query. Call once before opening a new window.
     */
    fun recoverUnclosedWindow(): Long? {
        val id = openWindowId ?: return null
        store.closeWindow(id, checkNotNull(cursorMs), CloseReason.PROCESS_ENDED)
        openWindowId = null
        return id
    }

    fun openWindow(info: WindowInfo): Long {
        check(openWindowId == null) { "A window is already open" }
        val id = store.openWindow(info)
        openWindowId = id
        cursorMs = info.startMs
        environment = info.environment
        return id
    }

    fun poll(hasLauncherEntry: (String) -> Boolean): PollResult {
        val windowId = openWindowId ?: return PollResult.NoOpenWindow
        val env = checkNotNull(environment) { "Open window without environment; call recoverUnclosedWindow first" }
        val begin = checkNotNull(cursorMs)
        val now = clock()
        val end = now - settleMs
        if (end <= begin) return PollResult.NotYet(begin, end)

        val events = source.query(begin, end)
        val result = pipeline.process(trace, events, windowId, now, env, hasLauncherEntry)
        store.commitBatch(result.records, end, now)
        trace = result.state
        cursorMs = end
        return PollResult.Polled(begin, end, events.size)
    }

    /** Closes the open window at the upper bound of its last successful query. */
    fun closeWindow(reason: CloseReason) {
        val id = openWindowId ?: return
        store.closeWindow(id, checkNotNull(cursorMs), reason)
        openWindowId = null
        environment = null
    }

    companion object {
        const val DEFAULT_SETTLE_MS = 2_000L
    }
}
