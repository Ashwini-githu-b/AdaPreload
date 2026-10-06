package com.adapreload.instrumentation.live

import com.adapreload.instrumentation.shadow.ShadowDecisionRecord
import com.adapreload.instrumentation.shadow.ShadowPreloadDecision
import com.adapreload.instrumentation.trace.InMemoryTraceStore
import com.adapreload.instrumentation.trace.TraceRecord
import com.adapreload.instrumentation.trace.TraceStore
import com.adapreload.instrumentation.trace.TraceTestSupport

/**
 * [Layer2Store] in memory with all-or-nothing commits, like one SQLite transaction. The state is
 * kept in its encoded form, so every load goes through [Layer2StateCodec] as it does on the device.
 */
class InMemoryLayer2Store(val trace: InMemoryTraceStore = InMemoryTraceStore()) : Layer2Store, TraceStore by trace {
    var stateBytes: ByteArray? = null
        private set
    val log = mutableListOf<Layer2LogEntry>()

    /** Every committed Layer 2 state, oldest first. */
    val history = mutableListOf<Layer2State>()

    /** Phase E1 shadow decisions, in commit order. */
    val shadow = mutableListOf<ShadowDecisionRecord>()

    /** Makes the next commit fail before anything is written, as a rolled-back transaction would. */
    var failNextCommit = false

    override fun loadLayer2(): Layer2State? = stateBytes?.let(Layer2StateCodec::decode)

    override fun recentLaunches(limit: Int): List<Int> = TraceTestSupport.sequenceOf(trace.records.map { it.record }).takeLast(limit)

    override fun lastLayer2Log(): Layer2LogEntry? = log.lastOrNull()

    override fun commitBatch(records: List<TraceRecord>, cursorMs: Long, polledAtMs: Long) {
        failIfRequested()
        trace.commitBatch(records, cursorMs, polledAtMs)
    }

    override fun commitBatch(
        records: List<TraceRecord>,
        cursorMs: Long,
        polledAtMs: Long,
        layer2: Layer2Commit,
        shadow: List<ShadowPreloadDecision>,
    ) {
        failIfRequested()
        // The layer2_log primary key: one prediction per launch.
        check(layer2.log.none { e -> log.any { it.sequencePosition == e.sequencePosition } }) { "Duplicate layer2_log position" }
        // The shadow_decisions primary key: one decision per (prediction, rank).
        val keys = (this.shadow.map { it.decision } + shadow).map { it.predictionPosition to it.rank }
        check(keys.size == keys.toSet().size) { "Duplicate shadow_decisions key" }
        val bytes = Layer2StateCodec.encode(layer2.state)
        trace.commitBatch(records, cursorMs, polledAtMs)
        stateBytes = bytes
        log += layer2.log
        history += Layer2StateCodec.decode(bytes)
        this.shadow += shadow.map { ShadowDecisionRecord(it, polledAtMs, null) }
        for (e in layer2.log) {
            val reveal = e.reveal ?: continue
            this.shadow.replaceAll { if (it.decision.predictionPosition == reveal.predictionPosition) it.copy(actualAppId = e.appId) else it }
        }
    }

    override fun shadowDecisions(): List<ShadowDecisionRecord> =
        shadow.sortedWith(compareBy({ it.decision.predictionPosition }, { it.decision.rank }))

    /** Overwrites the stored state, as damage or a foreign database would. */
    fun replaceState(bytes: ByteArray) {
        stateBytes = bytes
    }

    private fun failIfRequested() {
        if (failNextCommit) {
            failNextCommit = false
            throw IllegalStateException("Simulated commit failure (transaction rolled back)")
        }
    }
}
