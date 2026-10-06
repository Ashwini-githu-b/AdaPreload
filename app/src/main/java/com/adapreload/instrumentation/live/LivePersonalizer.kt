package com.adapreload.instrumentation.live

import com.adapreload.instrumentation.model.Layer1Model
import com.adapreload.instrumentation.model.Layer2Adapter
import com.adapreload.instrumentation.model.Layer2OnlineLearner
import com.adapreload.instrumentation.model.hiddenFp32
import com.adapreload.instrumentation.model.logitsFp32
import com.adapreload.instrumentation.trace.Outcome
import com.adapreload.instrumentation.trace.TraceRecord
import com.adapreload.instrumentation.trace.TraceStore

/**
 * Live Layer 2 personalization over the canonical launch sequence (Phase D2).
 *
 * For each APPENDED launch, in sequence order:
 * 1. if a prediction is pending (made after the previous launch), the launch is its target:
 *    score it, then apply exactly one SGD update with the prediction's own inputs;
 * 2. the launch joins the context (the last <= 20 launches, T7);
 * 3. Layer 1 runs on that context and Layer 2 predicts the next launch;
 * 4. that prediction becomes the pending prediction.
 *
 * COLLAPSED, DISCARDED and non-launch records are not launches of the sequence and change
 * nothing. The adapter starts at zero with the sequence (A6, T12) and continues across gaps (A7).
 *
 * [prepare] works on a copy and [accept] makes it current, so the caller can commit the result
 * with the batch first: a failed commit leaves this learner exactly as it was.
 */
class LivePersonalizer private constructor(
    private val layer1: Layer1Model,
    val identity: Layer2Identity,
    private var adapter: Layer2Adapter,
    private var pending: PendingPrediction?,
    private var recent: List<Int>,
    lastLog: Layer2LogEntry?,
) {
    /** Layer 2 work for one batch, not yet current. */
    class Step internal constructor(
        val commit: Layer2Commit,
        internal val basePosition: Int?,
        internal val adapter: Layer2Adapter,
        internal val recent: List<Int>,
    )

    val updateCount: Long get() = adapter.updateCount

    val pendingPrediction: PendingPrediction? get() = pending

    /** The latest committed log entry. */
    var lastLog: Layer2LogEntry? = lastLog
        private set

    /** The current state, as it is persisted. */
    fun state(): Layer2State = Layer2State(identity, adapter.weightSnapshot(), adapter.biasSnapshot(), adapter.updateCount, pending)

    /** Computes the Layer 2 work for [records] without changing this learner; null if they contain no APPENDED launch. */
    fun prepare(records: List<TraceRecord>): Step? {
        val launches = records.filter { it.outcome == Outcome.APPENDED }
        if (launches.isEmpty()) return null

        val learner = Layer2OnlineLearner(adapter.copy())
        var p = pending
        if (p != null) {
            // Re-establish the pending prediction on the copy: same state and inputs, so the same output.
            val same = learner.predict(p.hidden, p.backboneLogits)
            check(same.finalLogits.contentEquals(p.finalLogits)) { "Pending prediction does not reproduce" }
        }
        var context = recent
        val log = ArrayList<Layer2LogEntry>(launches.size)
        for (r in launches) {
            val position = checkNotNull(r.sequencePosition)
            val appId = checkNotNull(r.appId)
            val expected = (p?.position ?: -1) + 1
            check(position == expected) { "Launch at sequence position $position, but Layer 2 expects $expected" }

            val reveal = p?.let {
                val layer1Rank = rankOf(it.backboneLogits, appId)
                val layer2Rank = rankOf(it.finalLogits, appId)
                val update = learner.reveal(appId) // the one update for the prediction made after launch it.position
                Reveal(it.position, layer1Rank, layer2Rank, update.lossBefore)
            }

            context = (context + appId).takeLast(WINDOW)
            val l1 = layer1.predict(IntArray(WINDOW - context.size) + context.toIntArray())
            val hidden = l1.hiddenFp32()
            val logits = l1.logitsFp32()
            val prediction = learner.predict(hidden, logits)
            p = PendingPrediction(position, context.toIntArray(), hidden, logits, prediction.finalLogits)
            log += Layer2LogEntry(position, appId, reveal, learner.adapter.updateCount, top1(logits), top1(prediction.finalLogits))
        }
        val state = Layer2State(identity, learner.adapter.weightSnapshot(), learner.adapter.biasSnapshot(), learner.adapter.updateCount, p)
        return Step(Layer2Commit(state, log), pending?.position, learner.adapter, context)
    }

    /** Makes a prepared step current, after its commit succeeded. */
    fun accept(step: Step) {
        check(step.basePosition == pending?.position) { "Step was prepared from a different state" }
        adapter = step.adapter
        pending = step.commit.state.pending
        recent = step.recent
        lastLog = step.commit.log.last()
    }

    companion object {
        const val WINDOW = 20

        /**
         * Resumes Layer 2 from [store], or starts it at zero for a sequence that has not started.
         * Throws IllegalStateException if the persisted state does not belong to this model and
         * vocabulary or does not match the trace exactly; it is never repaired or reset silently.
         */
        fun open(store: Layer2Store, layer1: Layer1Model, vocabularySha256: String): LivePersonalizer {
            check(layer1.vocabularyAppsSha256 == vocabularySha256) { "Layer 1 was exported with a different vocabulary" }
            val identity = Layer2Identity(Layer1Model.CHECKPOINT_SHA256, layer1.weightsSha256, vocabularySha256, Layer2Adapter.LEARNING_RATE.toRawBits())
            val nextPosition = store.loadState().trace.nextPosition
            val saved = store.loadLayer2()
                ?: run {
                    check(nextPosition == 0) {
                        "The trace already holds $nextPosition launches recorded without Layer 2; " +
                            "Layer 2 must start at zero with the sequence (A6, T12)"
                    }
                    return LivePersonalizer(layer1, identity, Layer2Adapter(), null, emptyList(), null)
                }

            check(saved.identity == identity) { "Layer 2 state was made with a different model or vocabulary: ${saved.identity}" }
            val p = saved.pending
            val processed = p?.let { it.position + 1 } ?: 0
            check(processed == nextPosition) { "Layer 2 has processed $processed launches but the trace holds $nextPosition" }
            // Every launch after the first revealed exactly one prediction.
            check(saved.updateCount == (p?.position?.toLong() ?: 0L)) { "updateCount ${saved.updateCount} does not match $processed launches" }
            val adapter = Layer2Adapter.restore(saved.weight, saved.bias, saved.updateCount)
            val recent = store.recentLaunches(WINDOW)
            if (p != null) {
                check(p.context.toList() == recent) { "Pending prediction context differs from the trace" }
                val l1 = layer1.predict(IntArray(WINDOW - p.context.size) + p.context)
                check(l1.hiddenFp32().contentEquals(p.hidden) && l1.logitsFp32().contentEquals(p.backboneLogits)) {
                    "Pending prediction inputs do not match Layer 1"
                }
                check(adapter.finalLogits(p.hidden, p.backboneLogits).contentEquals(p.finalLogits)) {
                    "Pending prediction does not match the adapter state"
                }
            }
            return LivePersonalizer(layer1, identity, adapter, p, recent, store.lastLayer2Log())
        }

        /** Rank of [appId] among the real apps 1..87 by descending logit; ties rank the lower id first. */
        fun rankOf(logits: FloatArray, appId: Int): Int {
            val z = logits[appId]
            var rank = 1
            for (j in 1 until logits.size) if (j != appId && (logits[j] > z || (logits[j] == z && j < appId))) rank++
            return rank
        }

        private fun top1(logits: FloatArray): Int = (1 until logits.size).first { rankOf(logits, it) == 1 }
    }
}

/**
 * The trace store used while Layer 2 runs. Each batch is committed together with the Layer 2
 * work it causes in one transaction; the learner only advances once that commit succeeded.
 * A batch without an APPENDED launch is committed exactly as in Phase B.
 */
class PersonalizedTraceStore(
    private val inner: Layer2Store,
    val personalizer: LivePersonalizer,
    private val onCommitted: (List<Layer2LogEntry>) -> Unit = {},
) : TraceStore by inner {
    override fun commitBatch(records: List<TraceRecord>, cursorMs: Long, polledAtMs: Long) {
        val step = personalizer.prepare(records)
        if (step == null) {
            inner.commitBatch(records, cursorMs, polledAtMs)
            return
        }
        inner.commitBatch(records, cursorMs, polledAtMs, step.commit)
        personalizer.accept(step)
        // Diagnostics only: the batch is committed, so a failure here must not fail the poll.
        runCatching { onCommitted(step.commit.log) }
    }
}
