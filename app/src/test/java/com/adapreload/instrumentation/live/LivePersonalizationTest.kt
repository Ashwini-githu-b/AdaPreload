package com.adapreload.instrumentation.live

import com.adapreload.instrumentation.model.Layer1Assets
import com.adapreload.instrumentation.model.Layer1Model
import com.adapreload.instrumentation.model.Layer2Adapter
import com.adapreload.instrumentation.model.Layer2OnlineLearner
import com.adapreload.instrumentation.model.Layer2Prediction
import com.adapreload.instrumentation.model.hiddenFp32
import com.adapreload.instrumentation.model.logitsFp32
import com.adapreload.instrumentation.trace.CloseReason
import com.adapreload.instrumentation.trace.FakeEventSource
import com.adapreload.instrumentation.trace.Outcome
import com.adapreload.instrumentation.trace.RawUsageEvent
import com.adapreload.instrumentation.trace.TraceSession
import com.adapreload.instrumentation.trace.TraceTestSupport.HOME
import com.adapreload.instrumentation.trace.TraceTestSupport.IME
import com.adapreload.instrumentation.trace.TraceTestSupport.SELF
import com.adapreload.instrumentation.trace.TraceTestSupport.hasLauncherEntry
import com.adapreload.instrumentation.trace.TraceTestSupport.id
import com.adapreload.instrumentation.trace.TraceTestSupport.other
import com.adapreload.instrumentation.trace.TraceTestSupport.pipeline
import com.adapreload.instrumentation.trace.TraceTestSupport.resumed
import com.adapreload.instrumentation.trace.TraceTestSupport.sequenceOf
import com.adapreload.instrumentation.trace.TraceTestSupport.vocabulary
import com.adapreload.instrumentation.trace.TraceTestSupport.windowInfo
import com.adapreload.instrumentation.trace.UsageEventSource
import com.adapreload.instrumentation.trace.contextWindow
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Phase D2: Layer 2 driven by the Phase B trace, persisted atomically with each batch (D6-D14).
 *
 * Runs the real TraceSession, TracePipeline and Layer 1 model. [InMemoryLayer2Store] commits
 * all-or-nothing like a SQLite transaction and can fail a commit on demand. "Process death" is a
 * new [Process] (new learner and session) over the same store and event source.
 */
class LivePersonalizationTest {

    companion object {
        val model: Layer1Model by lazy {
            Layer1Model.load(
                File("src/main/assets/${Layer1Assets.MANIFEST}").readText(),
                File("src/main/assets/${Layer1Assets.WEIGHTS}").readBytes(),
            )
        }
    }

    private val wa = "com.whatsapp"
    private val ig = "com.instagram.android"
    private val yt = "com.google.android.youtube"
    private val st = "com.android.settings"
    private val appIds = mapOf(wa to id("WhatsApp Messenger"), ig to id("Instagram"), yt to id("YouTube"), st to id("Settings"))
    private val WA = appIds.getValue(wa)
    private val IG = appIds.getValue(ig)
    private val YT = appIds.getValue(yt)
    private val ST = appIds.getValue(st)

    private var now = 0L

    /** One app process: Layer 2 opened from the store, and a trace session committing through it. */
    private inner class Process(val store: InMemoryLayer2Store, source: UsageEventSource) {
        val personalizer = LivePersonalizer.open(store, model, vocabulary.appsSha256)
        val session = TraceSession(PersonalizedTraceStore(store, personalizer), source, pipeline(), { now }, 2_000)

        init {
            session.recoverUnclosedWindow()
        }

        fun open(at: Long) {
            now = at
            session.openWindow(windowInfo(at))
        }

        fun poll(at: Long) {
            now = at
            session.poll(hasLauncherEntry)
        }
    }

    /** The prequential definition, computed directly: for each launch, reveal it, then predict from the context ending with it. */
    private class Reference(val adapter: Layer2Adapter, val predictions: List<Layer2Prediction>, val losses: List<Double?>)

    private fun reference(sequence: List<Int>): Reference {
        val learner = Layer2OnlineLearner()
        val predictions = ArrayList<Layer2Prediction>()
        val losses = ArrayList<Double?>()
        for ((p, app) in sequence.withIndex()) {
            losses += if (p > 0) learner.reveal(app).lossBefore else null
            val l1 = layer1(contextWindow(sequence, p + 1))
            predictions += learner.predict(l1.hiddenFp32(), l1.logitsFp32())
        }
        return Reference(learner.adapter, predictions, losses)
    }

    private fun layer1(context: List<Int>) = model.predict(IntArray(20 - context.size) + context.toIntArray())

    private fun sequence(store: InMemoryLayer2Store) = sequenceOf(store.trace.records.map { it.record })

    private fun assertSameAdapter(expected: Layer2Adapter, state: Layer2State) {
        assertEquals(expected.updateCount, state.updateCount)
        assertTrue("W bit-identical", expected.weightSnapshot().contentEquals(state.weight))
        assertTrue("b bit-identical", expected.biasSnapshot().contentEquals(state.bias))
    }

    private fun assertSameState(expected: Layer2State, actual: Layer2State) {
        assertEquals(expected.identity, actual.identity)
        assertEquals(expected.updateCount, actual.updateCount)
        assertTrue(expected.weight.contentEquals(actual.weight))
        assertTrue(expected.bias.contentEquals(actual.bias))
        val e = expected.pending
        val a = actual.pending
        if (e == null) {
            assertNull(a)
            return
        }
        assertNotNull(a)
        assertEquals(e.position, a!!.position)
        assertArrayEquals(e.context, a.context)
        assertTrue(e.hidden.contentEquals(a.hidden))
        assertTrue(e.backboneLogits.contentEquals(a.backboneLogits))
        assertTrue(e.finalLogits.contentEquals(a.finalLogits))
    }

    /** A launch every second from 1 s over the four supported apps, never repeating the previous app. */
    private fun launches(count: Int): List<RawUsageEvent> {
        val apps = listOf(wa, ig, yt, st)
        var i = 0
        return (0 until count).map { k ->
            i = (i + 1 + (k * 7 + 3) % 3) % apps.size
            resumed(1_000L + 1_000L * k, apps[i])
        }
    }

    @Test
    fun d6FirstLaunchCreatesAPredictionButNoUpdate() {
        val store = InMemoryLayer2Store()
        val p = Process(store, FakeEventSource(listOf(resumed(1_000, wa))))
        p.open(500)
        p.poll(5_000)

        val entry = store.log.single()
        assertEquals(0, entry.sequencePosition)
        assertNull(entry.reveal)
        assertEquals(0L, entry.updateCount)
        val state = store.loadLayer2()!!
        assertEquals(0L, state.updateCount)
        assertTrue(state.weight.all { it.toRawBits() == 0 } && state.bias.all { it.toRawBits() == 0 })
        val pending = state.pending!!
        assertEquals(0, pending.position)
        assertArrayEquals(intArrayOf(WA), pending.context)
        // With the zero adapter the prediction is Layer 1's.
        assertTrue(layer1(listOf(WA)).logitsFp32().contentEquals(pending.finalLogits))
        assertSameState(p.personalizer.state(), state)
    }

    @Test
    fun d7SecondLaunchRevealsTheFirstPredictionWithExactlyOneUpdate() {
        val timeline = listOf(resumed(1_000, wa), resumed(6_000, ig))
        val store = InMemoryLayer2Store()
        val p = Process(store, FakeEventSource(timeline))
        p.open(500)
        p.poll(5_000) // wa
        val first = store.loadLayer2()!!.pending!!
        p.poll(10_000) // ig

        val reveal = store.log[1].reveal!!
        assertEquals(0, reveal.predictionPosition)
        assertEquals(1L, store.log[1].updateCount)
        val expected = Layer2Adapter()
        val loss = expected.update(first.hidden, first.backboneLogits, IG)
        assertSameAdapter(expected, store.loadLayer2()!!)
        assertEquals(loss, reveal.loss, 0.0)
        assertEquals(LivePersonalizer.rankOf(first.backboneLogits, IG), reveal.layer1Rank)
        assertEquals(reveal.layer1Rank, reveal.layer2Rank) // the revealed prediction was made with the zero adapter

        // Both launches in one batch give exactly the same persisted state and log.
        val oneBatch = InMemoryLayer2Store()
        Process(oneBatch, FakeEventSource(timeline)).apply { open(500); poll(10_000) }
        assertArrayEquals(store.stateBytes, oneBatch.stateBytes)
        assertEquals(store.log, oneBatch.log)
    }

    @Test
    fun d8CollapsedDuplicatesCauseNoPredictionOrUpdate() {
        val timeline = listOf(
            resumed(1_000, wa), resumed(2_000, wa),
            resumed(6_000, ig), resumed(7_000, ig),
            resumed(11_000, ig),
        )
        val store = InMemoryLayer2Store()
        val p = Process(store, FakeEventSource(timeline))
        p.open(500)
        p.poll(5_000)
        p.poll(10_000)
        val before = store.stateBytes!!.copyOf()
        p.poll(15_000) // only a collapsed ig

        assertEquals(3, store.trace.records.count { it.record.outcome == Outcome.COLLAPSED })
        assertEquals(listOf(0, 1), store.log.map { it.sequencePosition })
        assertEquals(2, store.history.size) // the collapse-only batch committed no Layer 2 work
        assertArrayEquals(before, store.stateBytes)
        assertEquals(1L, p.personalizer.updateCount)
        assertSameAdapter(reference(listOf(WA, IG)).adapter, store.loadLayer2()!!)
    }

    @Test
    fun d9UnsupportedOrExcludedEventsDoNotRevealThePendingPrediction() {
        val timeline = listOf(
            resumed(1_000, wa),
            resumed(1_100, "com.example.unknown"),       // OOV
            resumed(1_200, HOME), resumed(1_300, IME), resumed(1_400, SELF),
            resumed(1_500, "com.google.android.dialer"),  // AMBIGUOUS
            resumed(1_600, "com.example.decided"),        // UNSUPPORTED
            resumed(1_700, "com.android.systemui"),       // EXCLUDED_SYSTEM
            resumed(1_800, "com.example.service.only"),   // EXCLUDED_NON_LAUNCHABLE
            other(1_900, ig),                             // not ACTIVITY_RESUMED
            resumed(2_000, wa),                           // collapses: discards are no boundary (A3)
            resumed(6_000, "com.example.unknown"), resumed(6_500, HOME),
            resumed(11_000, ig),
        )
        val store = InMemoryLayer2Store()
        val p = Process(store, FakeEventSource(timeline))
        p.open(500)
        p.poll(5_000)
        val afterFirst = store.stateBytes!!.copyOf()
        p.poll(10_000) // only OOV and HOME
        assertEquals(1, store.history.size)
        assertArrayEquals(afterFirst, store.stateBytes)
        assertEquals(0L, p.personalizer.updateCount)
        p.poll(15_000) // ig

        assertEquals(listOf(WA, IG), sequence(store))
        assertEquals(2, store.log.size)
        assertEquals(0, store.log[1].reveal!!.predictionPosition)
        assertEquals(IG, store.log[1].appId)
        assertSameAdapter(reference(listOf(WA, IG)).adapter, store.loadLayer2()!!)
    }

    @Test
    fun d10PendingPredictionSurvivesAGapAndIsRevealedByTheNextLaunch() {
        for (reason in listOf(CloseReason.SERVICE_STOPPED, CloseReason.USAGE_ACCESS_LOST)) {
            val timeline = listOf(
                resumed(1_000, wa), resumed(2_000, ig),   // window 1
                resumed(6_000, yt), resumed(7_000, st),   // gap: never read
                resumed(11_000, st), resumed(12_000, yt), // window 2
            )
            val source = FakeEventSource(timeline)
            val store = InMemoryLayer2Store()
            val p = Process(store, source)
            p.open(500)
            p.poll(5_000)
            p.session.closeWindow(reason)
            val beforeGap = store.loadLayer2()!!
            assertEquals(1, beforeGap.pending!!.position)

            p.open(10_000)
            p.poll(15_000)

            assertTrue(source.queried.none { (b, e) -> 6_000L in b until e || 7_000L in b until e })
            assertEquals(listOf(WA, IG, ST, YT), sequence(store))
            val reveal = store.log[2].reveal!!
            assertEquals(1, reveal.predictionPosition) // the pre-gap prediction
            assertEquals(LivePersonalizer.rankOf(beforeGap.pending!!.finalLogits, ST), reveal.layer2Rank)
            assertEquals(3L, store.loadLayer2()!!.updateCount)
            assertSameAdapter(reference(listOf(WA, IG, ST, YT)).adapter, store.loadLayer2()!!)
        }

        // The first launch after the gap collapses with the last one before it (T3 across gaps): no reveal.
        val store = InMemoryLayer2Store()
        val p = Process(store, FakeEventSource(listOf(resumed(1_000, wa), resumed(2_000, ig), resumed(11_000, ig), resumed(12_000, yt))))
        p.open(500)
        p.poll(5_000)
        p.session.closeWindow(CloseReason.USAGE_ACCESS_LOST)
        p.open(10_000)
        p.poll(15_000)
        assertEquals(listOf(0, 1, 2), store.log.map { it.sequencePosition })
        assertEquals(1, store.log[2].reveal!!.predictionPosition)
        assertEquals(YT, store.log[2].appId)
        assertSameAdapter(reference(listOf(WA, IG, YT)).adapter, store.loadLayer2()!!)
    }

    @Test
    fun d11PersistedStateRestoresExactly() {
        val source = FakeEventSource(launches(12))
        val store = InMemoryLayer2Store()
        val first = Process(store, source)
        first.open(500)
        first.poll(6_000)
        first.poll(15_000)
        val inMemory = first.personalizer.state()
        assertSameState(inMemory, store.loadLayer2()!!)

        val second = Process(store, source) // a new process
        assertSameState(inMemory, second.personalizer.state())
        assertEquals(first.personalizer.lastLog, second.personalizer.lastLog)
        assertEquals(11, second.personalizer.pendingPrediction!!.position)
        assertEquals(11L, second.personalizer.updateCount)

        // The codec round-trips bit-exactly and rejects damaged or foreign state.
        val bytes = store.stateBytes!!
        assertArrayEquals(bytes, Layer2StateCodec.encode(Layer2StateCodec.decode(bytes)))
        val damaged = bytes.copyOf().also { it[200] = (it[200] + 1).toByte() }
        assertTrue(runCatching { Layer2StateCodec.decode(damaged) }.isFailure)
        val foreign = Layer2StateCodec.decode(bytes).let {
            Layer2State(it.identity.copy(layer1WeightsSha256 = "0".repeat(64)), it.weight, it.bias, it.updateCount, it.pending)
        }
        store.replaceState(Layer2StateCodec.encode(foreign))
        assertTrue(runCatching { Process(store, source) }.exceptionOrNull() is IllegalStateException)
    }

    @Test
    fun d12CrashOrFailedCommitNeverDuplicatesOrLosesAnUpdate() {
        val timeline = listOf(resumed(1_000, wa), resumed(2_000, ig), resumed(6_000, yt), resumed(7_000, st), resumed(11_000, wa))

        // (a) A failed commit leaves trace and learner untouched; the retry processes each launch once.
        run {
            val store = InMemoryLayer2Store()
            val p = Process(store, FakeEventSource(timeline))
            p.open(500)
            p.poll(5_000)
            val committed = store.loadLayer2()!!
            store.failNextCommit = true
            assertTrue(runCatching { p.poll(10_000) }.isFailure)
            assertSameState(committed, p.personalizer.state())
            assertEquals(2, store.trace.records.size)
            assertEquals(3_000L, store.trace.loadState().cursorMs)
            p.poll(10_000) // the same range again
            p.poll(15_000)
            assertEquals(listOf(0, 1, 2, 3, 4), store.log.map { it.sequencePosition })
            assertEquals(4L, store.loadLayer2()!!.updateCount)
            assertSameAdapter(reference(listOf(WA, IG, YT, ST, WA)).adapter, store.loadLayer2()!!)
        }

        // (b) Process death after a commit: the new process resumes the committed state and pending prediction.
        run {
            val store = InMemoryLayer2Store()
            val source = FakeEventSource(timeline)
            Process(store, source).apply { open(500); poll(5_000) } // then dies without closing its window
            val second = Process(store, source)
            assertEquals(CloseReason.PROCESS_ENDED, store.trace.windows[0].closeReason)
            second.open(10_000) // 6000 and 7000 fall in the gap
            second.poll(15_000)
            assertEquals(listOf(WA, IG, WA), sequence(store))
            assertEquals(listOf(0, 1, 2), store.log.map { it.sequencePosition })
            assertEquals(1, store.log[2].reveal!!.predictionPosition)
            assertSameAdapter(reference(listOf(WA, IG, WA)).adapter, store.loadLayer2()!!)
        }

        // (c) Process death with an uncommitted batch: neither its records nor its updates exist afterwards.
        run {
            val store = InMemoryLayer2Store()
            val source = FakeEventSource(timeline)
            val first = Process(store, source)
            first.open(500)
            first.poll(5_000)
            store.failNextCommit = true
            assertTrue(runCatching { first.poll(10_000) }.isFailure) // yt, st never committed; then the process dies
            val second = Process(store, source)
            assertEquals(1L, second.personalizer.updateCount)
            assertEquals(1, second.personalizer.pendingPrediction!!.position)
            second.open(10_000)
            second.poll(15_000)
            assertEquals(listOf(WA, IG, WA), sequence(store))
            assertSameAdapter(reference(listOf(WA, IG, WA)).adapter, store.loadLayer2()!!)
        }

        // (d) Trace and Layer 2 out of step is refused, never repaired.
        run {
            val phaseBOnly = InMemoryLayer2Store()
            TraceSession(phaseBOnly, FakeEventSource(timeline), pipeline(), { now }, 2_000).apply {
                now = 500; openWindow(windowInfo(500))
                now = 5_000; poll(hasLauncherEntry)
            }
            val e = runCatching { LivePersonalizer.open(phaseBOnly, model, vocabulary.appsSha256) }.exceptionOrNull()
            assertTrue(e is IllegalStateException && e.message!!.contains("without Layer 2"))

            val behind = InMemoryLayer2Store()
            Process(behind, FakeEventSource(timeline)).apply { open(500); poll(5_000) }
            // A launch committed around Layer 2.
            TraceSession(behind, FakeEventSource(timeline), pipeline(), { now }, 2_000).apply {
                recoverUnclosedWindow()
                now = 5_500; openWindow(windowInfo(5_500))
                now = 10_000; poll(hasLauncherEntry)
            }
            val e2 = runCatching { LivePersonalizer.open(behind, model, vocabulary.appsSha256) }.exceptionOrNull()
            assertTrue(e2 is IllegalStateException && e2.message!!.contains("processed 2 launches but the trace holds 4"))
        }

        // (e) A launch can never be processed twice, and a stale step is never accepted.
        run {
            val store = InMemoryLayer2Store()
            val p = Process(store, FakeEventSource(timeline))
            p.open(500)
            p.poll(5_000)
            val processed = store.trace.records.first { it.record.outcome == Outcome.APPENDED }.record
            assertTrue(runCatching { p.personalizer.prepare(listOf(processed)) }.exceptionOrNull() is IllegalStateException)
            val next = processed.copy(sequencePosition = 2, appId = YT)
            val step = p.personalizer.prepare(listOf(next))!!
            p.personalizer.accept(step)
            assertTrue(runCatching { p.personalizer.accept(step) }.exceptionOrNull() is IllegalStateException)
        }
    }

    @Test
    fun d13EachPredictionIsRecordedBeforeItsUpdate() {
        val source = FakeEventSource(launches(25))
        val store = InMemoryLayer2Store()
        val p = Process(store, source)
        p.open(500)
        for (t in listOf(4_000L, 4_500L, 9_000L, 10_000L, 20_000L, 30_000L)) p.poll(t) // batches of 1 to 10 launches
        val seq = sequence(store)
        assertEquals(25, seq.size)
        val ref = reference(seq)

        for ((pos, entry) in store.log.withIndex()) {
            assertEquals(pos, entry.sequencePosition)
            // The prediction made after launch pos used exactly pos updates: none from launch pos + 1 onwards.
            assertEquals(pos.toLong(), entry.updateCount)
            assertEquals(ref.predictions[pos].rankedAppIds()[0], entry.layer2Top1)
            if (pos > 0) {
                val r = entry.reveal!!
                // Scored against the recorded (pre-update) prediction, with the loss before the update.
                assertEquals(LivePersonalizer.rankOf(ref.predictions[pos - 1].finalLogits, seq[pos]), r.layer2Rank)
                assertEquals(ref.losses[pos]!!, r.loss, 0.0)
            }
        }
        // Every persisted pending prediction is the pre-update prediction for its position.
        for (state in store.history) {
            val pending = state.pending!!
            assertTrue(ref.predictions[pending.position].finalLogits.contentEquals(pending.finalLogits))
            assertEquals(pending.position.toLong(), state.updateCount)
        }
        assertSameAdapter(ref.adapter, store.loadLayer2()!!)
    }

    @Test
    fun d14ALaunchNeverUpdatesThePredictionMadeFromItself() {
        val source = FakeEventSource(launches(25))
        val store = InMemoryLayer2Store()
        val p = Process(store, source)
        p.open(500)
        p.poll(9_000)
        p.poll(30_000)
        val seq = sequence(store)

        for (entry in store.log.drop(1)) assertEquals(entry.sequencePosition - 1, entry.reveal!!.predictionPosition)
        // After each commit the newest prediction, made from the context ending with the newest launch, is still pending.
        for (state in store.history) {
            val pending = state.pending!!
            assertEquals(seq[pending.position], pending.context.last())
            assertEquals(pending.position.toLong(), state.updateCount)
        }

        // The wrong ordering (each launch also the target of the prediction made from itself) is detectably different.
        val wrong = Layer2Adapter()
        for (pos in seq.indices) {
            val l1 = layer1(contextWindow(seq, pos + 1))
            wrong.update(l1.hiddenFp32(), l1.logitsFp32(), seq[pos])
        }
        val actual = store.loadLayer2()!!
        assertFalse(wrong.weightSnapshot().contentEquals(actual.weight))
        assertSameAdapter(reference(seq).adapter, actual)
    }

    @Test
    fun adapterCopyAndRestoreAreExact() {
        val ref = reference(listOf(WA, IG, YT, ST, WA))
        val restored = Layer2Adapter.restore(ref.adapter.weightSnapshot(), ref.adapter.biasSnapshot(), ref.adapter.updateCount)
        val copy = ref.adapter.copy()
        for (a in listOf(restored, copy)) {
            assertEquals(4L, a.updateCount)
            assertTrue(a.weightSnapshot().contentEquals(ref.adapter.weightSnapshot()))
            assertTrue(a.biasSnapshot().contentEquals(ref.adapter.biasSnapshot()))
        }
        copy.update(layer1(listOf(WA)).hiddenFp32(), layer1(listOf(WA)).logitsFp32(), IG)
        assertEquals(4L, ref.adapter.updateCount) // the copy is independent
        assertTrue(runCatching { Layer2Adapter.restore(FloatArray(10), FloatArray(88), 0) }.isFailure)
    }
}
