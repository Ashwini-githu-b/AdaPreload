package com.adapreload.instrumentation.live

import com.adapreload.instrumentation.model.Layer1Model
import com.adapreload.instrumentation.model.Layer2Prediction
import com.adapreload.instrumentation.shadow.CandidateResolution
import com.adapreload.instrumentation.shadow.CandidateResolver
import com.adapreload.instrumentation.shadow.DecisionReason
import com.adapreload.instrumentation.shadow.Eligibility
import com.adapreload.instrumentation.shadow.MappingCandidateResolver
import com.adapreload.instrumentation.shadow.ShadowDecision
import com.adapreload.instrumentation.shadow.ShadowPolicyConfig
import com.adapreload.instrumentation.shadow.ShadowPreloadDecision
import com.adapreload.instrumentation.shadow.ShadowPreloadPolicy
import com.adapreload.instrumentation.trace.FakeEventSource
import com.adapreload.instrumentation.trace.RawUsageEvent
import com.adapreload.instrumentation.trace.TraceSession
import com.adapreload.instrumentation.trace.TraceTestSupport
import com.adapreload.instrumentation.trace.TraceTestSupport.hasLauncherEntry
import com.adapreload.instrumentation.trace.TraceTestSupport.pipeline
import com.adapreload.instrumentation.trace.TraceTestSupport.resumed
import com.adapreload.instrumentation.trace.TraceTestSupport.sequenceOf
import com.adapreload.instrumentation.trace.TraceTestSupport.vocabulary
import com.adapreload.instrumentation.trace.TraceTestSupport.windowInfo
import com.adapreload.instrumentation.trace.UsageEventSource
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Phase E1 on the live pipeline: the shadow policy reads the Layer 2 predictions, its decisions
 * are committed with the batch, and it changes nothing about Layer 2 (tests 11, 12) and starts
 * nothing on the device (zero side effects).
 */
class ShadowLivePolicyTest {

    private val model: Layer1Model get() = LivePersonalizationTest.model
    private var now = 0L

    private inner class Process(
        val store: InMemoryLayer2Store,
        source: UsageEventSource,
        policy: ShadowPreloadPolicy?,
        val failures: MutableList<Exception> = mutableListOf(),
    ) {
        val personalizer = LivePersonalizer.open(store, model, vocabulary.appsSha256)
        val announced = mutableListOf<ShadowPreloadDecision>()
        val session = TraceSession(
            PersonalizedTraceStore(store, personalizer, shadowPolicy = policy, onShadowDecisions = { announced += it }, onShadowFailure = { failures += it }),
            source, pipeline(), { now }, 2_000,
        )

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

    /** Every app eligible: makes PRELOAD decisions frequent enough to exercise the plumbing. */
    private val permissive = CandidateResolver { CandidateResolution(Eligibility.ELIGIBLE, "pkg.$it") }

    private fun realResolver() = MappingCandidateResolver(TraceTestSupport.mapping, vocabulary.appCount, { TraceTestSupport.env }, hasLauncherEntry)

    /** 25 launches over the four supported test apps, one per second from 1 s, never repeating the previous app. */
    private fun launches(count: Int): List<RawUsageEvent> {
        val apps = listOf("com.whatsapp", "com.instagram.android", "com.google.android.youtube", "com.android.settings")
        var i = 0
        return (0 until count).map { k ->
            i = (i + 1 + (k * 7 + 3) % 3) % apps.size
            resumed(1_000L + 1_000L * k, apps[i])
        }
    }

    private fun run(policy: ShadowPreloadPolicy?, polls: List<Long>, timeline: List<RawUsageEvent> = launches(25)): Process {
        val p = Process(InMemoryLayer2Store(), FakeEventSource(timeline), policy)
        p.open(500)
        for (t in polls) p.poll(t)
        return p
    }

    private val polls = listOf(4_000L, 4_500L, 9_000L, 10_000L, 20_000L, 30_000L)

    /** 11: the policy changes nothing about Layer 2 prediction, update order or persistence. */
    @Test
    fun layer2IsIdenticalWithAndWithoutTheShadowPolicy() {
        val off = run(null, polls)
        val configs = listOf(
            ShadowPolicyConfig.DEFAULT,
            ShadowPolicyConfig(maxCandidates = 3, maxRank = 5, minProbability = 0.0),
            ShadowPolicyConfig(enabled = false),
        )
        val variants = configs.map { run(ShadowPreloadPolicy(it, permissive), polls) } +
            run(ShadowPreloadPolicy(ShadowPolicyConfig.DEFAULT, realResolver()), polls) +
            run(ShadowPreloadPolicy(ShadowPolicyConfig.DEFAULT, CandidateResolver { error("resolver failure") }), polls)
        for (on in variants) {
            assertArrayEquals(off.store.stateBytes, on.store.stateBytes)
            assertEquals(off.store.log, on.store.log)
            assertEquals(off.store.trace.records, on.store.trace.records)
            assertEquals(off.store.trace.loadState(), on.store.trace.loadState())
        }
        assertTrue(off.store.shadowDecisions().isEmpty())
        // A failing policy drops its decisions, reports the failure, and Layer 2 still advanced.
        val failing = variants.last()
        assertTrue(failing.store.shadowDecisions().isEmpty())
        assertEquals(polls.size, failing.failures.size) // every poll here commits at least one launch
        assertEquals(24L, failing.personalizer.updateCount)
    }

    /** 12: decisions are made for each prediction, persisted with it, never duplicated, and resolved by the next launch. */
    @Test
    fun decisionsArePersistedWithTheirPredictionAndResolvedByTheNextLaunch() {
        val config = ShadowPolicyConfig(maxCandidates = 1, maxRank = 2, minProbability = 0.0)
        val p = Process(InMemoryLayer2Store(), FakeEventSource(launches(25)), ShadowPreloadPolicy(config, permissive))
        p.open(500)
        val pendingByPosition = HashMap<Int, PendingPrediction>()
        for (t in polls) {
            p.poll(t)
            p.personalizer.pendingPrediction?.let { pendingByPosition[it.position] = it }
        }
        val seq = sequenceOf(p.store.trace.records.map { it.record })
        val rows = p.store.shadowDecisions()

        assertEquals((0 until 25).flatMap { listOf(it to 1, it to 2) }, rows.map { it.decision.predictionPosition to it.decision.rank })
        assertEquals(rows.map { it.decision }, p.announced.sortedWith(compareBy({ it.predictionPosition }, { it.rank })))
        for (r in rows) {
            val d = r.decision
            assertEquals(seq[d.predictionPosition], d.currentAppId)          // decided after this launch
            assertEquals(seq.getOrNull(d.predictionPosition + 1), r.actualAppId) // resolved by the next launch only
            assertTrue(polls.contains(r.decidedAtMs))
            assertEquals(config, d.config)
        }
        assertNull(rows.last().actualAppId) // the pending prediction is not resolved yet
        // Each decision is made on the prediction as it was made (before its update): checked against
        // the pending predictions persisted at the end of each batch.
        for ((pos, pending) in pendingByPosition) {
            val probabilities = Layer2Prediction(pending.finalLogits, pos.toLong()).probabilities
            for (r in rows.filter { it.decision.predictionPosition == pos }) assertEquals(probabilities[r.decision.appId - 1], r.decision.probability, 0.0)
        }
        // One PRELOAD per prediction: rank 1, or rank 2 when rank 1 is the current app.
        for ((_, pair) in rows.groupBy { it.decision.predictionPosition }) {
            val (r1, r2) = pair.map { it.decision }
            assertEquals(1, pair.count { it.decision.decision == ShadowDecision.PRELOAD })
            if (r1.decision == ShadowDecision.PRELOAD) {
                assertTrue(r2.reason == DecisionReason.CANDIDATE_LIMIT || r2.eligibility == Eligibility.CURRENT_APP)
            } else {
                assertEquals(Eligibility.CURRENT_APP, r1.eligibility)
                assertEquals(DecisionReason.SELECTED, r2.reason)
            }
        }
    }

    @Test
    fun aFailedCommitOrARestartNeverDuplicatesOrLosesADecision() {
        val timeline = launches(10)
        val policy = ShadowPreloadPolicy(ShadowPolicyConfig.DEFAULT, permissive)
        val store = InMemoryLayer2Store()
        val source = FakeEventSource(timeline)
        val first = Process(store, source, policy)
        first.open(500)
        first.poll(4_000) // [0.5 s, 2 s): the 1 s launch, position 0
        store.failNextCommit = true
        assertTrue(runCatching { first.poll(6_000) }.isFailure) // [2 s, 4 s): positions 1, 2, rolled back with their decisions
        assertEquals(listOf(0), store.shadowDecisions().map { it.decision.predictionPosition })
        first.poll(6_000)                                         // the retry decides them exactly once
        assertEquals(listOf(0, 1, 2), store.shadowDecisions().map { it.decision.predictionPosition })

        // A new process: the restored pending prediction (position 2) is not decided again.
        val second = Process(store, source, policy)
        second.open(8_000)                                        // 4..7 s fall in the gap
        second.poll(12_000)                                       // [8 s, 10 s): positions 3, 4
        val rows = store.shadowDecisions()
        assertEquals(listOf(0, 1, 2, 3, 4), rows.map { it.decision.predictionPosition })
        val seq = sequenceOf(store.trace.records.map { it.record })
        assertEquals(seq.drop(1) + listOf(null), rows.map { it.actualAppId })
        assertEquals(listOf(3, 4), second.announced.map { it.predictionPosition })
    }

    /** Zero side effects: no code path in the app can start, bind or preload another app. */
    @Test
    fun theShadowPolicyHasNoWayToLaunchAnything() {
        val launchApi = Regex(
            "startActivit|startService|startForegroundService|bindService|sendBroadcast|moveTaskToBack|" +
                "killBackgroundProcesses|ActivityManager|Runtime\\.getRuntime|ProcessBuilder|PendingIntent|\\.exec\\(",
        )
        val sources = File("src/main/java").walkTopDown().filter { it.isFile && it.extension == "kt" }.toList()
        assertTrue(sources.size > 20)
        val uses = sources.flatMap { f ->
            f.readLines().filterNot { it.trim().startsWith("//") }.mapNotNull { line -> launchApi.find(line)?.let { "${f.name}:${it.value}" } }
        }.toSortedSet()
        // The only ones are AdaPreload's own UI and service (system settings screens, its notification,
        // its own foreground service), all present before Phase E.
        assertEquals(
            sortedSetOf(
                "MainActivity.kt:startActivit",
                "AdaPreloadForegroundService.kt:PendingIntent",
                "AdaPreloadForegroundService.kt:startForegroundService",
            ),
            uses,
        )
        // The decision path itself has no Android dependency at all.
        val shadow = sources.filter { it.parentFile?.name == "shadow" || it.parentFile?.name == "live" }
        assertTrue(shadow.size >= 3)
        for (f in shadow) assertTrue(f.name, f.readLines().none { it.startsWith("import android.") })
    }
}
