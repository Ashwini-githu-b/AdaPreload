package com.adapreload.instrumentation.shadow

import com.adapreload.instrumentation.trace.EnvironmentSnapshot
import com.adapreload.instrumentation.trace.LaunchClassifier
import com.adapreload.instrumentation.trace.MappingStatus
import com.adapreload.instrumentation.trace.PackageMapping
import com.adapreload.instrumentation.trace.TraceTestSupport
import com.adapreload.instrumentation.trace.TraceTestSupport.SELF
import com.adapreload.instrumentation.trace.TraceTestSupport.id
import com.adapreload.instrumentation.trace.TraceTestSupport.vocabulary
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Phase E1: the shadow preload policy decides; it never preloads. */
class ShadowPreloadPolicyTest {

    private val position = 7
    private val current = 50

    /** Probabilities for app ids 1..87: the given apps get the given values, every other app 0.001. */
    private fun probs(vararg apps: Pair<Int, Double>): DoubleArray =
        DoubleArray(87) { 0.001 }.also { p -> for ((app, value) in apps) p[app - 1] = value }

    private fun input(vararg apps: Pair<Int, Double>) = ShadowPolicyInput(position, current, probs(*apps))

    /** Every app eligible, with package "pkg.<id>", except the listed exclusions. */
    private fun resolver(excluded: Map<Int, Eligibility> = emptyMap()) = CandidateResolver { appId ->
        CandidateResolution(excluded[appId] ?: Eligibility.ELIGIBLE, "pkg.$appId")
    }

    private fun policy(config: ShadowPolicyConfig = ShadowPolicyConfig.DEFAULT, resolver: CandidateResolver = resolver()) =
        ShadowPreloadPolicy(config, resolver)

    // Real assets: the shipped vocabulary and mapping table, as TraceRuntime loads them.
    private val realMapping: PackageMapping by lazy {
        PackageMapping.parse(TraceTestSupport.asset("package_mapping.tsv").readBytes(), vocabulary, LaunchClassifier.SYSTEM_EXCLUSIONS + SELF)
    }
    private val env = EnvironmentSnapshot(SELF, "com.example.launcher", setOf("com.example.keyboard"))

    private fun realResolver(installed: (String) -> Boolean = { true }, environment: EnvironmentSnapshot = env) =
        MappingCandidateResolver(realMapping, vocabulary.appCount, { environment }, installed)

    @Test
    fun defaultsAreOneConservativeCandidate() {
        val c = ShadowPolicyConfig.DEFAULT
        assertEquals(ShadowPolicyConfig(enabled = true, maxCandidates = 1, maxRank = 1, minProbability = 0.5, excludeCurrentApp = true), c)
        assertEquals("e1-shadow-1", ShadowPolicyConfig.VERSION)
        assertTrue(runCatching { ShadowPolicyConfig(maxCandidates = 0) }.isFailure)
        assertTrue(runCatching { ShadowPolicyConfig(maxRank = 0) }.isFailure)
        assertTrue(runCatching { ShadowPolicyConfig(minProbability = 1.5) }.isFailure)
    }

    /** 1 */
    @Test
    fun disabledPolicySkipsEverything() {
        val d = policy(ShadowPolicyConfig(enabled = false)).decide(input(10 to 0.9)).single()
        assertEquals(ShadowDecision.SKIP, d.decision)
        assertEquals(DecisionReason.POLICY_DISABLED, d.reason)
        assertEquals(10, d.appId)
        val many = policy(ShadowPolicyConfig(enabled = false, maxCandidates = 3, maxRank = 3)).decide(input(10 to 0.9, 11 to 0.8, 12 to 0.7))
        assertTrue(many.all { it.decision == ShadowDecision.SKIP && it.reason == DecisionReason.POLICY_DISABLED })
    }

    /** 2 */
    @Test
    fun rankOneAboveThresholdIsPreload() {
        val d = policy().decide(input(10 to 0.62, 11 to 0.2)).single()
        assertEquals(ShadowPreloadDecision(position, current, 1, 10, "pkg.10", 0.62, Eligibility.ELIGIBLE, ShadowDecision.PRELOAD, DecisionReason.SELECTED, ShadowPolicyConfig.DEFAULT), d)
        assertEquals("selected", d.reasonLabel)
    }

    /** 3 */
    @Test
    fun probabilityBelowThresholdIsSkipped() {
        val below = policy().decide(input(10 to 0.49)).single()
        assertEquals(ShadowDecision.SKIP, below.decision)
        assertEquals(DecisionReason.BELOW_THRESHOLD, below.reason)
        assertEquals("below_threshold", below.reasonLabel)
        // The threshold is inclusive.
        assertEquals(ShadowDecision.PRELOAD, policy().decide(input(10 to 0.5)).single().decision)
    }

    /** 4 */
    @Test
    fun rankOutsideTheAllowedRangeIsSkipped() {
        // Two candidates allowed, but only rank 1 is in range.
        val d = policy(ShadowPolicyConfig(maxCandidates = 2, maxRank = 1)).decide(input(10 to 0.6, 11 to 0.55))
        assertEquals(listOf(ShadowDecision.PRELOAD, ShadowDecision.SKIP), d.map { it.decision })
        assertEquals(DecisionReason.RANK_OUT_OF_RANGE, d[1].reason)
        assertEquals(11, d[1].appId)
        // Defaults: an excluded top-1 is not replaced by the rank-2 app, however likely.
        val top1Excluded = policy(resolver = resolver(mapOf(10 to Eligibility.NO_MAPPED_PACKAGE))).decide(input(10 to 0.6, 11 to 0.55))
        assertEquals(1, top1Excluded.size)
        assertTrue(top1Excluded.none { it.decision == ShadowDecision.PRELOAD })
    }

    /** 5 */
    @Test
    fun candidateCountIsRespected() {
        val one = policy(ShadowPolicyConfig(maxRank = 3)).decide(input(10 to 0.7, 11 to 0.6, 12 to 0.55))
        assertEquals(listOf(DecisionReason.SELECTED, DecisionReason.CANDIDATE_LIMIT, DecisionReason.CANDIDATE_LIMIT), one.map { it.reason })
        val two = policy(ShadowPolicyConfig(maxCandidates = 2, maxRank = 3)).decide(input(10 to 0.7, 11 to 0.6, 12 to 0.55))
        assertEquals(listOf(10, 11), two.filter { it.decision == ShadowDecision.PRELOAD }.map { it.appId })
        // An excluded higher-ranked app does not use up a candidate slot.
        val skipFirst = policy(ShadowPolicyConfig(maxRank = 3), resolver(mapOf(10 to Eligibility.NOT_INSTALLED)))
            .decide(input(10 to 0.7, 11 to 0.6, 12 to 0.55))
        assertEquals(listOf(DecisionReason.EXCLUDED, DecisionReason.SELECTED, DecisionReason.CANDIDATE_LIMIT), skipFirst.map { it.reason })
        assertEquals(1, skipFirst.count { it.decision == ShadowDecision.PRELOAD })
    }

    /** 6 */
    @Test
    fun adaPreloadItselfIsExcluded() {
        // The mapping table can never map AdaPreload (it is a reserved package)...
        assertTrue(runCatching { TraceTestSupport.mappingOf(TraceTestSupport.row(SELF, "self", "WhatsApp Messenger")) }.isFailure)
        // ...and a candidate whose package is AdaPreload's is excluded by the Phase B classification.
        val wa = id("WhatsApp Messenger")
        val selfEnv = EnvironmentSnapshot("com.whatsapp", null, emptySet())
        val r = MappingCandidateResolver(TraceTestSupport.mapping, vocabulary.appCount, { selfEnv }, { true }).resolve(wa)
        assertEquals(Eligibility.EXCLUDED_SELF, r.eligibility)
        val d = ShadowPreloadPolicy(ShadowPolicyConfig.DEFAULT, MappingCandidateResolver(TraceTestSupport.mapping, vocabulary.appCount, { selfEnv }, { true }))
            .decide(input(wa to 0.9)).single()
        assertEquals(DecisionReason.EXCLUDED, d.reason)
        assertEquals("excluded:excluded_self", d.reasonLabel)
    }

    /** 7 */
    @Test
    fun homeAndInputMethodAreExcluded() {
        val wa = id("WhatsApp Messenger")
        val homeEnv = EnvironmentSnapshot(SELF, "com.whatsapp", emptySet())
        assertEquals(Eligibility.EXCLUDED_HOME, MappingCandidateResolver(TraceTestSupport.mapping, vocabulary.appCount, { homeEnv }, { true }).resolve(wa).eligibility)
        val imeEnv = EnvironmentSnapshot(SELF, null, setOf("com.whatsapp"))
        assertEquals(Eligibility.EXCLUDED_IME, MappingCandidateResolver(TraceTestSupport.mapping, vocabulary.appCount, { imeEnv }, { true }).resolve(wa).eligibility)
        val d = ShadowPreloadPolicy(ShadowPolicyConfig.DEFAULT, MappingCandidateResolver(TraceTestSupport.mapping, vocabulary.appCount, { homeEnv }, { true }))
            .decide(input(wa to 0.9)).single()
        assertEquals(ShadowDecision.SKIP, d.decision)
        assertTrue(d.excluded)
    }

    /** 8 */
    @Test
    fun unmappedInvalidOrMissingCandidatesAreExcluded() {
        val r = realResolver()
        assertEquals(Eligibility.INVALID_APP_ID, r.resolve(0).eligibility)   // padding is never a candidate
        assertEquals(Eligibility.INVALID_APP_ID, r.resolve(88).eligibility)
        assertEquals(Eligibility.NO_MAPPED_PACKAGE, r.resolve(id("Google Chrome")).eligibility) // its only package row is AMBIGUOUS
        assertEquals(Eligibility.NOT_INSTALLED, realResolver(installed = { false }).resolve(id("WhatsApp Messenger")).eligibility)
        val wa = realResolver().resolve(id("WhatsApp Messenger"))
        assertEquals(Eligibility.ELIGIBLE, wa.eligibility)
        assertEquals("com.whatsapp", wa.packageName)

        val chrome = ShadowPreloadPolicy(ShadowPolicyConfig.DEFAULT, r).decide(input(id("Google Chrome") to 0.95)).single()
        assertEquals(ShadowDecision.SKIP, chrome.decision)
        assertEquals("excluded:no_mapped_package", chrome.reasonLabel)
        assertNull(chrome.packageName)
        // An invalid id never becomes a decision, whatever the probability array says.
        val invalid = ShadowPreloadPolicy(ShadowPolicyConfig.DEFAULT, CandidateResolver { CandidateResolution(Eligibility.INVALID_APP_ID, null) })
            .decide(input(10 to 0.9)).single()
        assertEquals(Eligibility.INVALID_APP_ID, invalid.eligibility)
        assertEquals(ShadowDecision.SKIP, invalid.decision)
    }

    /** 9 */
    @Test
    fun ambiguousCandidatesAreExcluded() {
        // No AMBIGUOUS or UNSUPPORTED row can ever be chosen: only MAPPED packages resolve.
        val r = realResolver()
        val statusOf = realMapping.entries.associate { it.packageName to it.status }
        for (app in 1..vocabulary.appCount) {
            val res = r.resolve(app)
            if (res.eligibility == Eligibility.ELIGIBLE) assertEquals(MappingStatus.MAPPED, statusOf[res.packageName])
        }
        val ambiguousPackages = realMapping.entries.filter { it.status == MappingStatus.AMBIGUOUS }.map { it.packageName }.toSet()
        assertTrue(ambiguousPackages.isNotEmpty())
        assertTrue((1..vocabulary.appCount).none { r.resolve(it).packageName in ambiguousPackages })

        // An app id with two installed MAPPED packages is an ambiguous target.
        val amazon = id("Amazon Shopping")
        assertEquals(Eligibility.MULTIPLE_PACKAGES, r.resolve(amazon).eligibility)
        val onlyIndia = realResolver(installed = { it == "in.amazon.mShop.android.shopping" }).resolve(amazon)
        assertEquals(Eligibility.ELIGIBLE, onlyIndia.eligibility)
        assertEquals("in.amazon.mShop.android.shopping", onlyIndia.packageName)
        val d = ShadowPreloadPolicy(ShadowPolicyConfig.DEFAULT, r).decide(input(amazon to 0.9)).single()
        assertEquals("excluded:multiple_packages", d.reasonLabel)
    }

    /** 10 */
    @Test
    fun outputIsDeterministic() {
        val config = ShadowPolicyConfig(maxCandidates = 2, maxRank = 4, minProbability = 0.1)
        val p = probs(10 to 0.3, 11 to 0.3, 12 to 0.2, 13 to 0.15)
        val a = policy(config).decide(ShadowPolicyInput(position, current, p))
        val b = policy(config).decide(ShadowPolicyInput(position, current, p.copyOf()))
        assertEquals(a, b)
        // Equal probabilities rank the lower app id first, as Layer 2 rankings do.
        assertEquals(listOf(10, 11, 12, 13), a.map { it.appId })
        val realA = ShadowPreloadPolicy(ShadowPolicyConfig.DEFAULT, realResolver()).decide(input(id("YouTube") to 0.7))
        val realB = ShadowPreloadPolicy(ShadowPolicyConfig.DEFAULT, realResolver()).decide(input(id("YouTube") to 0.7))
        assertEquals(realA, realB)
    }

    @Test
    fun theCurrentAppIsNeverChosen() {
        val d = policy().decide(ShadowPolicyInput(position, 10, probs(10 to 0.8))).single()
        assertEquals(Eligibility.CURRENT_APP, d.eligibility)
        assertEquals(ShadowDecision.SKIP, d.decision)
        val off = policy(ShadowPolicyConfig(excludeCurrentApp = false)).decide(ShadowPolicyInput(position, 10, probs(10 to 0.8))).single()
        assertEquals(ShadowDecision.PRELOAD, off.decision)
    }
}
