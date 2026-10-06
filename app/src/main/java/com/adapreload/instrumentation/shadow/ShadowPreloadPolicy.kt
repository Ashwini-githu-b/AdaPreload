package com.adapreload.instrumentation.shadow

import com.adapreload.instrumentation.trace.Classification
import com.adapreload.instrumentation.trace.EnvironmentSnapshot
import com.adapreload.instrumentation.trace.LaunchClassifier
import com.adapreload.instrumentation.trace.MappingStatus
import com.adapreload.instrumentation.trace.PackageMapping
import java.util.Locale

/**
 * Phase E1: the shadow preload policy. It turns one Layer 2 prediction into preload *decisions*
 * and nothing else. A PRELOAD decision means "AdaPreload would preload this app now"; it is
 * recorded, never executed. No process, activity or service of another app is ever started.
 *
 * Prediction (Layer 2), preload decision (this policy), preload execution (E2, not implemented)
 * and the user's actual next launch are four separate things.
 */
data class ShadowPolicyConfig(
    /** When false every candidate is SKIP (POLICY_DISABLED). */
    val enabled: Boolean = true,
    /** At most this many PRELOAD decisions per prediction. */
    val maxCandidates: Int = 1,
    /** Only apps ranked 1..maxRank by Layer 2 can be chosen. */
    val maxRank: Int = 1,
    /** Minimum Layer 2 probability (padding-masked softmax, T13). An engineering starting point, not a tuned value. */
    val minProbability: Double = 0.5,
    /** Never choose the app the prediction was made after: by T3 it cannot be the next canonical launch. */
    val excludeCurrentApp: Boolean = true,
) {
    init {
        require(maxCandidates >= 1) { "maxCandidates must be at least 1" }
        require(maxRank >= 1) { "maxRank must be at least 1" }
        require(minProbability in 0.0..1.0) { "minProbability must be in [0, 1]" }
    }

    companion object {
        const val VERSION = "e1-shadow-1"
        val DEFAULT = ShadowPolicyConfig()
    }
}

enum class ShadowDecision { PRELOAD, SKIP }

/** Whether a candidate could be preloaded at all, independent of its probability. */
enum class Eligibility(val excluded: Boolean) {
    ELIGIBLE(false),

    /** Not an app id of the vocabulary (1..87). */
    INVALID_APP_ID(true),

    /** The app the prediction was made after (see [ShadowPolicyConfig.excludeCurrentApp]). */
    CURRENT_APP(true),

    /** No MAPPED row in the mapping table for this app id (AMBIGUOUS and UNSUPPORTED rows carry no app id). */
    NO_MAPPED_PACKAGE(true),

    /** Mapped, but no mapped package has a launcher entry on this device. */
    NOT_INSTALLED(true),

    /** More than one installed package maps to this app id, so the target package is ambiguous. */
    MULTIPLE_PACKAGES(true),

    /** The package is excluded by the Phase B classification (A4). */
    EXCLUDED_SELF(true),
    EXCLUDED_HOME(true),
    EXCLUDED_IME(true),
    EXCLUDED_SYSTEM(true),

    /** The Phase B classification of the package is not SUPPORTED for any other reason. */
    NOT_SUPPORTED(true),
}

enum class DecisionReason {
    /** PRELOAD: eligible, ranked within maxRank, at or above the threshold, within maxCandidates. */
    SELECTED,
    POLICY_DISABLED,
    RANK_OUT_OF_RANGE,
    EXCLUDED,
    BELOW_THRESHOLD,
    CANDIDATE_LIMIT,
}

class CandidateResolution(val eligibility: Eligibility, val packageName: String?)

/** Maps a predicted app id to the package that would be preloaded, using the existing mapping. */
fun interface CandidateResolver {
    fun resolve(appId: Int): CandidateResolution
}

/** One Layer 2 prediction as the policy sees it. */
class ShadowPolicyInput(
    /** Sequence position of the launch the prediction was made after. */
    val predictionPosition: Int,
    /** The app of that launch (the last context app). */
    val currentAppId: Int,
    /** Padding-masked Layer 2 probabilities: probabilities[k] is app id k + 1 (T13). */
    val probabilities: DoubleArray,
)

/** The policy's verdict on one ranked candidate of one prediction. */
data class ShadowPreloadDecision(
    val predictionPosition: Int,
    val currentAppId: Int,
    /** Layer 2 rank (1 = top-1); ties rank the lower id first. */
    val rank: Int,
    val appId: Int,
    /** The package that would be preloaded, when one could be resolved. */
    val packageName: String?,
    val probability: Double,
    val eligibility: Eligibility,
    val decision: ShadowDecision,
    val reason: DecisionReason,
    val config: ShadowPolicyConfig,
) {
    val excluded: Boolean get() = eligibility.excluded

    /** The reason in log form: "selected", "below_threshold", "excluded:no_mapped_package", ... */
    val reasonLabel: String
        get() = reason.name.lowercase(Locale.ROOT) + if (reason == DecisionReason.EXCLUDED) ":" + eligibility.name.lowercase(Locale.ROOT) else ""
}

/**
 * Examines the Layer 2 top max(maxRank, maxCandidates) apps in rank order and decides each one:
 * disabled → POLICY_DISABLED; rank > maxRank → RANK_OUT_OF_RANGE; not eligible → EXCLUDED;
 * probability < minProbability → BELOW_THRESHOLD; maxCandidates already chosen → CANDIDATE_LIMIT;
 * otherwise PRELOAD. Deterministic: the same input, configuration and resolver give the same output.
 */
class ShadowPreloadPolicy(val config: ShadowPolicyConfig, private val resolver: CandidateResolver) {

    fun decide(input: ShadowPolicyInput): List<ShadowPreloadDecision> {
        val p = input.probabilities
        require(p.isNotEmpty() && p.all { it.isFinite() }) { "Probabilities must be finite" }
        val ranked = p.indices.sortedWith(compareByDescending<Int> { p[it] }.thenBy { it }).map { it + 1 }
        val examined = minOf(maxOf(config.maxRank, config.maxCandidates), ranked.size)
        var selected = 0
        return (1..examined).map { rank ->
            val appId = ranked[rank - 1]
            val probability = p[appId - 1]
            val resolution = resolver.resolve(appId)
            val eligibility = when {
                resolution.eligibility == Eligibility.INVALID_APP_ID -> Eligibility.INVALID_APP_ID
                config.excludeCurrentApp && appId == input.currentAppId -> Eligibility.CURRENT_APP
                else -> resolution.eligibility
            }
            val reason = when {
                !config.enabled -> DecisionReason.POLICY_DISABLED
                rank > config.maxRank -> DecisionReason.RANK_OUT_OF_RANGE
                eligibility.excluded -> DecisionReason.EXCLUDED
                probability < config.minProbability -> DecisionReason.BELOW_THRESHOLD
                selected >= config.maxCandidates -> DecisionReason.CANDIDATE_LIMIT
                else -> DecisionReason.SELECTED.also { selected++ }
            }
            ShadowPreloadDecision(
                input.predictionPosition, input.currentAppId, rank, appId, resolution.packageName, probability,
                eligibility, if (reason == DecisionReason.SELECTED) ShadowDecision.PRELOAD else ShadowDecision.SKIP, reason, config,
            )
        }
    }
}

/**
 * Resolves app ids through the existing mapping table and the Phase B launch classifier, so the
 * policy never has a package vocabulary of its own:
 * 1. the app id must be in 1..[appCount];
 * 2. its packages are the MAPPED rows with that id (AMBIGUOUS and UNSUPPORTED rows have no id);
 * 3. exactly one of them must have a launcher entry on this device;
 * 4. that package must classify as SUPPORTED in the current environment (A4: not AdaPreload,
 *    the default home app, an input method or the explicit system list).
 * The checks only query package metadata; they never start anything.
 */
class MappingCandidateResolver(
    mapping: PackageMapping,
    private val appCount: Int,
    private val environment: () -> EnvironmentSnapshot,
    private val hasLauncherEntry: (String) -> Boolean,
) : CandidateResolver {
    private val classifier = LaunchClassifier(mapping)
    private val packagesById: Map<Int, List<String>> = mapping.entries
        .filter { it.status == MappingStatus.MAPPED }
        .groupBy({ it.lsappId!! }, { it.packageName })

    override fun resolve(appId: Int): CandidateResolution {
        if (appId !in 1..appCount) return CandidateResolution(Eligibility.INVALID_APP_ID, null)
        val packages = packagesById[appId].orEmpty()
        if (packages.isEmpty()) return CandidateResolution(Eligibility.NO_MAPPED_PACKAGE, null)
        val installed = packages.filter(hasLauncherEntry)
        if (installed.isEmpty()) return CandidateResolution(Eligibility.NOT_INSTALLED, packages.singleOrNull())
        if (installed.size > 1) return CandidateResolution(Eligibility.MULTIPLE_PACKAGES, null)
        val pkg = installed.single()
        val eligibility = when (classifier.classify(pkg, environment(), hasLauncherEntry).classification) {
            Classification.SUPPORTED -> Eligibility.ELIGIBLE
            Classification.SELF -> Eligibility.EXCLUDED_SELF
            Classification.EXCLUDED_HOME -> Eligibility.EXCLUDED_HOME
            Classification.EXCLUDED_IME -> Eligibility.EXCLUDED_IME
            Classification.EXCLUDED_SYSTEM -> Eligibility.EXCLUDED_SYSTEM
            else -> Eligibility.NOT_SUPPORTED
        }
        return CandidateResolution(eligibility, pkg)
    }
}

/** A persisted decision: when it was made and, once the next launch revealed it, which app was actually launched. */
data class ShadowDecisionRecord(
    val decision: ShadowPreloadDecision,
    /** Wall-clock time of the poll that made the prediction and the decision. */
    val decidedAtMs: Long,
    /** The next canonical launch (the prediction's target); null until it is observed. */
    val actualAppId: Int?,
)
