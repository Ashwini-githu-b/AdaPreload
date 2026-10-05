package com.adapreload.instrumentation.model

import com.adapreload.instrumentation.trace.Json
import kotlin.math.abs

/**
 * Compares [Layer1Model] outputs with the golden vectors written by tools/export_layer1.py
 * (Phase C numerical parity). Shared by the JVM unit test and the on-device instrumented test.
 */
object Layer1Parity {

    /**
     * Tolerances (docs/LAYER1_ANDROID_INFERENCE.md).
     *
     * fp64: Kotlin double arithmetic vs the PyTorch reference run in float64 on the same weights.
     * The only difference is summation order, so errors are at the 1e-15 level and anything
     * larger means the computation itself differs.
     *
     * fp32: Kotlin vs the PyTorch float32 reference, which is how the frozen outputs were made.
     * Across all 30,040 held-out events the float32 reference itself differs from float64 by at
     * most 8.6e-7 (probabilities) and 6.4e-6 (logits); the bounds below are about 6-8x that.
     *
     * Colab: Kotlin vs the float32 top-20 Layer 1 probabilities stored in the frozen Layer 2
     * artifact (computed in Colab); a local float32 reproduction matches them within 1.0e-6.
     */
    const val FP64_LOGIT_TOLERANCE = 1e-12
    const val FP64_PROBABILITY_TOLERANCE = 1e-13
    const val FP32_LOGIT_TOLERANCE = 5e-5
    const val FP32_PROBABILITY_TOLERANCE = 5e-6
    const val COLAB_PROBABILITY_TOLERANCE = 5e-6
    const val PROBABILITY_SUM_TOLERANCE = 1e-12

    data class CaseResult(
        val name: String,
        val contextLength: Int,
        val outputSize: Int,
        val probabilitySum: Double,
        val maxAbsErrorFp32: Double,
        val meanAbsErrorFp32: Double,
        val maxAbsLogitErrorFp32: Double,
        val maxAbsErrorFp64: Double,
        val meanAbsErrorFp64: Double,
        val maxAbsLogitErrorFp64: Double,
        val top1Match: Boolean,
        val top5Match: Boolean,
        /** Max error at the Colab top-20 ids, for cases taken from the Layer 2 artifact. */
        val maxAbsErrorColab: Double?,
    )

    data class SampleResult(
        val events: Int,
        val maxAbsError: Double,
        val meanAbsError: Double,
        val top1Matches: Int,
        val top5Matches: Int,
        val top20Matches: Int,
    )

    class GoldenSet(val checkpointSha256: String, val weightsSha256: String, val cases: List<Map<*, *>>)

    fun parseGolden(json: String): GoldenSet {
        val root = Json.parse(json) as Map<*, *>
        require(root["format"] == "adapreload-layer1-golden") { "Not a Layer 1 golden file" }
        val source = root["source"] as Map<*, *>
        return GoldenSet(source["checkpoint_sha256"] as String, source["weights_sha256"] as String, root["cases"] as List<Map<*, *>>)
    }

    fun evaluateGolden(model: Layer1Model, golden: GoldenSet): List<CaseResult> {
        require(golden.weightsSha256 == model.weightsSha256) { "Golden vectors were generated for different weights" }
        return golden.cases.map { case ->
            val prediction = model.predict(ints(case["context"]))
            val p = prediction.probabilities
            val p32 = doubles(case["expected_probabilities_fp32"])
            val p64 = doubles(case["expected_probabilities_fp64"])
            val ranked = prediction.rankedAppIds()
            val colabIds = case["colab_top20_l1_ids"]?.let(::ints)
            CaseResult(
                name = case["name"] as String,
                contextLength = (case["context_length"] as Long).toInt(),
                outputSize = p.size,
                probabilitySum = p.sum(),
                maxAbsErrorFp32 = maxAbs(p, p32),
                meanAbsErrorFp32 = meanAbs(p, p32),
                maxAbsLogitErrorFp32 = maxAbs(prediction.logits, doubles(case["expected_logits_fp32"])),
                maxAbsErrorFp64 = maxAbs(p, p64),
                meanAbsErrorFp64 = meanAbs(p, p64),
                maxAbsLogitErrorFp64 = maxAbs(prediction.logits, doubles(case["expected_logits_fp64"])),
                top1Match = ranked[0] == ints(case["expected_top5_ids"])[0],
                top5Match = ranked.take(5) == ints(case["expected_top5_ids"]).toList(),
                maxAbsErrorColab = colabIds?.let { ids ->
                    val colab = doubles(case["colab_top20_l1_probs"])
                    ids.indices.maxOf { abs(prediction.probabilityOf(ids[it]) - colab[it]) }
                },
            )
        }
    }

    /** Compares with the frozen Colab top-20 Layer 1 outputs for sampled held-out events. */
    fun evaluateColabSample(model: Layer1Model, json: String): SampleResult {
        val root = Json.parse(json) as Map<*, *>
        require(root["format"] == "adapreload-layer1-colab-sample") { "Not a Layer 1 Colab sample file" }
        val events = root["events"] as List<*>
        var max = 0.0
        var sum = 0.0
        var count = 0
        var top1 = 0
        var top5 = 0
        var top20 = 0
        for (e in events) {
            e as Map<*, *>
            val prediction = model.predict(ints(e["context"]))
            val ids = ints(e["top20_ids"])
            val probs = doubles(e["top20_probs"])
            for (k in ids.indices) {
                val err = abs(prediction.probabilityOf(ids[k]) - probs[k])
                max = maxOf(max, err)
                sum += err
                count++
            }
            val ranked = prediction.rankedAppIds()
            if (ranked[0] == ids[0]) top1++
            if (ranked.take(5) == ids.take(5)) top5++
            if (ranked.take(20) == ids.toList()) top20++
        }
        return SampleResult(events.size, max, sum / count, top1, top5, top20)
    }

    fun report(cases: List<CaseResult>, sample: SampleResult?): String = buildString {
        appendLine("Layer 1 parity: ${cases.size} golden cases")
        appendLine("  output shape ${cases.map { it.outputSize }.distinct()} probabilities (app ids 1..87)")
        appendLine("  max |sum(p) - 1|                 ${fmt(cases.maxOf { abs(it.probabilitySum - 1.0) })}")
        appendLine("  vs PyTorch fp32: max abs error   ${fmt(cases.maxOf { it.maxAbsErrorFp32 })}, mean abs error ${fmt(cases.map { it.meanAbsErrorFp32 }.average())}, max logit error ${fmt(cases.maxOf { it.maxAbsLogitErrorFp32 })}")
        appendLine("  vs PyTorch fp64: max abs error   ${fmt(cases.maxOf { it.maxAbsErrorFp64 })}, mean abs error ${fmt(cases.map { it.meanAbsErrorFp64 }.average())}, max logit error ${fmt(cases.maxOf { it.maxAbsLogitErrorFp64 })}")
        cases.mapNotNull { it.maxAbsErrorColab }.takeIf { it.isNotEmpty() }?.let {
            appendLine("  vs Colab top-20 (artifact cases): max abs error ${fmt(it.max())}")
        }
        appendLine("  top-1 match ${cases.count { it.top1Match }}/${cases.size}, top-5 match (ordered) ${cases.count { it.top5Match }}/${cases.size}")
        sample?.let {
            appendLine("Frozen Colab sample: ${it.events} events")
            appendLine("  max abs error ${fmt(it.maxAbsError)}, mean abs error ${fmt(it.meanAbsError)} (at Colab top-20 ids)")
            appendLine("  top-1 match ${it.top1Matches}/${it.events}, top-5 (ordered) ${it.top5Matches}/${it.events}, top-20 (ordered) ${it.top20Matches}/${it.events}")
        }
    }

    private fun fmt(x: Double) = String.format(java.util.Locale.ROOT, "%.3e", x)

    private fun ints(v: Any?): IntArray = (v as List<*>).map { (it as Long).toInt() }.toIntArray()

    private fun doubles(v: Any?): DoubleArray = (v as List<*>).map { (it as Number).toDouble() }.toDoubleArray()

    private fun maxAbs(a: DoubleArray, b: DoubleArray): Double {
        require(a.size == b.size) { "Size mismatch: ${a.size} vs ${b.size}" }
        return a.indices.maxOf { abs(a[it] - b[it]) }
    }

    private fun meanAbs(a: DoubleArray, b: DoubleArray): Double = a.indices.sumOf { abs(a[it] - b[it]) } / a.size
}
