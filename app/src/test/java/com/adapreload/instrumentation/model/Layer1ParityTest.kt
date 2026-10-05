package com.adapreload.instrumentation.model

import com.adapreload.instrumentation.model.Layer1Parity.COLAB_PROBABILITY_TOLERANCE
import com.adapreload.instrumentation.model.Layer1Parity.FP32_LOGIT_TOLERANCE
import com.adapreload.instrumentation.model.Layer1Parity.FP32_PROBABILITY_TOLERANCE
import com.adapreload.instrumentation.model.Layer1Parity.FP64_LOGIT_TOLERANCE
import com.adapreload.instrumentation.model.Layer1Parity.FP64_PROBABILITY_TOLERANCE
import com.adapreload.instrumentation.model.Layer1Parity.PROBABILITY_SUM_TOLERANCE
import com.adapreload.instrumentation.trace.TraceTestSupport
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.math.abs

/** Phase C: the Kotlin Layer 1 matches the PyTorch reference and the frozen Colab outputs. */
class Layer1ParityTest {

    private val model: Layer1Model by lazy {
        Layer1Model.load(
            File("src/main/assets/${Layer1Assets.MANIFEST}").readText(),
            File("src/main/assets/${Layer1Assets.WEIGHTS}").readBytes(),
        )
    }

    private fun testAsset(name: String) = File("src/androidTest/assets/$name").readText()

    @Test
    fun weightsComeFromTheAuditedCheckpointAndVocabulary() {
        assertEquals(Layer1Config(88, 0, 20, 64, 2, 4, 16, 128, 1e-5), model.config)
        assertEquals(TraceTestSupport.vocabulary.appsSha256, model.vocabularyAppsSha256)
    }

    @Test
    fun goldenVectorsMatchThePyTorchReference() {
        val results = Layer1Parity.evaluateGolden(model, Layer1Parity.parseGolden(testAsset("layer1_golden.json")))
        val sample = Layer1Parity.evaluateColabSample(model, testAsset("layer1_artifact_sample.json"))
        println(Layer1Parity.report(results, sample))

        assertTrue(results.size >= 10)
        assertTrue("padded contexts covered", results.any { it.contextLength < 20 } && results.any { it.contextLength == 20 })
        for (r in results) {
            val where = "case ${r.name}"
            assertEquals(where, 87, r.outputSize)
            assertTrue(where, abs(r.probabilitySum - 1.0) <= PROBABILITY_SUM_TOLERANCE)
            assertTrue("$where fp64 probabilities ${r.maxAbsErrorFp64}", r.maxAbsErrorFp64 <= FP64_PROBABILITY_TOLERANCE)
            assertTrue("$where fp64 logits ${r.maxAbsLogitErrorFp64}", r.maxAbsLogitErrorFp64 <= FP64_LOGIT_TOLERANCE)
            assertTrue("$where fp32 probabilities ${r.maxAbsErrorFp32}", r.maxAbsErrorFp32 <= FP32_PROBABILITY_TOLERANCE)
            assertTrue("$where fp32 logits ${r.maxAbsLogitErrorFp32}", r.maxAbsLogitErrorFp32 <= FP32_LOGIT_TOLERANCE)
            r.maxAbsErrorColab?.let { assertTrue("$where Colab $it", it <= COLAB_PROBABILITY_TOLERANCE) }
            assertTrue("$where top-1", r.top1Match)
            assertTrue("$where top-5", r.top5Match)
        }
    }

    @Test
    fun frozenColabOutputsMatchForSampledHeldOutEvents() {
        val s = Layer1Parity.evaluateColabSample(model, testAsset("layer1_artifact_sample.json"))
        assertEquals(256, s.events)
        assertTrue("max abs error ${s.maxAbsError}", s.maxAbsError <= COLAB_PROBABILITY_TOLERANCE)
        assertEquals(s.events, s.top1Matches)
        assertEquals(s.events, s.top5Matches)
    }

    @Test
    fun paddingIsOnlyAllowedAsALeftPrefixAndIdsMustBeInRange() {
        fun rejects(context: IntArray) = runCatching { model.predict(context) }.isFailure
        assertTrue(rejects(IntArray(20)))                                   // no real app id
        assertTrue(rejects(IntArray(19) { 1 }))                             // wrong length
        assertTrue(rejects(IntArray(20) { if (it == 19) 0 else 5 }))        // padding after real ids
        assertTrue(rejects(IntArray(20) { if (it == 19) 88 else 0 }))       // id outside 1..87
        assertTrue(rejects(IntArray(20) { if (it == 19) -1 else 0 }))
    }

    @Test
    fun predictionIsDeterministic() {
        val context = IntArray(20) { if (it < 12) 0 else 80 - it }
        val a = model.predict(context)
        val b = model.predict(context)
        assertArrayEquals(a.logits, b.logits, 0.0)
        assertArrayEquals(a.probabilities, b.probabilities, 0.0)
    }
}
