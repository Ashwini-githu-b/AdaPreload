package com.adapreload.instrumentation

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.adapreload.instrumentation.model.Layer1Assets
import com.adapreload.instrumentation.model.Layer1Parity
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.abs

/**
 * Phase C on a real device: the model is loaded from the app's own assets with the Android
 * loader and compared with the golden vectors, using the same tolerances as the JVM test.
 */
@RunWith(AndroidJUnit4::class)
class Layer1ParityInstrumentedTest {

    @Test
    fun layer1MatchesGoldenVectorsOnDevice() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val model = Layer1Assets.load(instrumentation.targetContext)
        fun testAsset(name: String) = instrumentation.context.assets.open(name).use { String(it.readBytes()) }

        val results = Layer1Parity.evaluateGolden(model, Layer1Parity.parseGolden(testAsset("layer1_golden.json")))
        val sample = Layer1Parity.evaluateColabSample(model, testAsset("layer1_artifact_sample.json"))
        Log.i("Layer1Parity", Layer1Parity.report(results, sample))

        for (r in results) {
            assertEquals(r.name, 87, r.outputSize)
            assertTrue(r.name, abs(r.probabilitySum - 1.0) <= Layer1Parity.PROBABILITY_SUM_TOLERANCE)
            assertTrue(r.name, r.maxAbsErrorFp64 <= Layer1Parity.FP64_PROBABILITY_TOLERANCE)
            assertTrue(r.name, r.maxAbsLogitErrorFp64 <= Layer1Parity.FP64_LOGIT_TOLERANCE)
            assertTrue(r.name, r.maxAbsErrorFp32 <= Layer1Parity.FP32_PROBABILITY_TOLERANCE)
            assertTrue(r.name, r.maxAbsLogitErrorFp32 <= Layer1Parity.FP32_LOGIT_TOLERANCE)
            assertTrue(r.name, r.top1Match && r.top5Match)
        }
        assertTrue(sample.maxAbsError <= Layer1Parity.COLAB_PROBABILITY_TOLERANCE)
        assertEquals(sample.events, sample.top1Matches)
        assertEquals(sample.events, sample.top5Matches)
    }
}
