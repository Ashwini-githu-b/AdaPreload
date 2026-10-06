package com.adapreload.instrumentation.model

/**
 * The frozen Layer 2 adapter (Phase D): Linear(64 -> 88) in FP32, zero-initialized, trained by
 * plain SGD with learning rate 0.001, one step per observed launch. It mirrors the notebook
 * (nb[22] L28-32, L50-51, L68-71; identical in nb[33] L127-130, L148-149, L173-177):
 *
 * - adapterLogits = W hidden + b
 * - finalLogits = backboneLogits + adapterLogits
 * - loss = CrossEntropy(finalLogits, target) over all 88 outputs. Padding is NOT masked in the
 *   loss, exactly as offline; it is masked only when ranking apps ([Layer2Prediction]).
 * - g = softmax(finalLogits) - onehot(target)
 * - W[j][k] -= lr * g[j] * hidden[k], and b[j] -= lr * g[j]
 *
 * The state is FP32 ([FloatArray]; W is row-major [88][64] like nn.Linear.weight). Each stored
 * value is computed in double from FP32 operands and rounded to FP32 once. The gradient is
 * rounded to FP32 first, as PyTorch holds it in an FP32 tensor.
 */
class Layer2Adapter(val inputSize: Int = HIDDEN_SIZE, val outputSize: Int = OUTPUT_SIZE) {
    private val weight = FloatArray(outputSize * inputSize)
    private val bias = FloatArray(outputSize)

    /** Number of SGD steps applied since creation or the last [reset]. */
    var updateCount: Long = 0
        private set

    fun weightSnapshot(): FloatArray = weight.copyOf()

    fun biasSnapshot(): FloatArray = bias.copyOf()

    /** W hidden + b, each output rounded to FP32. */
    fun adapterLogits(hidden: FloatArray): FloatArray {
        require(hidden.size == inputSize) { "hidden must have $inputSize values, got ${hidden.size}" }
        return FloatArray(outputSize) { j ->
            var acc = bias[j].toDouble()
            val row = j * inputSize
            for (k in 0 until inputSize) acc += weight[row + k].toDouble() * hidden[k]
            acc.toFloat()
        }
    }

    /** backboneLogits + adapterLogits, an FP32 addition per output. */
    fun finalLogits(hidden: FloatArray, backboneLogits: FloatArray): FloatArray {
        require(backboneLogits.size == outputSize) { "backboneLogits must have $outputSize values, got ${backboneLogits.size}" }
        val adapter = adapterLogits(hidden)
        return FloatArray(outputSize) { backboneLogits[it] + adapter[it] }
    }

    /**
     * Applies exactly one SGD step for [target] and returns the cross-entropy loss before the step.
     * The target must be a real app id (1..outputSize-1): padding id 0 is never an observed target.
     */
    fun update(hidden: FloatArray, backboneLogits: FloatArray, target: Int): Double {
        require(target in 1 until outputSize) { "Target $target is not a real app id (1..${outputSize - 1})" }
        val logits = finalLogits(hidden, backboneLogits)
        var max = Double.NEGATIVE_INFINITY
        for (z in logits) if (z > max) max = z.toDouble()
        val e = DoubleArray(outputSize) { StrictMath.exp(logits[it] - max) }
        val sum = e.sum()
        val loss = StrictMath.log(sum) + max - logits[target]
        val lr = LEARNING_RATE.toDouble()
        for (j in 0 until outputSize) {
            val g = (e[j] / sum - if (j == target) 1.0 else 0.0).toFloat().toDouble()
            val row = j * inputSize
            for (k in 0 until inputSize) weight[row + k] = (weight[row + k] - lr * g * hidden[k]).toFloat()
            bias[j] = (bias[j] - lr * g).toFloat()
        }
        updateCount++
        return loss
    }

    /** Back to the initial state: W = 0, b = 0, no updates. */
    fun reset() {
        weight.fill(0f)
        bias.fill(0f)
        updateCount = 0
    }

    companion object {
        const val HIDDEN_SIZE = 64
        const val OUTPUT_SIZE = 88

        /** 0.001 as FP32 (0.0010000000474974513): PyTorch applies the learning rate to FP32 parameters in FP32. */
        const val LEARNING_RATE = 0.001f
    }
}

/**
 * Layer 2 output for one context: the FP32 [finalLogits] (88, index = app id) and, for ranking,
 * [probabilities] over the 87 real apps: softmax with the padding logit excluded, and
 * probabilities[k] is app id k + 1 (as for Layer 1 and nb[33] L154-156).
 */
class Layer2Prediction(val finalLogits: FloatArray, val updatesBefore: Long) {
    val probabilities: DoubleArray

    init {
        var max = Double.NEGATIVE_INFINITY
        for (k in 1 until finalLogits.size) if (finalLogits[k] > max) max = finalLogits[k].toDouble()
        val e = DoubleArray(finalLogits.size - 1) { StrictMath.exp(finalLogits[it + 1] - max) }
        val sum = e.sum()
        probabilities = DoubleArray(e.size) { e[it] / sum }
    }

    /** App ids by descending probability; ties keep the lower id first. */
    fun rankedAppIds(): IntArray =
        probabilities.indices.sortedWith(compareByDescending<Int> { probabilities[it] }.thenBy { it })
            .map { it + 1 }.toIntArray()
}

/**
 * Enforces the frozen live ordering for one adapter:
 * predict -> record the prediction -> reveal the actual target -> update exactly once.
 *
 * [predict] uses the state left by targets revealed so far, so the prediction for event t only
 * reflects events up to t - 1. The target for t is not available until [reveal], which applies
 * the single update for that prediction with the same inputs the prediction used.
 */
class Layer2OnlineLearner(val adapter: Layer2Adapter = Layer2Adapter()) {

    class Update(val prediction: Layer2Prediction, val target: Int, val lossBefore: Double, val updateCount: Long)

    private class Pending(val hidden: FloatArray, val backboneLogits: FloatArray, val prediction: Layer2Prediction)

    private var pending: Pending? = null

    val hasPendingPrediction: Boolean get() = pending != null

    fun predict(hidden: FloatArray, backboneLogits: FloatArray): Layer2Prediction {
        check(pending == null) { "The previous prediction's target has not been revealed" }
        val prediction = Layer2Prediction(adapter.finalLogits(hidden, backboneLogits), adapter.updateCount)
        pending = Pending(hidden.copyOf(), backboneLogits.copyOf(), prediction)
        return prediction
    }

    fun reveal(target: Int): Update {
        val p = checkNotNull(pending) { "No prediction is waiting for its target" }
        val loss = adapter.update(p.hidden, p.backboneLogits, target)
        pending = null
        return Update(p.prediction, target, loss, adapter.updateCount)
    }

    /** Zeroes the adapter and drops any pending prediction. */
    fun reset() {
        adapter.reset()
        pending = null
    }
}

/** Layer 2 inputs in FP32, as offline, where hidden and backbone logits are FP32 tensors. */
fun Layer1Prediction.hiddenFp32(): FloatArray = FloatArray(hidden.size) { hidden[it].toFloat() }

fun Layer1Prediction.logitsFp32(): FloatArray = FloatArray(logits.size) { logits[it].toFloat() }
