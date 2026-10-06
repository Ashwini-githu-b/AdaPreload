package com.adapreload.instrumentation.model

import com.adapreload.instrumentation.trace.Json
import com.adapreload.instrumentation.trace.sha256Hex
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Frozen Layer 1 architecture, read from the exported manifest (never assumed). */
data class Layer1Config(
    val vocabSize: Int,
    val paddingId: Int,
    val window: Int,
    val dModel: Int,
    val numLayers: Int,
    val numHeads: Int,
    val headDim: Int,
    val dimFeedforward: Int,
    val layerNormEps: Double,
)

/**
 * Layer 1 output for one context.
 *
 * [hidden] is the encoder output at the last real token (dModel entries): the representation the
 * output layer reads, and the input of the Layer 2 adapter (notebook get_hidden, nb[22] L11-17).
 * [logits] has vocabSize entries (index = app id, index 0 = padding). [probabilities] has
 * vocabSize - 1 entries: softmax over the logits with the padding logit set to -inf, and
 * probabilities[k] is app id k + 1 (notebook nb[33] L152-156).
 */
class Layer1Prediction(val hidden: DoubleArray, val logits: DoubleArray, val probabilities: DoubleArray) {
    /** App ids by descending probability; ties keep the lower id first. */
    fun rankedAppIds(): IntArray =
        probabilities.indices.sortedWith(compareByDescending<Int> { probabilities[it] }.thenBy { it })
            .map { it + 1 }.toIntArray()

    fun probabilityOf(appId: Int): Double = probabilities[appId - 1]
}

/**
 * The frozen Layer 1 backbone (notebook BackboneMorph, morph_mode 'none') in plain Kotlin.
 *
 * Mirrors the PyTorch eval-mode computation used to produce the frozen Layer 1 outputs (nb[22]
 * get_hidden, nb[33] L135-158):
 * - token = app embedding + position embedding, with right-aligned positions (window - L .. window - 1);
 * - each encoder layer is post-norm: x = norm1(x + selfAttention(x)); x = norm2(x + linear2(relu(linear1(x))));
 * - the readout is the encoder output at the last token, then the output layer gives logits.
 *
 * Padding tokens are dropped before the encoder, which is exactly equivalent to the PyTorch key
 * padding mask: only real tokens are ever attended to, and positions are unchanged.
 * The float32 weights are used exactly; all arithmetic is in double precision, with
 * deterministic StrictMath functions so JVM and Android results are bit-identical.
 */
class Layer1Model private constructor(
    val config: Layer1Config,
    private val weights: Map<String, DoubleArray>,
    /** SHA-256 of the weights file. */
    val weightsSha256: String,
    /** apps_sha256 of the vocabulary the weights were exported with. */
    val vocabularyAppsSha256: String,
) {

    /**
     * Runs Layer 1 on a context of exactly [Layer1Config.window] app ids, left-padded with the
     * padding id: zero or more padding ids, then at least one real app id, oldest first.
     */
    fun predict(context: IntArray): Layer1Prediction {
        val ids = realTokens(context)
        val n = ids.size
        val d = config.dModel
        val appEmb = w("app_emb.weight")
        val posEmb = w("pos_emb.weight")
        var x = DoubleArray(n * d)
        for (t in 0 until n) {
            val pos = config.window - n + t
            for (j in 0 until d) x[t * d + j] = appEmb[ids[t] * d + j] + posEmb[pos * d + j]
        }
        for (layer in 0 until config.numLayers) x = encoderLayer(x, n, "encoder.layers.$layer.")

        val last = x.copyOfRange((n - 1) * d, n * d)
        val logits = linear(last, 1, d, w("out.weight"), w("out.bias"), config.vocabSize)

        // Softmax over real apps only: the padding logit is excluded (set to -inf in the notebook).
        var max = Double.NEGATIVE_INFINITY
        for (k in 1 until config.vocabSize) if (logits[k] > max) max = logits[k]
        val probabilities = DoubleArray(config.vocabSize - 1) { StrictMath.exp(logits[it + 1] - max) }
        val sum = probabilities.sum()
        for (k in probabilities.indices) probabilities[k] /= sum
        return Layer1Prediction(last, logits, probabilities)
    }

    private fun realTokens(context: IntArray): IntArray {
        require(context.size == config.window) { "Context must have exactly ${config.window} ids, got ${context.size}" }
        val first = context.indexOfFirst { it != config.paddingId }
        require(first >= 0) { "Context has no real app id" }
        val ids = context.copyOfRange(first, context.size)
        for (id in ids) {
            require(id != config.paddingId) { "Padding is only allowed before the first real app id (left padding)" }
            require(id in 1 until config.vocabSize) { "App id $id is outside 1..${config.vocabSize - 1}" }
        }
        return ids
    }

    private fun encoderLayer(x: DoubleArray, n: Int, prefix: String): DoubleArray {
        val d = config.dModel
        val hd = config.headDim
        // in_proj packs the query, key and value projections: rows [0, d), [d, 2d), [2d, 3d).
        val qkv = linear(x, n, d, w(prefix + "self_attn.in_proj_weight"), w(prefix + "self_attn.in_proj_bias"), 3 * d)
        val attended = DoubleArray(n * d)
        val scale = 1.0 / StrictMath.sqrt(hd.toDouble())
        val scores = DoubleArray(n)
        for (h in 0 until config.numHeads) {
            val off = h * hd
            for (i in 0 until n) {
                var max = Double.NEGATIVE_INFINITY
                for (j in 0 until n) {
                    var s = 0.0
                    for (k in 0 until hd) s += qkv[i * 3 * d + off + k] * qkv[j * 3 * d + d + off + k]
                    scores[j] = s * scale
                    if (scores[j] > max) max = scores[j]
                }
                var sum = 0.0
                for (j in 0 until n) {
                    scores[j] = StrictMath.exp(scores[j] - max)
                    sum += scores[j]
                }
                for (k in 0 until hd) {
                    var acc = 0.0
                    for (j in 0 until n) acc += scores[j] * qkv[j * 3 * d + 2 * d + off + k]
                    attended[i * d + off + k] = acc / sum
                }
            }
        }
        val attention = linear(attended, n, d, w(prefix + "self_attn.out_proj.weight"), w(prefix + "self_attn.out_proj.bias"), d)
        val norm1 = layerNorm(add(x, attention), n, w(prefix + "norm1.weight"), w(prefix + "norm1.bias"))
        val hidden = linear(norm1, n, d, w(prefix + "linear1.weight"), w(prefix + "linear1.bias"), config.dimFeedforward)
        for (k in hidden.indices) if (hidden[k] < 0.0) hidden[k] = 0.0 // ReLU
        val feedForward = linear(hidden, n, config.dimFeedforward, w(prefix + "linear2.weight"), w(prefix + "linear2.bias"), d)
        return layerNorm(add(norm1, feedForward), n, w(prefix + "norm2.weight"), w(prefix + "norm2.bias"))
    }

    /** y = x W^T + b for n rows; W is (outDim, inDim) row-major as in PyTorch. */
    private fun linear(x: DoubleArray, n: Int, inDim: Int, weight: DoubleArray, bias: DoubleArray, outDim: Int): DoubleArray {
        val y = DoubleArray(n * outDim)
        for (r in 0 until n) {
            for (o in 0 until outDim) {
                var acc = bias[o]
                for (i in 0 until inDim) acc += x[r * inDim + i] * weight[o * inDim + i]
                y[r * outDim + o] = acc
            }
        }
        return y
    }

    private fun add(a: DoubleArray, b: DoubleArray) = DoubleArray(a.size) { a[it] + b[it] }

    /** LayerNorm over the last dimension: biased variance, eps inside the square root. */
    private fun layerNorm(x: DoubleArray, n: Int, gamma: DoubleArray, beta: DoubleArray): DoubleArray {
        val d = config.dModel
        val y = DoubleArray(x.size)
        for (r in 0 until n) {
            var mean = 0.0
            for (j in 0 until d) mean += x[r * d + j]
            mean /= d
            var variance = 0.0
            for (j in 0 until d) {
                val c = x[r * d + j] - mean
                variance += c * c
            }
            variance /= d
            val inv = 1.0 / StrictMath.sqrt(variance + config.layerNormEps)
            for (j in 0 until d) y[r * d + j] = (x[r * d + j] - mean) * inv * gamma[j] + beta[j]
        }
        return y
    }

    private fun w(name: String): DoubleArray = weights.getValue(name)

    companion object {
        /** The audited checkpoint the weights must come from (ANDROID_TRACE_SPEC Provenance). */
        const val CHECKPOINT_SHA256 = "fd668f160363c9b32fe5ac561bec7b46ff5492d84e10f31d0bfd6d7a0c21c092"

        /**
         * Loads the exported weights, verifying every hash and that the manifest describes the
         * architecture this implementation computes. Throws IllegalArgumentException otherwise.
         */
        fun load(manifestJson: String, weightsFile: ByteArray): Layer1Model {
            val manifest = Json.parse(manifestJson) as Map<*, *>
            require(manifest["format"] == "adapreload-layer1-weights" && manifest["format_version"] == 1L) { "Unknown manifest format" }
            val source = manifest["source"] as Map<*, *>
            require((source["checkpoint"] as Map<*, *>)["sha256"] == CHECKPOINT_SHA256) { "Weights are not from the audited checkpoint" }

            val a = manifest["architecture"] as Map<*, *>
            require(a["morph_mode"] == "none") { "Only morph_mode 'none' is implemented" }
            require(a["activation"] == "relu" && a["norm_first"] == false && a["final_encoder_norm"] == false) {
                "Only the post-norm ReLU encoder without a final norm is implemented"
            }
            fun int(key: String) = (a[key] as Long).toInt()
            val config = Layer1Config(
                vocabSize = int("vocab_size"),
                paddingId = int("padding_id"),
                window = int("window"),
                dModel = int("d_model"),
                numLayers = int("num_layers"),
                numHeads = int("num_heads"),
                headDim = int("head_dim"),
                dimFeedforward = int("dim_feedforward"),
                layerNormEps = a["layer_norm_eps"] as Double,
            )
            require(config.paddingId == 0 && config.numHeads * config.headDim == config.dModel) { "Inconsistent architecture" }

            val weightsInfo = manifest["weights"] as Map<*, *>
            val sha = sha256Hex(weightsFile)
            require(weightsInfo["sha256"] == sha) { "Weights file sha256 $sha does not match the manifest" }
            require(weightsInfo["bytes"] == weightsFile.size.toLong()) { "Weights file size does not match the manifest" }

            val tensors = HashMap<String, DoubleArray>()
            for (t in manifest["tensors"] as List<*>) {
                t as Map<*, *>
                val name = t["name"] as String
                val shape = (t["shape"] as List<*>).map { (it as Long).toInt() }
                val count = (t["count"] as Long).toInt()
                val offset = (t["offset"] as Long).toInt()
                require(shape == expectedShape(name, config)) { "Unexpected shape $shape for $name" }
                require(sha256Hex(weightsFile.copyOfRange(offset, offset + 4 * count)) == t["sha256"]) { "Tensor $name hash mismatch" }
                val floats = FloatArray(count)
                ByteBuffer.wrap(weightsFile, offset, 4 * count).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(floats)
                tensors[name] = DoubleArray(count) { floats[it].toDouble() }
            }
            val expected = expectedNames(config)
            require(tensors.keys == expected) { "Tensor set differs: missing ${expected - tensors.keys}, extra ${tensors.keys - expected}" }
            return Layer1Model(config, tensors, sha, source["vocabulary_apps_sha256"] as String)
        }

        private fun expectedNames(c: Layer1Config): Set<String> =
            setOf("app_emb.weight", "pos_emb.weight", "out.weight", "out.bias") +
                (0 until c.numLayers).flatMap { l -> LAYER_TENSORS.map { "encoder.layers.$l.$it" } }

        private val LAYER_TENSORS = listOf(
            "self_attn.in_proj_weight", "self_attn.in_proj_bias", "self_attn.out_proj.weight", "self_attn.out_proj.bias",
            "linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias",
            "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
        )

        private fun expectedShape(name: String, c: Layer1Config): List<Int> {
            val d = c.dModel
            val key = if (name.startsWith("encoder.layers.")) name.substringAfter("encoder.layers.").substringAfter('.') else name
            return when (key) {
                "app_emb.weight" -> listOf(c.vocabSize, d)
                "pos_emb.weight" -> listOf(c.window, d)
                "out.weight" -> listOf(c.vocabSize, d)
                "out.bias" -> listOf(c.vocabSize)
                "self_attn.in_proj_weight" -> listOf(3 * d, d)
                "self_attn.in_proj_bias" -> listOf(3 * d)
                "self_attn.out_proj.weight" -> listOf(d, d)
                "linear1.weight" -> listOf(c.dimFeedforward, d)
                "linear1.bias" -> listOf(c.dimFeedforward)
                "linear2.weight" -> listOf(d, c.dimFeedforward)
                else -> listOf(d) // out_proj.bias, linear2.bias, norm weights and biases
            }
        }
    }
}
