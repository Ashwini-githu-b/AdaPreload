package com.adapreload.instrumentation.live

import com.adapreload.instrumentation.trace.TraceRecord
import com.adapreload.instrumentation.trace.TraceStore
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.security.MessageDigest

/**
 * The prediction made after launch [position], waiting for the next launch to reveal its target.
 * It keeps its exact inputs so the update uses them, and its output so a restore can be verified.
 */
class PendingPrediction(
    /** Sequence position of the launch this prediction was made after (the last context launch). */
    val position: Int,
    /** The Layer 1 context: the last <= 20 launches through [position], oldest first. */
    val context: IntArray,
    /** Layer 1 hidden vector, FP32 (64). */
    val hidden: FloatArray,
    /** Layer 1 logits, FP32 (88). */
    val backboneLogits: FloatArray,
    /** Layer 2 final logits at prediction time (88). */
    val finalLogits: FloatArray,
)

/** What the persisted state was computed with; it must match the running model to be resumed. */
data class Layer2Identity(
    val checkpointSha256: String,
    val layer1WeightsSha256: String,
    val vocabularySha256: String,
    /** Float.toRawBits of the learning rate. */
    val learningRateBits: Int,
)

/** Everything needed to resume Layer 2 exactly. */
class Layer2State(
    val identity: Layer2Identity,
    val weight: FloatArray,
    val bias: FloatArray,
    val updateCount: Long,
    val pending: PendingPrediction?,
)

/** How the launch at [Layer2LogEntry.sequencePosition] scored the prediction it revealed. */
data class Reveal(
    val predictionPosition: Int,
    /** Rank of the revealed app among the 87 real apps (1 = top-1), Layer 1 alone and with the adapter. */
    val layer1Rank: Int,
    val layer2Rank: Int,
    /** Cross-entropy of the prediction against the revealed app, before the update. */
    val loss: Double,
)

/** Diagnostic record of what one APPENDED launch did to Layer 2. */
data class Layer2LogEntry(
    val sequencePosition: Int,
    val appId: Int,
    /** The pending prediction this launch revealed and updated on; null for the first launch. */
    val reveal: Reveal?,
    /** Updates applied after this launch; the prediction made after it is based on exactly these. */
    val updateCount: Long,
    /** Top-1 of the prediction made after this launch (the next app), Layer 1 alone and with the adapter. */
    val layer1Top1: Int,
    val layer2Top1: Int,
)

/** Layer 2 work to commit in the same transaction as the batch of records that caused it. */
class Layer2Commit(val state: Layer2State, val log: List<Layer2LogEntry>)

/** Trace persistence that can also commit Layer 2 state atomically with a batch. */
interface Layer2Store : TraceStore {
    /** The persisted Layer 2 state, or null if Layer 2 has never committed. */
    fun loadLayer2(): Layer2State?

    /** App ids of the last [limit] APPENDED launches, oldest first. */
    fun recentLaunches(limit: Int): List<Int>

    /** Appends the records, advances the cursor, replaces the Layer 2 state and appends its log, all in one transaction. */
    fun commitBatch(records: List<TraceRecord>, cursorMs: Long, polledAtMs: Long, layer2: Layer2Commit)

    /** The latest Layer 2 log entry, if any. */
    fun lastLayer2Log(): Layer2LogEntry?
}

/**
 * Binary form of [Layer2State]: big-endian fields, floats as raw IEEE bits, followed by the
 * SHA-256 of everything before it. [decode] rejects any other version, a bad digest or
 * trailing bytes.
 */
object Layer2StateCodec {
    const val FORMAT_VERSION = 1
    private const val MAGIC = 0x4C32_5354 // "L2ST"
    private const val DIGEST_BYTES = 32

    fun encode(state: Layer2State): ByteArray {
        val body = ByteArrayOutputStream()
        DataOutputStream(body).use { out ->
            out.writeInt(MAGIC)
            out.writeInt(FORMAT_VERSION)
            with(state.identity) {
                out.writeUTF(checkpointSha256)
                out.writeUTF(layer1WeightsSha256)
                out.writeUTF(vocabularySha256)
                out.writeInt(learningRateBits)
            }
            out.writeFloats(state.weight)
            out.writeFloats(state.bias)
            out.writeLong(state.updateCount)
            val p = state.pending
            out.writeBoolean(p != null)
            if (p != null) {
                out.writeInt(p.position)
                out.writeInt(p.context.size)
                p.context.forEach(out::writeInt)
                out.writeFloats(p.hidden)
                out.writeFloats(p.backboneLogits)
                out.writeFloats(p.finalLogits)
            }
        }
        val bytes = body.toByteArray()
        return bytes + sha256(bytes)
    }

    fun decode(bytes: ByteArray): Layer2State {
        require(bytes.size > DIGEST_BYTES) { "Layer 2 state is truncated" }
        val body = bytes.copyOfRange(0, bytes.size - DIGEST_BYTES)
        require(sha256(body).contentEquals(bytes.copyOfRange(body.size, bytes.size))) { "Layer 2 state digest mismatch" }
        val input = DataInputStream(ByteArrayInputStream(body))
        require(input.readInt() == MAGIC) { "Not a Layer 2 state" }
        val version = input.readInt()
        require(version == FORMAT_VERSION) { "Unsupported Layer 2 state version $version" }
        val identity = Layer2Identity(input.readUTF(), input.readUTF(), input.readUTF(), input.readInt())
        val weight = input.readFloats()
        val bias = input.readFloats()
        val updateCount = input.readLong()
        val pending = if (input.readBoolean()) {
            val position = input.readInt()
            val context = IntArray(input.readInt()) { input.readInt() }
            PendingPrediction(position, context, input.readFloats(), input.readFloats(), input.readFloats())
        } else {
            null
        }
        require(input.available() == 0) { "Trailing bytes in Layer 2 state" }
        return Layer2State(identity, weight, bias, updateCount, pending)
    }

    private fun DataOutputStream.writeFloats(values: FloatArray) {
        writeInt(values.size)
        for (v in values) writeInt(v.toRawBits())
    }

    private fun DataInputStream.readFloats(): FloatArray = FloatArray(readInt()) { Float.fromBits(readInt()) }

    private fun sha256(bytes: ByteArray): ByteArray = MessageDigest.getInstance("SHA-256").digest(bytes)
}
