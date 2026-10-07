package com.adapreload.instrumentation.eval

import android.content.Context
import com.adapreload.instrumentation.collect.SqliteTraceStore
import com.adapreload.instrumentation.shadow.ShadowDecisionRecord
import com.adapreload.instrumentation.trace.Outcome
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * DEBUG-ONLY. Reproduces the off-device JVM replay on the phone by *exporting what Phase D2 (Layer 2)
 * and Phase E1 (the shadow policy) already recorded live* — it runs no model, applies no Layer 2
 * update, opens no binding, and never writes to the trace database. It is strictly read-only:
 * `snapshot()`, `shadowDecisions()` and a `SELECT` over the frozen v3 `layer2_log` table.
 *
 * It joins three already-persisted sources on the canonical sequence position:
 *   - `events` (APPENDED)      -> position, app id, lsapp name, timestamp, window id, and the actual next app;
 *   - `layer2_log`             -> the recorded Layer 1 and Layer 2 top-1 for the prediction made after each launch;
 *   - `shadow_decisions` (rank 1) -> the recorded E1 PRELOAD/SKIP decision, its reason, and the top-1 probability.
 *
 * The output file is formatted identically to the off-device JVM replay (tools/replay dry run) so the
 * two can be diffed line-for-line to confirm the device reproduced the same 40-row result.
 *
 * Present only in debug builds.
 */
object DryRunEvaluator {

    data class Result(val file: File, val summary: List<String>)

    /** One prediction point, as recorded on the device. */
    private data class Point(
        val idx: Int,
        val ts: Long,
        val actual: Int,
        val l1Top1: Int,
        val l2Top1: Int,
        val l1Hit: Boolean,
        val l2Hit: Boolean,
        val change: String,
        val decision: String,
        val reason: String,
        val top1Prob: Double?,
        val windowId: Long,
    )

    fun export(context: Context): Result {
        val store = SqliteTraceStore(context.applicationContext)
        try {
            val lines = ArrayList<String>()
            val summary = ArrayList<String>()
            fun out(s: String) = lines.add(s)
            fun sum(s: String) { summary.add(s); lines.add(s) }

            val snapshot = store.snapshot()
            val appended = snapshot.records
                .map { it.record }
                .filter { it.outcome == Outcome.APPENDED }
                .sortedBy { it.sequencePosition!! }
            val seq = appended.map { it.appId!! }
            val tsByPos = appended.associate { it.sequencePosition!! to it.event.timestampMs }
            val winByPos = appended.associate { it.sequencePosition!! to it.windowId }
            val idToName = appended.associate { it.appId!! to (it.lsappName ?: "?") }
            fun nm(id: Int) = "$id:${idToName[id] ?: "?"}"

            // layer2_log: recorded Layer 1 / Layer 2 top-1 per prediction position (read-only, frozen v3 schema).
            val l1Top1 = HashMap<Int, Int>()
            val l2Top1 = HashMap<Int, Int>()
            store.readableDatabase.rawQuery(
                "SELECT sequence_position, layer1_top1, layer2_top1 FROM layer2_log ORDER BY sequence_position",
                null,
            ).use { c ->
                val pi = c.getColumnIndexOrThrow("sequence_position")
                val a = c.getColumnIndexOrThrow("layer1_top1")
                val b = c.getColumnIndexOrThrow("layer2_top1")
                while (c.moveToNext()) {
                    val p = c.getInt(pi)
                    l1Top1[p] = c.getInt(a)
                    l2Top1[p] = c.getInt(b)
                }
            }

            // shadow_decisions rank-1 row per prediction position.
            val shadow: Map<Int, ShadowDecisionRecord> = store.shadowDecisions()
                .filter { it.decision.rank == 1 }
                .associateBy { it.decision.predictionPosition }

            // Provenance / sanity header (not part of the comparable table block).
            val vocabSha = snapshot.windows.firstOrNull()?.info?.vocabularySha256
            out("DRYRUN source=on-device-recorded(layer2_log+shadow_decisions) exported_at=${iso(System.currentTimeMillis())}")
            out("PROVENANCE sequence_length=${seq.size} layer2_log_rows=${l1Top1.size} shadow_rank1_rows=${shadow.size} vocab_sha256=$vocabSha")

            if (seq.isEmpty() || l1Top1.isEmpty()) {
                out("DRYRUN_EMPTY no APPENDED launches or no layer2_log — nothing to evaluate.")
                return writeResult(context, lines, summary)
            }

            // Evaluable points: predictions made after launch p that have an observed next launch (p in 0..N-2).
            val points = ArrayList<Point>()
            var rank1EqL2 = 0
            var actualMatches = 0
            var missingL2Log = 0
            var missingShadow = 0
            for (p in 0 until seq.size - 1) {
                val l1 = l1Top1[p]
                val l2 = l2Top1[p]
                if (l1 == null || l2 == null) { missingL2Log++; continue }
                val actual = seq[p + 1]
                val l1hit = l1 == actual
                val l2hit = l2 == actual
                val change = when {
                    l1 == l2 -> "no_change"
                    l2hit && !l1hit -> "helpful"
                    l1hit && !l2hit -> "harmful"
                    else -> "neutral_change"
                }
                val sd = shadow[p]
                if (sd == null) missingShadow++
                if (sd != null && sd.decision.appId == l2) rank1EqL2++
                if (sd?.actualAppId == actual) actualMatches++
                points += Point(
                    idx = p,
                    ts = tsByPos[p] ?: -1,
                    actual = actual,
                    l1Top1 = l1,
                    l2Top1 = l2,
                    l1Hit = l1hit,
                    l2Hit = l2hit,
                    change = change,
                    decision = sd?.decision?.decision?.name ?: "NR",
                    reason = sd?.decision?.reasonLabel ?: "not_recorded",
                    top1Prob = sd?.decision?.probability,
                    windowId = winByPos[p] ?: -1,
                )
            }

            out("REPLAY_TABLE_BEGIN")
            out("idx | ts_ms | actual_next | L1_top1 | L2_top1 | L1_hit | L2_hit | L2_change | E1 | E1_reason | top1_prob | win")
            for (pt in points) {
                val prob = pt.top1Prob?.let { "%.4f".format(it) } ?: "  nan "
                out(
                    "%2d | %d | %-14s | %-14s | %-14s | %s | %s | %-14s | %-7s | %-28s | %s | %d".format(
                        pt.idx, pt.ts, nm(pt.actual), nm(pt.l1Top1), nm(pt.l2Top1),
                        if (pt.l1Hit) "Y" else "n", if (pt.l2Hit) "Y" else "n",
                        pt.change, pt.decision, pt.reason, prob, pt.windowId,
                    ),
                )
            }
            out("REPLAY_TABLE_END")

            // Compact summary.
            val n = points.size
            val l1Hits = points.count { it.l1Hit }
            val l2Hits = points.count { it.l2Hit }
            val preload = points.count { it.decision == "PRELOAD" }
            val skip = points.count { it.decision == "SKIP" }
            val changeCounts = linkedMapOf("helpful" to 0, "harmful" to 0, "neutral_change" to 0, "no_change" to 0)
            points.forEach { changeCounts[it.change] = (changeCounts[it.change] ?: 0) + 1 }
            val reasonCounts = linkedMapOf<String, Int>()
            points.forEach { reasonCounts[it.reason] = (reasonCounts[it.reason] ?: 0) + 1 }
            val preloadHits = points.count { it.decision == "PRELOAD" && it.l2Hit }

            sum("SUMMARY evaluable=$n L1_hit@1=%d(%.3f) L2_hit@1=%d(%.3f) preload=%d skip=%d".format(
                l1Hits, if (n > 0) l1Hits.toDouble() / n else 0.0,
                l2Hits, if (n > 0) l2Hits.toDouble() / n else 0.0, preload, skip))
            sum("CHANGE $changeCounts")
            sum("DECISION_PCT preload=%d(%.1f%%) skip=%d(%.1f%%)".format(
                preload, if (n > 0) 100.0 * preload / n else 0.0,
                skip, if (n > 0) 100.0 * skip / n else 0.0))
            sum("E1_REASONS $reasonCounts")
            sum("PRELOAD_PRECISION %d/%d=%.3f".format(preloadHits, preload, if (preload > 0) preloadHits.toDouble() / preload else 0.0))
            sum("CONSISTENCY rank1_app==layer2_top1:%d/%d shadow_actual==next:%d/%d missing_l2log:%d missing_shadow:%d".format(
                rank1EqL2, preload + skip, actualMatches, n, missingL2Log, missingShadow))

            return writeResult(context, lines, summary)
        } finally {
            store.close()
        }
    }

    private fun writeResult(context: Context, lines: List<String>, summary: List<String>): Result {
        val dir = context.getExternalFilesDir(null) ?: context.filesDir
        val file = File(dir, "dryrun_eval_${stamp()}.txt")
        file.writeText(lines.joinToString("\n") + "\n")
        return Result(file, summary)
    }

    private fun stamp(): String = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
    private fun iso(ms: Long): String = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US).format(Date(ms))
}
