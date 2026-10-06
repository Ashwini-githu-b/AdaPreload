package com.adapreload.instrumentation.collect

import android.content.ContentValues
import android.content.Context
import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import com.adapreload.instrumentation.live.Layer2Commit
import com.adapreload.instrumentation.live.Layer2LogEntry
import com.adapreload.instrumentation.live.Layer2State
import com.adapreload.instrumentation.live.Layer2StateCodec
import com.adapreload.instrumentation.live.Layer2Store
import com.adapreload.instrumentation.live.Reveal
import com.adapreload.instrumentation.shadow.DecisionReason
import com.adapreload.instrumentation.shadow.Eligibility
import com.adapreload.instrumentation.shadow.ShadowDecision
import com.adapreload.instrumentation.shadow.ShadowDecisionRecord
import com.adapreload.instrumentation.shadow.ShadowPolicyConfig
import com.adapreload.instrumentation.shadow.ShadowPreloadDecision
import com.adapreload.instrumentation.trace.Classification
import com.adapreload.instrumentation.trace.CloseReason
import com.adapreload.instrumentation.trace.EnvironmentSnapshot
import com.adapreload.instrumentation.trace.ObservationWindow
import com.adapreload.instrumentation.trace.Outcome
import com.adapreload.instrumentation.trace.PersistedState
import com.adapreload.instrumentation.trace.RawUsageEvent
import com.adapreload.instrumentation.trace.StoredRecord
import com.adapreload.instrumentation.trace.TraceCounts
import com.adapreload.instrumentation.trace.TraceRecord
import com.adapreload.instrumentation.trace.TraceSnapshot
import com.adapreload.instrumentation.trace.TraceState
import com.adapreload.instrumentation.trace.WindowInfo

/**
 * SQLite persistence for the trace. The events table is the single source of truth for the
 * sequence: the sequence state is derived from it on load, and every batch of records is
 * committed together with the new cursor, so processing is exactly-once across restarts.
 *
 * Phase D2 (schema version 2) adds the Layer 2 state and log. They are written in the same
 * transaction as the batch whose launches changed them, so the trace and Layer 2 can never
 * disagree about which launches were processed.
 *
 * Phase E1 (schema version 3) adds the shadow preload decisions, also written in that
 * transaction. Upgrades only add tables; existing rows are never changed.
 */
class SqliteTraceStore(context: Context, databaseName: String = DB_NAME) :
    SQLiteOpenHelper(context, databaseName, null, DB_VERSION), Layer2Store {

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(
            """CREATE TABLE windows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_ms INTEGER NOT NULL,
                end_ms INTEGER,
                close_reason TEXT,
                time_zone TEXT NOT NULL,
                self_package TEXT NOT NULL,
                home_package TEXT,
                ime_packages TEXT NOT NULL,
                profile_count INTEGER NOT NULL,
                mapping_sha256 TEXT NOT NULL,
                vocabulary_sha256 TEXT NOT NULL,
                api_level INTEGER NOT NULL,
                device TEXT NOT NULL)"""
        )
        db.execSQL(
            """CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                window_id INTEGER NOT NULL REFERENCES windows(id),
                observed_at_ms INTEGER NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                event_type INTEGER NOT NULL,
                package TEXT NOT NULL,
                class_name TEXT,
                outcome TEXT NOT NULL,
                classification TEXT,
                app_id INTEGER,
                lsapp_name TEXT,
                sequence_position INTEGER,
                timestamp_anomaly INTEGER NOT NULL)"""
        )
        db.execSQL("CREATE UNIQUE INDEX events_sequence ON events(sequence_position) WHERE outcome = 'APPENDED'")
        db.execSQL("CREATE TABLE state (key TEXT PRIMARY KEY, value INTEGER NOT NULL)")
        createLayer2Tables(db)
        createShadowTables(db)
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        // Version 1 is the Phase B schema; version 2 adds the Layer 2 tables; version 3 adds the shadow decisions.
        check(oldVersion in 1..2 && newVersion == 3) { "No schema migration defined (version $oldVersion → $newVersion)" }
        if (oldVersion < 2) createLayer2Tables(db)
        createShadowTables(db)
    }

    private fun createShadowTables(db: SQLiteDatabase) {
        // One row per examined candidate of a prediction; the key forbids deciding a candidate twice.
        // actual_app_id is filled when the next launch reveals the prediction.
        db.execSQL(
            """CREATE TABLE shadow_decisions (
                prediction_position INTEGER NOT NULL,
                rank INTEGER NOT NULL,
                app_id INTEGER NOT NULL,
                package TEXT,
                probability REAL NOT NULL,
                eligibility TEXT NOT NULL,
                decision TEXT NOT NULL,
                reason TEXT NOT NULL,
                current_app_id INTEGER NOT NULL,
                decided_at_ms INTEGER NOT NULL,
                actual_app_id INTEGER,
                policy_version TEXT NOT NULL,
                policy_enabled INTEGER NOT NULL,
                min_probability REAL NOT NULL,
                max_candidates INTEGER NOT NULL,
                max_rank INTEGER NOT NULL,
                exclude_current_app INTEGER NOT NULL,
                PRIMARY KEY (prediction_position, rank))"""
        )
    }

    private fun createLayer2Tables(db: SQLiteDatabase) {
        // One row: the Layer 2 state in Layer2StateCodec form (versioned, with its own SHA-256).
        db.execSQL("CREATE TABLE layer2_state (id INTEGER PRIMARY KEY CHECK (id = 1), state BLOB NOT NULL)")
        // One row per APPENDED launch processed by Layer 2; the key forbids a second prediction for a launch.
        db.execSQL(
            """CREATE TABLE layer2_log (
                sequence_position INTEGER PRIMARY KEY,
                app_id INTEGER NOT NULL,
                revealed_position INTEGER,
                layer1_rank INTEGER,
                layer2_rank INTEGER,
                loss REAL,
                update_count INTEGER NOT NULL,
                layer1_top1 INTEGER NOT NULL,
                layer2_top1 INTEGER NOT NULL)"""
        )
    }

    override fun loadState(): PersistedState {
        val db = readableDatabase
        val count = db.longFor("SELECT COUNT(*) FROM events WHERE outcome = 'APPENDED'")!!
        val lastApp = db.longFor("SELECT app_id FROM events WHERE outcome = 'APPENDED' ORDER BY sequence_position DESC LIMIT 1")
        val lastTimestamp = db.longFor("SELECT timestamp_ms FROM events ORDER BY id DESC LIMIT 1")
        return PersistedState(
            trace = TraceState(count.toInt(), lastApp?.toInt(), lastTimestamp),
            cursorMs = db.stateValue(KEY_CURSOR),
            openWindowId = db.stateValue(KEY_OPEN_WINDOW),
            experimentStartMs = db.stateValue(KEY_EXPERIMENT_START),
            lastPollMs = db.stateValue(KEY_LAST_POLL),
        )
    }

    override fun openWindow(info: WindowInfo): Long = writableDatabase.inTransaction {
        val env = info.environment
        val id = insertOrThrow("windows", null, ContentValues().apply {
            put("start_ms", info.startMs)
            put("time_zone", info.timeZoneId)
            put("self_package", env.selfPackage)
            put("home_package", env.defaultHomePackage)
            put("ime_packages", env.imePackages.sorted().joinToString(","))
            put("profile_count", info.profileCount)
            put("mapping_sha256", info.mappingSha256)
            put("vocabulary_sha256", info.vocabularySha256)
            put("api_level", info.apiLevel)
            put("device", info.device)
        })
        setState(KEY_OPEN_WINDOW, id)
        setState(KEY_CURSOR, info.startMs)
        if (stateValue(KEY_EXPERIMENT_START) == null) setState(KEY_EXPERIMENT_START, info.startMs)
        id
    }

    override fun closeWindow(windowId: Long, endMs: Long, reason: CloseReason) {
        writableDatabase.inTransaction {
            update("windows", ContentValues().apply {
                put("end_ms", endMs)
                put("close_reason", reason.name)
            }, "id = ?", arrayOf(windowId.toString()))
            if (stateValue(KEY_OPEN_WINDOW) == windowId) {
                delete("state", "key = ?", arrayOf(KEY_OPEN_WINDOW))
            }
            Unit
        }
    }

    override fun commitBatch(records: List<TraceRecord>, cursorMs: Long, polledAtMs: Long) {
        writableDatabase.inTransaction { writeBatch(records, cursorMs, polledAtMs) }
    }

    override fun commitBatch(
        records: List<TraceRecord>,
        cursorMs: Long,
        polledAtMs: Long,
        layer2: Layer2Commit,
        shadow: List<ShadowPreloadDecision>,
    ) {
        writableDatabase.inTransaction {
            writeBatch(records, cursorMs, polledAtMs)
            insertWithOnConflict("layer2_state", null, ContentValues().apply {
                put("id", 1)
                put("state", Layer2StateCodec.encode(layer2.state))
            }, SQLiteDatabase.CONFLICT_REPLACE)
            for (e in layer2.log) {
                insertOrThrow("layer2_log", null, ContentValues().apply {
                    put("sequence_position", e.sequencePosition)
                    put("app_id", e.appId)
                    put("revealed_position", e.reveal?.predictionPosition)
                    put("layer1_rank", e.reveal?.layer1Rank)
                    put("layer2_rank", e.reveal?.layer2Rank)
                    put("loss", e.reveal?.loss)
                    put("update_count", e.updateCount)
                    put("layer1_top1", e.layer1Top1)
                    put("layer2_top1", e.layer2Top1)
                })
            }
            for (d in shadow) {
                insertOrThrow("shadow_decisions", null, ContentValues().apply {
                    put("prediction_position", d.predictionPosition)
                    put("rank", d.rank)
                    put("app_id", d.appId)
                    put("package", d.packageName)
                    put("probability", d.probability)
                    put("eligibility", d.eligibility.name)
                    put("decision", d.decision.name)
                    put("reason", d.reason.name)
                    put("current_app_id", d.currentAppId)
                    put("decided_at_ms", polledAtMs)
                    put("policy_version", ShadowPolicyConfig.VERSION)
                    put("policy_enabled", if (d.config.enabled) 1 else 0)
                    put("min_probability", d.config.minProbability)
                    put("max_candidates", d.config.maxCandidates)
                    put("max_rank", d.config.maxRank)
                    put("exclude_current_app", if (d.config.excludeCurrentApp) 1 else 0)
                })
            }
            // Each revealed launch is the actual next app of the prediction it revealed.
            for (e in layer2.log) {
                val reveal = e.reveal ?: continue
                update("shadow_decisions", ContentValues().apply { put("actual_app_id", e.appId) },
                    "prediction_position = ?", arrayOf(reveal.predictionPosition.toString()))
            }
        }
    }

    override fun shadowDecisions(): List<ShadowDecisionRecord> =
        readableDatabase.rawQuery("SELECT * FROM shadow_decisions ORDER BY prediction_position, rank", null).use { c ->
            c.map { r ->
                val config = ShadowPolicyConfig(
                    enabled = r.long("policy_enabled") == 1L,
                    maxCandidates = r.long("max_candidates")!!.toInt(),
                    maxRank = r.long("max_rank")!!.toInt(),
                    minProbability = r.getDouble(r.getColumnIndexOrThrow("min_probability")),
                    excludeCurrentApp = r.long("exclude_current_app") == 1L,
                )
                ShadowDecisionRecord(
                    decision = ShadowPreloadDecision(
                        predictionPosition = r.long("prediction_position")!!.toInt(),
                        currentAppId = r.long("current_app_id")!!.toInt(),
                        rank = r.long("rank")!!.toInt(),
                        appId = r.long("app_id")!!.toInt(),
                        packageName = r.string("package"),
                        probability = r.getDouble(r.getColumnIndexOrThrow("probability")),
                        eligibility = Eligibility.valueOf(r.string("eligibility")!!),
                        decision = ShadowDecision.valueOf(r.string("decision")!!),
                        reason = DecisionReason.valueOf(r.string("reason")!!),
                        config = config,
                    ),
                    decidedAtMs = r.long("decided_at_ms")!!,
                    actualAppId = r.long("actual_app_id")?.toInt(),
                )
            }
        }

    override fun loadLayer2(): Layer2State? =
        readableDatabase.rawQuery("SELECT state FROM layer2_state WHERE id = 1", null).use {
            if (it.moveToFirst()) Layer2StateCodec.decode(it.getBlob(0)) else null
        }

    override fun recentLaunches(limit: Int): List<Int> =
        readableDatabase.rawQuery(
            "SELECT app_id FROM events WHERE outcome = 'APPENDED' ORDER BY sequence_position DESC LIMIT $limit", null,
        ).use { c -> c.map { it.long("app_id")!!.toInt() } }.reversed()

    override fun lastLayer2Log(): Layer2LogEntry? =
        readableDatabase.rawQuery("SELECT * FROM layer2_log ORDER BY sequence_position DESC LIMIT 1", null).use { c ->
            c.map { r ->
                Layer2LogEntry(
                    sequencePosition = r.long("sequence_position")!!.toInt(),
                    appId = r.long("app_id")!!.toInt(),
                    reveal = r.long("revealed_position")?.let { position ->
                        Reveal(position.toInt(), r.long("layer1_rank")!!.toInt(), r.long("layer2_rank")!!.toInt(), r.getDouble(r.getColumnIndexOrThrow("loss")))
                    },
                    updateCount = r.long("update_count")!!,
                    layer1Top1 = r.long("layer1_top1")!!.toInt(),
                    layer2Top1 = r.long("layer2_top1")!!.toInt(),
                )
            }.firstOrNull()
        }

    private fun SQLiteDatabase.writeBatch(records: List<TraceRecord>, cursorMs: Long, polledAtMs: Long) {
        for (r in records) {
            insertOrThrow("events", null, ContentValues().apply {
                put("window_id", r.windowId)
                put("observed_at_ms", r.observedAtMs)
                put("timestamp_ms", r.event.timestampMs)
                put("event_type", r.event.eventType)
                put("package", r.event.packageName)
                put("class_name", r.event.className)
                put("outcome", r.outcome.name)
                put("classification", r.classification?.name)
                put("app_id", r.appId)
                put("lsapp_name", r.lsappName)
                put("sequence_position", r.sequencePosition)
                put("timestamp_anomaly", if (r.timestampAnomaly) 1 else 0)
            })
        }
        setState(KEY_CURSOR, cursorMs)
        setState(KEY_LAST_POLL, polledAtMs)
    }

    override fun snapshot(): TraceSnapshot {
        val db = readableDatabase
        val state = loadState()
        val windows = db.rawQuery("SELECT * FROM windows ORDER BY id", null).use { c -> c.map(::window) }
        val records = db.rawQuery("SELECT * FROM events ORDER BY id", null).use { c -> c.map { StoredRecord(it.long("id")!!, record(it)) } }
        return TraceSnapshot(state.experimentStartMs, state.cursorMs, state.lastPollMs, windows, records)
    }

    override fun counts(): TraceCounts {
        val db = readableDatabase
        fun count(where: String) = db.longFor("SELECT COUNT(*) FROM events WHERE $where")!!.toInt()
        fun inGroup(group: Classification.Group) =
            Classification.entries.filter { it.group == group }.joinToString(",", "classification IN (", ")") { "'${it.name}'" }
        val last = db.rawQuery(
            "SELECT * FROM events WHERE classification IS NOT NULL ORDER BY id DESC LIMIT 1", null
        ).use { c -> c.map(::record).firstOrNull() }
        return TraceCounts(
            observedEvents = count("1"),
            launchCandidates = count("classification IS NOT NULL"),
            supportedLaunches = count("classification = 'SUPPORTED'"),
            sequenceLength = count("outcome = 'APPENDED'"),
            collapsed = count("outcome = 'COLLAPSED'"),
            unmapped = count(inGroup(Classification.Group.UNMAPPED)),
            excluded = count(inGroup(Classification.Group.EXCLUDED)),
            windows = db.longFor("SELECT COUNT(*) FROM windows")!!.toInt(),
            lastCandidate = last,
            lastPollMs = db.stateValue(KEY_LAST_POLL),
        )
    }

    private fun window(c: Cursor) = ObservationWindow(
        id = c.long("id")!!,
        info = WindowInfo(
            startMs = c.long("start_ms")!!,
            timeZoneId = c.string("time_zone")!!,
            environment = EnvironmentSnapshot(
                selfPackage = c.string("self_package")!!,
                defaultHomePackage = c.string("home_package"),
                imePackages = c.string("ime_packages")!!.split(',').filter { it.isNotEmpty() }.toSet(),
            ),
            profileCount = c.long("profile_count")!!.toInt(),
            mappingSha256 = c.string("mapping_sha256")!!,
            vocabularySha256 = c.string("vocabulary_sha256")!!,
            apiLevel = c.long("api_level")!!.toInt(),
            device = c.string("device")!!,
        ),
        endMs = c.long("end_ms"),
        closeReason = c.string("close_reason")?.let(CloseReason::valueOf),
    )

    private fun record(c: Cursor) = TraceRecord(
        windowId = c.long("window_id")!!,
        observedAtMs = c.long("observed_at_ms")!!,
        event = RawUsageEvent(c.long("timestamp_ms")!!, c.long("event_type")!!.toInt(), c.string("package")!!, c.string("class_name")),
        outcome = Outcome.valueOf(c.string("outcome")!!),
        classification = c.string("classification")?.let(Classification::valueOf),
        appId = c.long("app_id")?.toInt(),
        lsappName = c.string("lsapp_name"),
        sequencePosition = c.long("sequence_position")?.toInt(),
        timestampAnomaly = c.long("timestamp_anomaly") == 1L,
    )

    private fun <T> SQLiteDatabase.inTransaction(block: SQLiteDatabase.() -> T): T {
        beginTransaction()
        try {
            val result = block()
            setTransactionSuccessful()
            return result
        } finally {
            endTransaction()
        }
    }

    private fun SQLiteDatabase.longFor(sql: String): Long? =
        rawQuery(sql, null).use { if (it.moveToFirst() && !it.isNull(0)) it.getLong(0) else null }

    private fun SQLiteDatabase.stateValue(key: String): Long? =
        rawQuery("SELECT value FROM state WHERE key = ?", arrayOf(key)).use { if (it.moveToFirst()) it.getLong(0) else null }

    private fun SQLiteDatabase.setState(key: String, value: Long) {
        insertWithOnConflict("state", null, ContentValues().apply {
            put("key", key)
            put("value", value)
        }, SQLiteDatabase.CONFLICT_REPLACE)
    }

    private fun Cursor.long(column: String): Long? = getColumnIndexOrThrow(column).let { if (isNull(it)) null else getLong(it) }

    private fun Cursor.string(column: String): String? = getColumnIndexOrThrow(column).let { if (isNull(it)) null else getString(it) }

    private fun <T> Cursor.map(transform: (Cursor) -> T): List<T> {
        val out = ArrayList<T>(count)
        while (moveToNext()) out += transform(this)
        return out
    }

    companion object {
        const val DB_NAME = "adapreload_trace.db"
        private const val DB_VERSION = 3
        private const val KEY_CURSOR = "cursor_ms"
        private const val KEY_OPEN_WINDOW = "open_window_id"
        private const val KEY_EXPERIMENT_START = "experiment_start_ms"
        private const val KEY_LAST_POLL = "last_poll_ms"
    }
}
