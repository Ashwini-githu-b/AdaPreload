package com.adapreload.instrumentation

import android.database.sqlite.SQLiteConstraintException
import android.database.sqlite.SQLiteDatabase
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.adapreload.instrumentation.collect.SqliteTraceStore
import com.adapreload.instrumentation.live.Layer2Commit
import com.adapreload.instrumentation.live.Layer2StateCodec
import com.adapreload.instrumentation.live.LivePersonalizer
import com.adapreload.instrumentation.live.PersonalizedTraceStore
import com.adapreload.instrumentation.model.Layer1Assets
import com.adapreload.instrumentation.model.Layer1Model
import com.adapreload.instrumentation.trace.Classification
import com.adapreload.instrumentation.trace.EnvironmentSnapshot
import com.adapreload.instrumentation.trace.LaunchClassifier
import com.adapreload.instrumentation.trace.Outcome
import com.adapreload.instrumentation.trace.PackageMapping
import com.adapreload.instrumentation.trace.RawUsageEvent
import com.adapreload.instrumentation.trace.TracePipeline
import com.adapreload.instrumentation.trace.TraceRecord
import com.adapreload.instrumentation.trace.TraceSession
import com.adapreload.instrumentation.trace.UsageEventSource
import com.adapreload.instrumentation.trace.UsageEventTypes
import com.adapreload.instrumentation.trace.Vocabulary
import com.adapreload.instrumentation.trace.WindowInfo
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Phase D2 on SQLite: the Layer 2 state commits in the same transaction as the trace batch, is
 * restored exactly by a new connection, and a Phase B (version 1) database is upgraded without
 * touching its trace. Uses its own database file, never the real trace.
 */
@RunWith(AndroidJUnit4::class)
class Layer2SqliteStoreTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val dbName = "layer2_store_test.db"
    private lateinit var vocabulary: Vocabulary
    private lateinit var pipeline: TracePipeline
    private lateinit var layer1: Layer1Model

    private val timeline = listOf(
        resumed(1_000, "com.whatsapp"), resumed(2_000, "com.instagram.android"),
        resumed(21_000, "com.google.android.youtube"), resumed(22_000, "com.whatsapp"),
    )
    private val source = UsageEventSource { begin, end -> timeline.filter { it.timestampMs in begin until end } }
    private var now = 0L

    @Before
    fun setUp() {
        context.deleteDatabase(dbName)
        vocabulary = Vocabulary.parse(context.assets.open("lsapp_vocabulary.json").use { String(it.readBytes()) })
        val mapping = PackageMapping.parse(
            context.assets.open("package_mapping.tsv").use { it.readBytes() },
            vocabulary,
            LaunchClassifier.SYSTEM_EXCLUSIONS + context.packageName,
        )
        pipeline = TracePipeline(LaunchClassifier(mapping))
        layer1 = Layer1Assets.load(context)
    }

    @After
    fun tearDown() {
        context.deleteDatabase(dbName)
    }

    private fun session(store: SqliteTraceStore): TraceSession =
        TraceSession(PersonalizedTraceStore(store, LivePersonalizer.open(store, layer1, vocabulary.appsSha256)), source, pipeline, { now }, 2_000)

    @Test
    fun layer2StateIsRestoredExactlyByANewConnection() {
        val first = SqliteTraceStore(context, dbName)
        session(first).apply {
            now = 500; openWindow(window(500))
            now = 5_000; poll { true }
        }
        val committed = Layer2StateCodec.encode(first.loadLayer2()!!)
        first.close() // the process dies without closing the window

        val second = SqliteTraceStore(context, dbName)
        val personalizer = LivePersonalizer.open(second, layer1, vocabulary.appsSha256)
        assertArrayEquals(committed, Layer2StateCodec.encode(personalizer.state()))
        assertEquals(1L, personalizer.updateCount)
        assertEquals(1, personalizer.pendingPrediction!!.position)

        TraceSession(PersonalizedTraceStore(second, personalizer), source, pipeline, { now }, 2_000).apply {
            recoverUnclosedWindow()
            now = 20_000; openWindow(window(20_000))
            now = 25_000; poll { true }
        }
        assertEquals(3L, second.loadLayer2()!!.updateCount)
        val last = second.lastLayer2Log()!!
        assertEquals(3, last.sequencePosition)
        assertEquals(2, last.reveal!!.predictionPosition)
        second.close()
    }

    @Test
    fun aFailedLayer2WriteRollsBackTheWholeBatch() {
        val store = SqliteTraceStore(context, dbName)
        session(store).apply {
            now = 500; openWindow(window(500))
            now = 5_000; poll { true }
        }
        val stateBefore = Layer2StateCodec.encode(store.loadLayer2()!!)
        val traceBefore = store.loadState()
        val eventsBefore = store.snapshot().records.size

        // A batch whose Layer 2 log repeats an existing launch position: the layer2_log key rejects it.
        val record = TraceRecord(
            1, 6_000, resumed(3_500, "com.google.android.youtube"), Outcome.APPENDED, Classification.SUPPORTED,
            vocabulary.idOf("YouTube"), "YouTube", 2, false,
        )
        val duplicate = Layer2Commit(store.loadLayer2()!!, listOf(store.lastLayer2Log()!!))
        val failure = runCatching { store.commitBatch(listOf(record), 4_000, 6_000, duplicate) }.exceptionOrNull()
        assertTrue(failure is SQLiteConstraintException)

        assertEquals(eventsBefore, store.snapshot().records.size)
        assertEquals(traceBefore, store.loadState())
        assertArrayEquals(stateBefore, Layer2StateCodec.encode(store.loadLayer2()!!))
        store.close()
    }

    @Test
    fun phaseBDatabaseIsUpgradedAndARecordedSequenceIsNotTakenOverByLayer2() {
        // A version 1 (Phase B) database holding one appended launch.
        val file = context.getDatabasePath(dbName).also { it.parentFile?.mkdirs() }
        SQLiteDatabase.openOrCreateDatabase(file, null).use { db ->
            PHASE_B_SCHEMA.forEach(db::execSQL)
            db.execSQL(
                "INSERT INTO windows (start_ms, time_zone, self_package, ime_packages, profile_count, mapping_sha256, " +
                    "vocabulary_sha256, api_level, device) VALUES (500, 'UTC', 'self', '', 1, 'm', 'v', 36, 'test')"
            )
            db.execSQL(
                "INSERT INTO events (window_id, observed_at_ms, timestamp_ms, event_type, package, outcome, classification, " +
                    "app_id, lsapp_name, sequence_position, timestamp_anomaly) VALUES " +
                    "(1, 5000, 1000, 1, 'com.whatsapp', 'APPENDED', 'SUPPORTED', ${vocabulary.idOf("WhatsApp Messenger")}, 'WhatsApp Messenger', 0, 0)"
            )
            db.version = 1
        }
        val store = SqliteTraceStore(context, dbName)
        assertEquals(1, store.loadState().trace.nextPosition) // the trace is untouched
        assertNull(store.loadLayer2())
        val refused = runCatching { LivePersonalizer.open(store, layer1, vocabulary.appsSha256) }.exceptionOrNull()
        assertTrue(refused is IllegalStateException)
        store.close()
    }

    private fun resumed(ts: Long, pkg: String) = RawUsageEvent(ts, UsageEventTypes.ACTIVITY_RESUMED, pkg, "$pkg.Main")

    private fun window(start: Long) = WindowInfo(
        start, "UTC", EnvironmentSnapshot(context.packageName, "com.example.launcher", emptySet()),
        1, "test", vocabulary.appsSha256, 36, "test",
    )

    private companion object {
        /** SqliteTraceStore.onCreate as of schema version 1 (commit d509fb4). */
        val PHASE_B_SCHEMA = listOf(
            """CREATE TABLE windows (
                id INTEGER PRIMARY KEY AUTOINCREMENT, start_ms INTEGER NOT NULL, end_ms INTEGER, close_reason TEXT,
                time_zone TEXT NOT NULL, self_package TEXT NOT NULL, home_package TEXT, ime_packages TEXT NOT NULL,
                profile_count INTEGER NOT NULL, mapping_sha256 TEXT NOT NULL, vocabulary_sha256 TEXT NOT NULL,
                api_level INTEGER NOT NULL, device TEXT NOT NULL)""",
            """CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, window_id INTEGER NOT NULL REFERENCES windows(id),
                observed_at_ms INTEGER NOT NULL, timestamp_ms INTEGER NOT NULL, event_type INTEGER NOT NULL,
                package TEXT NOT NULL, class_name TEXT, outcome TEXT NOT NULL, classification TEXT, app_id INTEGER,
                lsapp_name TEXT, sequence_position INTEGER, timestamp_anomaly INTEGER NOT NULL)""",
            "CREATE UNIQUE INDEX events_sequence ON events(sequence_position) WHERE outcome = 'APPENDED'",
            "CREATE TABLE state (key TEXT PRIMARY KEY, value INTEGER NOT NULL)",
        )
    }
}
