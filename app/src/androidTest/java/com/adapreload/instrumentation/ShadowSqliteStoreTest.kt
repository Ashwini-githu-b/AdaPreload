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
import com.adapreload.instrumentation.shadow.CandidateResolution
import com.adapreload.instrumentation.shadow.CandidateResolver
import com.adapreload.instrumentation.shadow.Eligibility
import com.adapreload.instrumentation.shadow.ShadowPolicyConfig
import com.adapreload.instrumentation.shadow.ShadowPreloadPolicy
import com.adapreload.instrumentation.trace.EnvironmentSnapshot
import com.adapreload.instrumentation.trace.LaunchClassifier
import com.adapreload.instrumentation.trace.Outcome
import com.adapreload.instrumentation.trace.PackageMapping
import com.adapreload.instrumentation.trace.RawUsageEvent
import com.adapreload.instrumentation.trace.TracePipeline
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
 * Phase E1 on SQLite: shadow decisions commit and roll back with their batch, are resolved by the
 * next launch, and a version 2 (Phase D2) database is upgraded without changing its data.
 * Uses its own database file, never the real trace.
 */
@RunWith(AndroidJUnit4::class)
class ShadowSqliteStoreTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val dbName = "shadow_store_test.db"
    private lateinit var vocabulary: Vocabulary
    private lateinit var pipeline: TracePipeline
    private lateinit var layer1: Layer1Model

    private val timeline = listOf(
        resumed(1_000, "com.whatsapp"), resumed(2_000, "com.instagram.android"),
        resumed(21_000, "com.google.android.youtube"), resumed(22_000, "com.whatsapp"),
    )
    private val source = UsageEventSource { begin, end -> timeline.filter { it.timestampMs in begin until end } }
    private val policy = ShadowPreloadPolicy(
        ShadowPolicyConfig(minProbability = 0.0),
        CandidateResolver { CandidateResolution(Eligibility.ELIGIBLE, "pkg.$it") },
    )
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

    private fun session(store: SqliteTraceStore) = TraceSession(
        PersonalizedTraceStore(store, LivePersonalizer.open(store, layer1, vocabulary.appsSha256), shadowPolicy = policy),
        source, pipeline, { now }, 2_000,
    )

    @Test
    fun decisionsArePersistedAndResolvedAcrossBatchesAndConnections() {
        val first = SqliteTraceStore(context, dbName)
        session(first).apply {
            now = 500; openWindow(window(500))
            now = 5_000; poll { true } // positions 0, 1
        }
        first.close()
        val second = SqliteTraceStore(context, dbName)
        session(second).apply {
            recoverUnclosedWindow()
            now = 20_000; openWindow(window(20_000))
            now = 25_000; poll { true } // positions 2, 3
        }
        val rows = second.shadowDecisions()
        assertEquals(listOf(0, 1, 2, 3), rows.map { it.decision.predictionPosition })
        val seq = second.snapshot().records.filter { it.record.outcome == Outcome.APPENDED }.map { it.record.appId!! }
        assertEquals(seq.drop(1), rows.dropLast(1).map { it.actualAppId })
        assertNull(rows.last().actualAppId)
        assertEquals(listOf(5_000L, 5_000L, 25_000L, 25_000L), rows.map { it.decidedAtMs })
        second.close()
    }

    @Test
    fun aDuplicateDecisionRollsBackTheWholeBatch() {
        val store = SqliteTraceStore(context, dbName)
        session(store).apply {
            now = 500; openWindow(window(500))
            now = 5_000; poll { true }
        }
        val stateBefore = Layer2StateCodec.encode(store.loadLayer2()!!)
        val traceBefore = store.loadState()
        val eventsBefore = store.snapshot().records.size
        val decisionsBefore = store.shadowDecisions()

        val duplicate = decisionsBefore.first().decision
        val failure = runCatching {
            store.commitBatch(emptyList(), 4_000, 6_000, Layer2Commit(store.loadLayer2()!!, emptyList()), listOf(duplicate))
        }.exceptionOrNull()
        assertTrue(failure is SQLiteConstraintException)

        assertEquals(eventsBefore, store.snapshot().records.size)
        assertEquals(traceBefore, store.loadState())
        assertArrayEquals(stateBefore, Layer2StateCodec.encode(store.loadLayer2()!!))
        assertEquals(decisionsBefore, store.shadowDecisions())
        store.close()
    }

    @Test
    fun aPhaseD2DatabaseIsUpgradedWithItsDataUnchanged() {
        // Build a version 2 database: version 3 is version 2 plus the shadow_decisions table.
        val d2 = SqliteTraceStore(context, dbName)
        TraceSession(PersonalizedTraceStore(d2, LivePersonalizer.open(d2, layer1, vocabulary.appsSha256)), source, pipeline, { now }, 2_000).apply {
            now = 500; openWindow(window(500))
            now = 5_000; poll { true }
        }
        val state = Layer2StateCodec.encode(d2.loadLayer2()!!)
        val log = d2.lastLayer2Log()
        val snapshot = d2.snapshot()
        d2.close()
        SQLiteDatabase.openDatabase(context.getDatabasePath(dbName).path, null, SQLiteDatabase.OPEN_READWRITE).use { db ->
            db.execSQL("DROP TABLE shadow_decisions")
            db.version = 2
        }

        val upgraded = SqliteTraceStore(context, dbName)
        assertArrayEquals(state, Layer2StateCodec.encode(upgraded.loadLayer2()!!))
        assertEquals(log, upgraded.lastLayer2Log())
        assertEquals(snapshot, upgraded.snapshot())
        assertTrue(upgraded.shadowDecisions().isEmpty())
        // Layer 2 resumes exactly, and decisions start with the next prediction.
        val personalizer = LivePersonalizer.open(upgraded, layer1, vocabulary.appsSha256)
        assertEquals(1L, personalizer.updateCount)
        upgraded.close()
    }

    private fun resumed(ts: Long, pkg: String) = RawUsageEvent(ts, UsageEventTypes.ACTIVITY_RESUMED, pkg, "$pkg.Main")

    private fun window(start: Long) = WindowInfo(
        start, "UTC", EnvironmentSnapshot(context.packageName, "com.example.launcher", emptySet()),
        1, "test", vocabulary.appsSha256, 36, "test",
    )
}
