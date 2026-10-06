package com.adapreload.instrumentation.collect

import android.annotation.SuppressLint
import android.content.Context
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.adapreload.instrumentation.live.Layer2LogEntry
import com.adapreload.instrumentation.live.LivePersonalizer
import com.adapreload.instrumentation.live.PersonalizedTraceStore
import com.adapreload.instrumentation.live.Reveal
import com.adapreload.instrumentation.inventory.AndroidPackageServiceSource
import com.adapreload.instrumentation.inventory.InventoryReport
import com.adapreload.instrumentation.inventory.ServiceInventoryResult
import com.adapreload.instrumentation.inventory.ServiceWarmabilityInventory
import com.adapreload.instrumentation.model.Layer1Assets
import com.adapreload.instrumentation.permission.UsageAccess
import com.adapreload.instrumentation.shadow.MappingCandidateResolver
import com.adapreload.instrumentation.shadow.ShadowDecision
import com.adapreload.instrumentation.shadow.ShadowPolicyConfig
import com.adapreload.instrumentation.shadow.ShadowPreloadDecision
import com.adapreload.instrumentation.shadow.ShadowPreloadPolicy
import com.adapreload.instrumentation.trace.CloseReason
import com.adapreload.instrumentation.trace.EnvironmentSnapshot
import com.adapreload.instrumentation.trace.LaunchClassifier
import com.adapreload.instrumentation.trace.MappingStatus
import com.adapreload.instrumentation.trace.PackageMapping
import com.adapreload.instrumentation.trace.TraceCounts
import com.adapreload.instrumentation.trace.TraceExporter
import com.adapreload.instrumentation.trace.TracePipeline
import com.adapreload.instrumentation.trace.TraceSession
import com.adapreload.instrumentation.trace.Vocabulary
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit

/** What the status screen shows about trace collection. */
data class TraceUiStatus(
    val observing: Boolean = false,
    val counts: TraceCounts = TraceCounts(),
    val mappedPackages: Int = 0,
    val ambiguousPackages: Int = 0,
    /** Latest notice: why observation is paused, an error, or an export result. */
    val message: String? = null,
    /** Layer 2 diagnostics; null until Layer 2 has been opened in this process. */
    val layer2: Layer2UiStatus? = null,
    /** Phase E1 shadow preload policy; null until Layer 2 has been opened in this process. */
    val shadow: ShadowUiStatus? = null,
    /** Phase E2a service warmability inventory; null until a read-only scan is run. */
    val inventory: InventoryUiStatus? = null,
)

/** Phase E2a diagnostics. A read-only count; no service is bound or started. */
data class InventoryUiStatus(
    val summary: String,
    val warmableApps: Int,
    val installedApps: Int,
    /** True once a report has been built this process, so it can be exported. */
    val reportAvailable: Boolean,
)

/** Phase E1 diagnostics. A PRELOAD decision is only recorded: nothing is ever preloaded. */
data class ShadowUiStatus(
    val config: ShadowPolicyConfig,
    /** The latest prediction's PRELOAD decision, or else its top-ranked decision. */
    val last: ShadowPreloadDecision?,
    val lastAppName: String?,
)

/** Minimal Layer 2 diagnostics for verifying the live pipeline. */
data class Layer2UiStatus(
    val updateCount: Long,
    /** Sequence position of the latest launch Layer 2 processed. */
    val lastPosition: Int?,
    /** What that launch revealed; null for the first launch. */
    val lastReveal: Reveal?,
    /** Top-1 of the pending prediction (the next app), Layer 1 alone and with the adapter. */
    val nextLayer1Top1: String?,
    val nextLayer2Top1: String?,
)

/**
 * Process-wide trace collection with live Layer 2 personalization (Phase D2). Polling, status
 * refreshes and exports all run on one background thread, so the store, the session and the
 * learner are only ever touched from that thread.
 *
 * Every batch is committed together with its Layer 2 work ([PersonalizedTraceStore]). If Layer 2
 * cannot be opened, nothing is observed: the trace never advances without the adapter.
 */
object TraceRuntime {
    private const val TAG = "AdaPreloadTrace"
    private const val TAG_LAYER2 = "AdaPreloadLayer2"
    private const val TAG_SHADOW = "AdaPreloadShadow"
    private const val TAG_INVENTORY = "AdaPreloadInventory"
    private const val POLL_INTERVAL_MS = 5_000L
    private const val VOCABULARY_ASSET = "lsapp_vocabulary.json"
    private const val MAPPING_ASSET = "package_mapping.tsv"

    /** Phase E1 policy; see docs/PHASE_E1_SHADOW_PRELOAD.md. */
    private val SHADOW_POLICY = ShadowPolicyConfig.DEFAULT

    private val executor = Executors.newSingleThreadScheduledExecutor { Thread(it, "adapreload-trace") }
    private val mainHandler = Handler(Looper.getMainLooper())

    /** Observable from Compose; written on the main thread only. */
    var status by mutableStateOf(TraceUiStatus())
        private set

    private class Config(val vocabulary: Vocabulary, val mapping: PackageMapping)

    // Touched only on the executor thread.
    private var config: Config? = null
    private var store: SqliteTraceStore? = null
    private var liveStore: PersonalizedTraceStore? = null
    // Holds the application context only (see startCollection), which lives as long as the process.
    @SuppressLint("StaticFieldLeak")
    private var environment: AndroidEnvironment? = null
    private var session: TraceSession? = null
    private var pollTask: ScheduledFuture<*>? = null

    /** Why the latest poll did not observe, if it did not. Reset every poll. */
    private var pauseReason: String? = null

    /** Latest export result or configuration error; kept until replaced. */
    private var notice: String? = null

    /** Why Layer 2 could not be opened, and therefore nothing is observed; cleared once it opens. */
    private var layer2Error: String? = null

    // Phase E2a: the latest read-only inventory, held in memory only (no persistence).
    private var inventoryStatus: InventoryUiStatus? = null
    private var lastInventoryJson: String? = null

    /** The environment of the open observation window (A4), used to classify shadow preload candidates. */
    private var windowEnvironment: EnvironmentSnapshot? = null
    private var lastShadow: ShadowPreloadDecision? = null
    private var shadowFailures = 0

    fun startCollection(context: Context) {
        val app = context.applicationContext
        executor.execute { start(app) }
    }

    fun stopCollection(context: Context) {
        val app = context.applicationContext
        executor.execute { stop(app) }
    }

    fun refresh(context: Context) {
        val app = context.applicationContext
        executor.execute { publish(app) }
    }

    fun export(context: Context, uri: Uri) {
        val app = context.applicationContext
        executor.execute { exportTo(app, uri) }
    }

    /** Phase E2a: run the read-only service warmability scan. Reads PackageManager metadata only. */
    fun inventoryServices(context: Context) {
        val app = context.applicationContext
        executor.execute { runInventory(app) }
    }

    /** Phase E2a: write the latest inventory as JSON to [uri] (builds a fresh one if none is held). */
    fun exportInventory(context: Context, uri: Uri) {
        val app = context.applicationContext
        executor.execute { exportInventoryTo(app, uri) }
    }

    private fun start(app: Context) {
        if (pollTask != null) return
        val cfg = config(app) ?: return publish(app)
        val live = liveStore(app, cfg) ?: return publish(app)
        val env = AndroidEnvironment(app).also { environment = it }
        val s = TraceSession(live, UsageStatsEventSource(app), TracePipeline(LaunchClassifier(cfg.mapping)), System::currentTimeMillis)
        s.recoverUnclosedWindow()?.let { Log.i(TAG, "Closed window $it left open by an earlier process") }
        session = s
        pollTask = executor.scheduleWithFixedDelay({ tick(app, s, env, cfg) }, 0, POLL_INTERVAL_MS, TimeUnit.MILLISECONDS)
    }

    private fun tick(app: Context, s: TraceSession, env: AndroidEnvironment, cfg: Config) {
        try {
            pauseReason = pauseReason(app, env)
            if (pauseReason != null) {
                if (s.openWindowId != null && !UsageAccess.isGranted(app)) s.closeWindow(CloseReason.USAGE_ACCESS_LOST)
            } else {
                if (s.openWindowId == null) {
                    val info = env.windowInfo(System.currentTimeMillis(), cfg.mapping.sha256, cfg.vocabulary.appsSha256)
                    s.openWindow(info)
                    windowEnvironment = info.environment
                }
                s.poll(env::hasLauncherEntry)
            }
        } catch (e: Exception) {
            // Keep polling: the cursor only advances after a committed batch, so nothing is lost or duplicated.
            Log.w(TAG, "Poll failed", e)
            pauseReason = "Poll failed: ${e.message}"
        }
        publish(app)
    }

    /** Why observation cannot run right now, or null if it can. */
    private fun pauseReason(app: Context, env: AndroidEnvironment): String? = when {
        !env.isPrimaryUser -> "Not observing: only the primary user's personal profile is supported (A8)."
        !env.isUserUnlocked -> "Not observing: the device has not been unlocked since boot."
        !UsageAccess.isGranted(app) -> "Not observing: Usage Access is not granted."
        else -> null
    }

    private fun stop(app: Context) {
        pollTask?.cancel(false)
        pollTask = null
        val s = session ?: return
        val env = environment
        if (s.openWindowId != null) {
            // Without access queryEvents returns nothing, which would stretch the window over unobserved time.
            if (env != null && pauseReason(app, env) == null) {
                runCatching { s.poll(env::hasLauncherEntry) }.onFailure { Log.w(TAG, "Final poll failed", it) }
            }
            s.closeWindow(CloseReason.SERVICE_STOPPED)
        }
        session = null
        pauseReason = null
        store?.let { publishCounts(it, observing = false) }
    }

    private fun buildInventory(app: Context): ServiceInventoryResult? {
        val cfg = config(app) ?: return null
        // Read-only: AndroidPackageServiceSource only calls PackageManager.getPackageInfo.
        return ServiceWarmabilityInventory.build(cfg.mapping, AndroidPackageServiceSource(app))
    }

    private fun runInventory(app: Context) {
        try {
            val result = buildInventory(app) ?: return publish(app)
            lastInventoryJson = InventoryReport.json(result)
            inventoryStatus = InventoryUiStatus(
                InventoryReport.summary(result), result.warmableAppIds.size, result.installedAppIds.size, reportAvailable = true,
            )
            Log.i(TAG_INVENTORY, InventoryReport.text(result))
            notice = "Service inventory: ${result.warmableAppIds.size} of ${result.installedAppIds.size} " +
                "installed mapped apps have a candidate service (logcat $TAG_INVENTORY)."
        } catch (e: Exception) {
            Log.e(TAG_INVENTORY, "Inventory failed", e)
            notice = "Inventory failed: ${e.message}"
        }
        publish(app)
    }

    private fun exportInventoryTo(app: Context, uri: Uri) {
        notice = try {
            val json = lastInventoryJson ?: InventoryReport.json(buildInventory(app) ?: return publish(app))
            val out = checkNotNull(app.contentResolver.openOutputStream(uri, "wt")) { "Cannot open $uri" }
            out.use { it.write(json.toByteArray(Charsets.US_ASCII)) }
            "Exported service inventory."
        } catch (e: Exception) {
            Log.w(TAG_INVENTORY, "Inventory export failed", e)
            "Inventory export failed: ${e.message}"
        }
        publish(app)
    }

    private fun exportTo(app: Context, uri: Uri) {
        notice = try {
            val cfg = config(app) ?: return publish(app)
            val snapshot = store(app).snapshot()
            val text = TraceExporter.export(snapshot, cfg.vocabulary, cfg.mapping, System.currentTimeMillis())
            val out = checkNotNull(app.contentResolver.openOutputStream(uri, "wt")) { "Cannot open $uri" }
            out.use { it.write(text.toByteArray(Charsets.US_ASCII)) }
            "Exported ${snapshot.records.size} events."
        } catch (e: Exception) {
            Log.w(TAG, "Export failed", e)
            "Export failed: ${e.message}"
        }
        publish(app)
    }

    private fun config(app: Context): Config? {
        config?.let { return it }
        return try {
            val vocabulary = Vocabulary.parse(app.assets.open(VOCABULARY_ASSET).use { String(it.readBytes(), Charsets.UTF_8) })
            val mapping = PackageMapping.parse(
                app.assets.open(MAPPING_ASSET).use { it.readBytes() },
                vocabulary,
                LaunchClassifier.SYSTEM_EXCLUSIONS + app.packageName,
            )
            Config(vocabulary, mapping).also { config = it }
        } catch (e: Exception) {
            // A broken vocabulary or mapping table must never silently change the sequence.
            Log.e(TAG, "Invalid trace configuration", e)
            notice = "Trace configuration invalid: ${e.message}"
            null
        }
    }

    private fun store(app: Context): SqliteTraceStore = store ?: SqliteTraceStore(app).also { store = it }

    /**
     * The store with Layer 2 resumed from it (or started at zero for a new sequence), once per
     * process. Null if the persisted state does not match the model or the trace exactly.
     */
    private fun liveStore(app: Context, cfg: Config): PersonalizedTraceStore? {
        liveStore?.let { return it }
        return try {
            val store = store(app)
            val personalizer = LivePersonalizer.open(store, Layer1Assets.load(app), cfg.vocabulary.appsSha256)
            Log.i(TAG_LAYER2, "Layer 2 opened: ${personalizer.updateCount} updates, pending after launch ${personalizer.pendingPrediction?.position}")
            layer2Error = null
            // Phase E1: decisions only. The resolver queries package metadata; nothing is launched.
            val resolver = MappingCandidateResolver(
                cfg.mapping,
                cfg.vocabulary.appCount,
                environment = { checkNotNull(windowEnvironment) { "No observation window" } },
                hasLauncherEntry = { pkg -> checkNotNull(environment) { "No environment" }.hasLauncherEntry(pkg) },
            )
            PersonalizedTraceStore(
                store,
                personalizer,
                onCommitted = ::logLayer2,
                shadowPolicy = ShadowPreloadPolicy(SHADOW_POLICY, resolver),
                onShadowDecisions = { logShadow(it, cfg.vocabulary) },
                onShadowFailure = { e ->
                    shadowFailures++
                    Log.w(TAG_SHADOW, "Shadow decisions dropped for a batch (Layer 2 work committed)", e)
                },
            ).also { liveStore = it }
        } catch (e: Exception) {
            // Observing without Layer 2 would let the trace and the adapter diverge.
            Log.e(TAG_LAYER2, "Layer 2 cannot start", e)
            layer2Error = "Not observing: Layer 2 cannot start: ${e.message}"
            null
        }
    }

    private fun logLayer2(entries: List<Layer2LogEntry>) {
        for (e in entries) {
            val reveal = e.reveal?.let {
                String.format(Locale.ROOT, "revealed #%d (L1 rank %d, L2 rank %d, loss %.4f)", it.predictionPosition, it.layer1Rank, it.layer2Rank, it.loss)
            } ?: "nothing to reveal"
            Log.i(TAG_LAYER2, "launch #${e.sequencePosition} app ${e.appId}: $reveal; updates ${e.updateCount}; next L1 top-1 ${e.layer1Top1}, L2 top-1 ${e.layer2Top1}")
        }
    }

    private fun logShadow(decisions: List<ShadowPreloadDecision>, vocabulary: Vocabulary) {
        for (d in decisions) {
            val verdict = if (d.decision == ShadowDecision.PRELOAD) "PRELOAD" else "SKIP reason=${d.reasonLabel}"
            Log.i(
                TAG_SHADOW,
                String.format(
                    Locale.ROOT, "prediction #%d (after app %d): rank %d app %d %s [%s] p=%.3f -> %s; actual preload: NONE (shadow mode)",
                    d.predictionPosition, d.currentAppId, d.rank, d.appId, vocabulary.nameOf(d.appId), d.packageName ?: "-",
                    d.probability, verdict,
                ),
            )
        }
        decisions.groupBy { it.predictionPosition }.maxByOrNull { it.key }?.value?.let { last ->
            lastShadow = last.firstOrNull { it.decision == ShadowDecision.PRELOAD } ?: last.first()
        }
    }

    private fun publish(app: Context) {
        publishCounts(store(app), observing = session?.openWindowId != null)
    }

    private fun publishCounts(store: SqliteTraceStore, observing: Boolean) {
        val mapping = config?.mapping
        val vocabulary = config?.vocabulary
        val layer2 = liveStore?.personalizer?.let { p ->
            val last = p.lastLog
            Layer2UiStatus(
                updateCount = p.updateCount,
                lastPosition = last?.sequencePosition,
                lastReveal = last?.reveal,
                nextLayer1Top1 = last?.let { vocabulary?.nameOf(it.layer1Top1) },
                nextLayer2Top1 = last?.let { vocabulary?.nameOf(it.layer2Top1) },
            )
        }
        val shadow = liveStore?.let {
            ShadowUiStatus(SHADOW_POLICY, lastShadow, lastShadow?.let { d -> vocabulary?.nameOf(d.appId) })
        }
        // Batches whose decisions could not be computed (their Layer 2 work was still committed).
        val shadowNotice = if (shadowFailures > 0) "Shadow decisions dropped for $shadowFailures batches (logcat $TAG_SHADOW)." else null
        val next = TraceUiStatus(
            observing = observing,
            counts = store.counts(),
            mappedPackages = mapping?.entries?.count { it.status == MappingStatus.MAPPED } ?: 0,
            ambiguousPackages = mapping?.entries?.count { it.status == MappingStatus.AMBIGUOUS } ?: 0,
            message = listOfNotNull(pauseReason, layer2Error, shadowNotice, notice).joinToString("\n").ifEmpty { null },
            layer2 = layer2,
            shadow = shadow,
            inventory = inventoryStatus,
        )
        mainHandler.post { status = next }
    }
}
