package com.adapreload.instrumentation.prewarm

import android.annotation.SuppressLint
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import android.util.Log
import androidx.browser.customtabs.CustomTabsClient
import androidx.browser.customtabs.CustomTabsServiceConnection

/** androidx.browser CustomTabsService.ACTION_CUSTOM_TABS_CONNECTION (verified from the AndroidX source). */
const val CUSTOM_TABS_CONNECTION_ACTION = "android.support.customtabs.action.CustomTabsService"

const val PREWARM_TAG = "AdaPreloadPrewarm"

/**
 * Phase E2b behavioural-validation candidates. The allowlist is fixed in code; nothing outside it can
 * be bound, and only [approved] candidates may be bound at all. #2..#6 are listed for traceability and
 * are NOT approved — enabling one needs an explicit code change and a fresh review (per the E2a audit).
 */
enum class PrewarmCandidate(
    val display: String,
    val pkg: String,
    val service: String,
    /** The documented intent-filter action, or null when none is known (then bind by explicit component only). */
    val action: String?,
    val approved: Boolean,
) {
    // #1 — the only approved candidate: documented Custom Tabs connection/warm-up contract.
    BRAVE_CUSTOMTABS(
        "Brave", "com.brave.browser",
        "org.chromium.chrome.browser.customtabs.CustomTabsConnectionService",
        CUSTOM_TABS_CONNECTION_ACTION, approved = true,
    ),
    // #2..#6 — listed, NOT approved. action=null: no action is asserted without evidence.
    SNAPCHAT_POSTMESSAGE(
        "Snapchat", "com.snapchat.android",
        "androidx.browser.customtabs.PostMessageService", null, approved = false,
    ),
    DRIVE_DELEGATION(
        "Google Drive", "com.google.android.apps.docs",
        "com.google.androidbrowserhelper.trusted.DelegationService", null, approved = false,
    ),
    PLAY_PREWARM(
        "Google Play Store", "com.android.vending",
        "com.google.android.finsky.prewarmservice.PrewarmService", null, approved = false,
    ),
    SPOTIFY_APPREMOTE(
        "Spotify", "com.spotify.music",
        "com.spotify.interapp.service.service.AppProtocolRemoteService", null, approved = false,
    ),
    PLAY_KEEPALIVE(
        "Google Play Store", "com.android.vending",
        "org.chromium.customtabsclient.shared.KeepAliveService", null, approved = false,
    );

    companion object {
        val approved: List<PrewarmCandidate> get() = entries.filter { it.approved }
    }
}

/** Process-wide single [PrewarmLab] so the debug UI and the debug broadcast trigger share one binding. */
object PrewarmLabHolder {
    // Holds only the application context (via PrewarmLab), which lives as long as the process.
    @SuppressLint("StaticFieldLeak")
    @Volatile private var instance: PrewarmLab? = null

    fun get(context: Context): PrewarmLab =
        instance ?: synchronized(this) { instance ?: PrewarmLab(context.applicationContext).also { instance = it } }
}

/** A structured E2B_VALIDATE log line. Pure, so it is unit-tested. */
fun prewarmLogLine(trial: Long, candidate: PrewarmCandidate?, fields: String): String =
    "E2B_VALIDATE trial=$trial candidate=${candidate?.display ?: "-"} " +
        "pkg=${candidate?.pkg ?: "-"} service=${candidate?.service ?: "-"} " +
        "action=${candidate?.action ?: "none"} $fields"

/**
 * DEBUG-ONLY behavioural-validation harness. Binds exactly ONE allowlisted candidate, so a researcher
 * can observe (via adb: pidof / dumpsys / oom_score_adj) whether binding pre-creates or warms the
 * target process, then unbinds. It is completely separate from the predictor, Layer 1/Layer 2 and the
 * E1 shadow policy, and lives in src/debug so it is never present in a release build.
 *
 * It only ever binds and unbinds, plus the documented Custom Tabs `warmup(0)` for Brave. It NEVER calls
 * mayLaunchUrl, launchUrl, navigation, or any other binder/IPC method, and never starts an activity.
 */
class PrewarmLab(private val context: Context) {
    private var connection: ServiceConnection? = null
    private var bound: PrewarmCandidate? = null

    val isBound: Boolean get() = connection != null

    /** Bind-only: raw bindService with the explicit component and, when known, the documented action. */
    fun bind(candidate: PrewarmCandidate, trial: Long) {
        if (!guard(candidate, trial)) return
        val intent = Intent().setClassName(candidate.pkg, candidate.service)
        candidate.action?.let { intent.action = it } // explicit component + action: API 36 safer-intents
        val conn = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName, binder: IBinder?) {
                Log.i(PREWARM_TAG, prewarmLogLine(trial, candidate, "connected=true binder=${binder != null}"))
            }
            override fun onServiceDisconnected(name: ComponentName) {
                Log.i(PREWARM_TAG, prewarmLogLine(trial, candidate, "disconnected=true"))
            }
            override fun onNullBinding(name: ComponentName) {
                Log.i(PREWARM_TAG, prewarmLogLine(trial, candidate, "null_binding=true"))
            }
        }
        attempt(candidate, trial, warmup = false) { context.bindService(intent, conn, Context.BIND_AUTO_CREATE) to conn }
    }

    /** Brave only: bind via the Custom Tabs client and call the documented warmup(0). No navigation. */
    fun bindBraveWarmup(trial: Long) {
        val candidate = PrewarmCandidate.BRAVE_CUSTOMTABS
        if (!guard(candidate, trial)) return
        val conn = object : CustomTabsServiceConnection() {
            override fun onCustomTabsServiceConnected(name: ComponentName, client: CustomTabsClient) {
                Log.i(PREWARM_TAG, prewarmLogLine(trial, candidate, "connected=true"))
                val warmed = runCatching { client.warmup(0L) } // warmup() only; never mayLaunchUrl/launchUrl
                    .onFailure { Log.i(PREWARM_TAG, prewarmLogLine(trial, candidate, "warmup_exception=${it.javaClass.simpleName}")) }
                    .getOrDefault(false)
                Log.i(PREWARM_TAG, prewarmLogLine(trial, candidate, "warmup=$warmed"))
            }
            override fun onServiceDisconnected(name: ComponentName) {
                Log.i(PREWARM_TAG, prewarmLogLine(trial, candidate, "disconnected=true"))
            }
        }
        attempt(candidate, trial, warmup = true) { CustomTabsClient.bindCustomTabsService(context, candidate.pkg, conn) to conn }
    }

    fun unbind(trial: Long) {
        val conn = connection
        if (conn == null) {
            Log.i(PREWARM_TAG, prewarmLogLine(trial, bound, "unbind_skipped=not_bound"))
            return
        }
        runCatching { context.unbindService(conn) }
            .onFailure { Log.i(PREWARM_TAG, prewarmLogLine(trial, bound, "unbind_exception=${it.javaClass.simpleName}")) }
        Log.i(PREWARM_TAG, prewarmLogLine(trial, bound, "unbound=true"))
        connection = null
        bound = null
    }

    private fun guard(candidate: PrewarmCandidate, trial: Long): Boolean {
        if (!candidate.approved) {
            Log.w(PREWARM_TAG, prewarmLogLine(trial, candidate, "bind_refused=not_approved"))
            return false
        }
        if (connection != null) {
            Log.w(PREWARM_TAG, prewarmLogLine(trial, candidate, "bind_refused=already_bound_to=${bound?.display}"))
            return false
        }
        return true
    }

    private inline fun attempt(candidate: PrewarmCandidate, trial: Long, warmup: Boolean, bindCall: () -> Pair<Boolean, ServiceConnection>) {
        val (ok, conn) = try {
            bindCall()
        } catch (e: SecurityException) {
            Log.i(PREWARM_TAG, prewarmLogLine(trial, candidate, "bind_attempt=true bind_success=false bind_exception=SecurityException warmup=$warmup"))
            return
        } catch (e: IllegalArgumentException) {
            Log.i(PREWARM_TAG, prewarmLogLine(trial, candidate, "bind_attempt=true bind_success=false bind_exception=IllegalArgumentException warmup=$warmup"))
            return
        }
        if (ok) {
            connection = conn
            bound = candidate
        } else {
            // bindService returned false: the system did not bind; release any partial connection.
            runCatching { context.unbindService(conn) }
        }
        Log.i(PREWARM_TAG, prewarmLogLine(trial, candidate, "bind_attempt=true bind_success=$ok warmup=$warmup"))
    }
}
