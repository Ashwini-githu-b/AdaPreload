package com.adapreload.instrumentation.collect

import android.app.usage.UsageEvents
import android.app.usage.UsageStatsManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.UserManager
import android.view.inputmethod.InputMethodManager
import com.adapreload.instrumentation.trace.EnvironmentSnapshot
import com.adapreload.instrumentation.trace.RawUsageEvent
import com.adapreload.instrumentation.trace.UsageEventSource
import com.adapreload.instrumentation.trace.WindowInfo
import java.util.TimeZone

/** Reads raw events with UsageStatsManager.queryEvents, in the order the platform returns them. */
class UsageStatsEventSource(context: Context) : UsageEventSource {
    private val usageStats = context.getSystemService(Context.USAGE_STATS_SERVICE) as UsageStatsManager

    override fun query(beginMs: Long, endMs: Long): List<RawUsageEvent> {
        // queryEvents covers [beginMs, endMs) and only returns the calling user's events (spec A8).
        val events = usageStats.queryEvents(beginMs, endMs) ?: return emptyList()
        val out = ArrayList<RawUsageEvent>()
        val event = UsageEvents.Event()
        while (events.hasNextEvent()) {
            events.getNextEvent(event)
            out += RawUsageEvent(event.timeStamp, event.eventType, event.packageName.orEmpty(), event.className)
        }
        return out
    }
}

/** Device facts used for classification (spec A4) and observation eligibility (A8). */
class AndroidEnvironment(private val context: Context) {
    private val packageManager = context.packageManager
    private val userManager = context.getSystemService(Context.USER_SERVICE) as UserManager
    private val launcherEntries = HashMap<String, Boolean>()

    /** Observation runs only in the primary user's personal profile (A8). */
    val isPrimaryUser: Boolean get() = userManager.isSystemUser

    /** Usage events cannot be read until the user unlocks after boot; queryEvents then returns nothing. */
    val isUserUnlocked: Boolean get() = userManager.isUserUnlocked

    fun windowInfo(startMs: Long, mappingSha256: String, vocabularySha256: String): WindowInfo {
        launcherEntries.clear()
        return WindowInfo(
            startMs = startMs,
            timeZoneId = TimeZone.getDefault().id,
            environment = EnvironmentSnapshot(context.packageName, defaultHomePackage(), enabledImePackages()),
            profileCount = userManager.userProfiles.size,
            mappingSha256 = mappingSha256,
            vocabularySha256 = vocabularySha256,
            apiLevel = Build.VERSION.SDK_INT,
            device = "${Build.MANUFACTURER} ${Build.MODEL}",
        )
    }

    /** Cached per window; only labels discards (mapped packages are classified before this test). */
    fun hasLauncherEntry(packageName: String): Boolean =
        launcherEntries.getOrPut(packageName) { packageManager.getLaunchIntentForPackage(packageName) != null }

    /**
     * The current default home app only. Not every HOME-capable app: Settings declares a fallback
     * HOME activity. With no default chosen this resolves to the system chooser ("android").
     */
    @Suppress("DEPRECATION")
    private fun defaultHomePackage(): String? =
        packageManager.resolveActivity(
            Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_HOME),
            PackageManager.MATCH_DEFAULT_ONLY,
        )?.activityInfo?.packageName

    private fun enabledImePackages(): Set<String> =
        (context.getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager)
            .enabledInputMethodList.map { it.packageName }.toSet()
}
