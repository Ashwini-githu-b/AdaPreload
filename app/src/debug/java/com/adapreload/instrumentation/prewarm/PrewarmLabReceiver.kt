package com.adapreload.instrumentation.prewarm

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.SystemClock
import android.util.Log

/**
 * DEBUG-ONLY broadcast trigger for the E2b prewarm lab, so a researcher can drive bind/unbind from
 * adb with an exact, repeatable delay instead of tapping the UI. It only ever binds the single
 * approved candidate (Brave) or unbinds; it cannot bind anything else and does nothing the UI
 * harness does not. Present only in debug builds.
 *
 *   adb shell am broadcast -n com.adapreload.instrumentation/.prewarm.PrewarmLabReceiver \
 *       -a com.adapreload.instrumentation.PREWARM_BIND --es mode bind_only|warmup
 *   adb shell am broadcast -n com.adapreload.instrumentation/.prewarm.PrewarmLabReceiver \
 *       -a com.adapreload.instrumentation.PREWARM_UNBIND
 */
class PrewarmLabReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val lab = PrewarmLabHolder.get(context)
        val trial = SystemClock.elapsedRealtime()
        when (intent.action) {
            ACTION_BIND -> when (intent.getStringExtra("mode")) {
                "warmup" -> lab.bindBraveWarmup(trial)          // Brave only; documented warmup(0)
                else -> lab.bind(PrewarmCandidate.BRAVE_CUSTOMTABS, trial) // bind-only, the only approved candidate
            }
            ACTION_UNBIND -> lab.unbind(trial)
            else -> Log.w(PREWARM_TAG, prewarmLogLine(trial, null, "unknown_action=${intent.action}"))
        }
    }

    companion object {
        const val ACTION_BIND = "com.adapreload.instrumentation.PREWARM_BIND"
        const val ACTION_UNBIND = "com.adapreload.instrumentation.PREWARM_UNBIND"
    }
}
