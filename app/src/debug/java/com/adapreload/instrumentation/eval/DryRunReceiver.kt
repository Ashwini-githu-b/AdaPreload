package com.adapreload.instrumentation.eval

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log

/**
 * DEBUG-ONLY adb trigger for the on-device dry-run export, so a researcher can reproduce the
 * off-device JVM replay headlessly instead of tapping the UI. It only reads already-persisted
 * Phase D2/E1 data and writes one report file; it runs no model, makes no Layer 2 update, and
 * opens no binding. Present only in debug builds.
 *
 *   adb shell am broadcast -n com.adapreload.instrumentation/.eval.DryRunReceiver \
 *       -a com.adapreload.instrumentation.DRYRUN_EXPORT
 */
class DryRunReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != ACTION_EXPORT) {
            Log.w(TAG, "unknown_action=${intent.action}")
            return
        }
        try {
            val result = DryRunEvaluator.export(context.applicationContext)
            Log.i(TAG, "dry-run export written: ${result.file.absolutePath}")
            result.summary.forEach { Log.i(TAG, it) }
        } catch (e: Exception) {
            Log.e(TAG, "dry-run export failed", e)
        }
    }

    companion object {
        const val ACTION_EXPORT = "com.adapreload.instrumentation.DRYRUN_EXPORT"
        private const val TAG = "AdaPreloadDryRun"
    }
}
