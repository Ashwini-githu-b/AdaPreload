package com.adapreload.instrumentation.prewarm

import android.annotation.SuppressLint
import android.app.Activity
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import android.os.Bundle
import android.os.Process
import android.util.Log
import android.widget.TextView
import java.nio.ByteBuffer
import kotlin.random.Random

/**
 * DEBUG-ONLY memory-pressure generator for the E2b robustness experiment. It runs in a SEPARATE
 * ":stress" process (see the debug manifest) so that releasing it — which kills only its own
 * process — never disturbs the main AdaPreload process that holds the Brave binding.
 *
 * It allocates and TOUCHES `mb` MB of incompressible native memory (direct ByteBuffers, off the
 * Dalvik heap, filled with random bytes so Android's zram cannot cheaply absorb it) to create
 * genuine RAM pressure, holds it until released, then frees it and kills ONLY the :stress process.
 * It is completely separate from the predictor, Layer 1/2, the E1 shadow policy and the prewarm
 * binding, and lives in src/debug so it is never present in a release build.
 *
 *   adb shell am start -n com.adapreload.instrumentation/.prewarm.MemPressureActivity --ei mb 1024
 *   adb shell am broadcast -a com.adapreload.instrumentation.PREWARM_MEM_RELEASE
 */
class MemPressureActivity : Activity() {

    private val held = ArrayList<ByteBuffer>()
    @Volatile private var releasing = false

    private val releaser = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) { release() }
    }

    @SuppressLint("SetTextI18n") // debug-only diagnostic label; never user-facing / localized
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // A visible activity keeps this :stress process at a high adj, so the OS reclaims OTHER
        // (cached) processes — e.g. a prewarmed Brave — rather than this one, under the pressure.
        setContentView(TextView(this).apply { text = "AdaPreload mem pressure (debug)" })

        val filter = IntentFilter(ACTION_MEM_RELEASE)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(releaser, filter, Context.RECEIVER_EXPORTED)
        } else {
            @Suppress("UnspecifiedRegisterReceiverFlag")
            registerReceiver(releaser, filter)
        }

        val requested = intent.getIntExtra("mb", 0).coerceIn(0, MAX_MB) // bounded: hard safety ceiling
        Log.i(PREWARM_TAG, "MEMPRESSURE requested_mb=$requested ceiling_mb=$MAX_MB")

        // Allocate off the main thread so the activity stays responsive (no ANR) during a large fill.
        Thread {
            val rnd = Random(42)
            // The ONLY Java-heap buffer: a single reusable 1 MB array, refreshed and copied
            // repeatedly into each direct chunk. No chunk-sized ByteArray is ever allocated, so
            // the ART heap growth limit is never approached (that was the earlier false OOM).
            val fill = ByteArray(FILL_BYTES)
            var done = 0
            try {
                while (done < requested) {
                    if (releasing) break
                    val buf = ByteBuffer.allocateDirect(CHUNK_BYTES) // native, off the Java heap
                    repeat(CHUNK_BYTES / FILL_BYTES) {
                        rnd.nextBytes(fill)  // refresh => incompressible per page, defeats zram; no new heap
                        buf.put(fill)        // copy into native memory => every page written/resident
                    }
                    synchronized(held) { held.add(buf) }
                    done += CHUNK_MB
                }
            } catch (e: OutOfMemoryError) {
                Log.i(PREWARM_TAG, "MEMPRESSURE oom_at_mb=$done requested=$requested")
            }
            if (!releasing) Log.i(PREWARM_TAG, "MEMPRESSURE ready_mb=$done requested=$requested")
        }.start()
    }

    private fun release() {
        releasing = true
        synchronized(held) { held.clear() }
        Log.i(PREWARM_TAG, "MEMPRESSURE released")
        finish()
        // Kill ONLY this (:stress) process; frees all its memory and cannot touch the main
        // AdaPreload process that holds the Brave binding.
        Process.killProcess(Process.myPid())
    }

    override fun onDestroy() {
        runCatching { unregisterReceiver(releaser) }
        super.onDestroy()
    }

    companion object {
        const val ACTION_MEM_RELEASE = "com.adapreload.instrumentation.PREWARM_MEM_RELEASE"
        private const val CHUNK_MB = 16                        // small direct chunks: robust allocation
        private const val CHUNK_BYTES = CHUNK_MB * 1024 * 1024
        private const val FILL_MB = 1                          // single reusable heap fill buffer
        private const val FILL_BYTES = FILL_MB * 1024 * 1024
        private const val MAX_MB = 8192 // hard ceiling; calibration stays well under device RAM
    }
}
