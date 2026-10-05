package com.adapreload.instrumentation.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.app.NotificationCompat
import com.adapreload.instrumentation.MainActivity
import com.adapreload.instrumentation.R
import com.adapreload.instrumentation.collect.TraceRuntime

/**
 * Long-running foreground service that hosts AdaPreload's on-device work.
 *
 * Phase B: while running, it observes this device's app launches through UsageEvents and
 * records the launch trace ([TraceRuntime]). It makes no predictions and launches or warms no
 * other app.
 */
class AdaPreloadForegroundService : Service() {

    override fun onCreate() {
        super.onCreate()
        isRunning = true
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (!enterForeground()) {
            stopSelf()
            return START_NOT_STICKY
        }
        // Idempotent: repeated start commands keep the existing collection running.
        TraceRuntime.startCollection(this)
        return START_STICKY
    }

    override fun onDestroy() {
        // Polls once more, then closes the observation window (A7).
        TraceRuntime.stopCollection(this)
        isRunning = false
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun enterForeground(): Boolean {
        createNotificationChannel()
        val notification = buildNotification()
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                startForeground(
                    NOTIFICATION_ID,
                    notification,
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE
                )
            } else {
                startForeground(NOTIFICATION_ID, notification)
            }
            true
        } catch (e: IllegalStateException) {
            // Android 12+ throws ForegroundServiceStartNotAllowedException (an IllegalStateException)
            // when the foreground start isn't allowed, e.g. a sticky restart while the app is in the
            // background. Stop instead of crashing; the user can start the service again from the app.
            Log.w(TAG, "Foreground start not allowed; stopping service", e)
            false
        }
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            getString(R.string.service_channel_name),
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = getString(R.string.service_channel_description)
        }
        (getSystemService(NOTIFICATION_SERVICE) as NotificationManager)
            .createNotificationChannel(channel)
    }

    private fun buildNotification(): Notification {
        val openApp = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(getString(R.string.service_notification_title))
            .setContentText(getString(R.string.service_notification_text))
            .setContentIntent(openApp)
            .setOngoing(true)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            // Android 12+ otherwise may delay showing a foreground-service notification by ~10 s.
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .build()
    }

    companion object {
        const val CHANNEL_ID = "adapreload_service"
        private const val NOTIFICATION_ID = 1
        private const val TAG = "AdaPreloadService"

        /** Whether a service instance is alive in this process; observable from Compose. */
        var isRunning by mutableStateOf(false)
            private set

        fun start(context: Context) {
            context.startForegroundService(Intent(context, AdaPreloadForegroundService::class.java))
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, AdaPreloadForegroundService::class.java))
        }
    }
}
