package com.adapreload.instrumentation.permission

import android.Manifest
import android.app.NotificationManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.provider.Settings
import com.adapreload.instrumentation.service.AdaPreloadForegroundService

object NotificationAccess {

    /**
     * True when the service notification can be shown: notifications are on for the app (which
     * on Android 13+ includes the POST_NOTIFICATIONS grant) and the service channel isn't blocked.
     */
    fun areEnabled(context: Context): Boolean {
        val manager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (!manager.areNotificationsEnabled()) return false
        // The channel is created when the service first starts; until then it can't be blocked.
        val channel = manager.getNotificationChannel(AdaPreloadForegroundService.CHANNEL_ID)
        return channel == null || channel.importance != NotificationManager.IMPORTANCE_NONE
    }

    /** POST_NOTIFICATIONS is a runtime permission only on Android 13 (API 33) and higher. */
    fun canRequestPermission(context: Context): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return false
        return context.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) !=
            PackageManager.PERMISSION_GRANTED
    }

    fun settingsIntent(context: Context): Intent =
        Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS)
            .putExtra(Settings.EXTRA_APP_PACKAGE, context.packageName)
}
