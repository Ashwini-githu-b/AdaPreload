package com.adapreload.instrumentation

import android.Manifest
import android.content.ActivityNotFoundException
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import com.adapreload.instrumentation.collect.TraceRuntime
import com.adapreload.instrumentation.permission.NotificationAccess
import com.adapreload.instrumentation.permission.UsageAccess
import com.adapreload.instrumentation.service.AdaPreloadForegroundService
import com.adapreload.instrumentation.ui.NotificationAction
import com.adapreload.instrumentation.ui.SetupScreen
import com.adapreload.instrumentation.ui.SetupStatus
import com.adapreload.instrumentation.ui.theme.AdapreloadTheme
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : ComponentActivity() {

    private var status by mutableStateOf(SetupStatus())

    // After a denial the system may stop showing the dialog, so offer Settings instead.
    private var notificationRequestDenied = false

    private val notificationPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            notificationRequestDenied = !granted
            refreshStatus()
        }

    // The system file picker chooses where the export goes; no storage permission is needed.
    private val exportLauncher =
        registerForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { uri ->
            if (uri != null) TraceRuntime.export(this, uri)
        }

    // Phase E2a: a separate picker for the read-only service inventory report.
    private val exportInventoryLauncher =
        registerForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { uri ->
            if (uri != null) TraceRuntime.exportInventory(this, uri)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            AdapreloadTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    SetupScreen(
                        androidVersion = stringResource(
                            R.string.android_version,
                            Build.VERSION.RELEASE,
                            Build.VERSION.SDK_INT
                        ),
                        status = status,
                        serviceRunning = AdaPreloadForegroundService.isRunning,
                        onOpenUsageAccessSettings = { openSettings(UsageAccess.settingsIntent()) },
                        onRequestNotificationPermission = { requestNotificationPermission() },
                        onOpenNotificationSettings = {
                            openSettings(NotificationAccess.settingsIntent(this@MainActivity))
                        },
                        onStartService = { AdaPreloadForegroundService.start(this@MainActivity) },
                        onStopService = { AdaPreloadForegroundService.stop(this@MainActivity) },
                        trace = TraceRuntime.status,
                        onExportTrace = { exportLauncher.launch(exportFileName()) },
                        onScanInventory = { TraceRuntime.inventoryServices(this@MainActivity) },
                        onExportInventory = { exportInventoryLauncher.launch(inventoryFileName()) },
                        modifier = Modifier.padding(innerPadding)
                    )
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        // Usage access and notification settings change outside the app; re-read them on return.
        refreshStatus()
        TraceRuntime.refresh(this)
    }

    private fun exportFileName(): String =
        "adapreload_trace_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date()) + ".json"

    private fun inventoryFileName(): String =
        "adapreload_service_inventory_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date()) + ".json"

    private fun refreshStatus() {
        val notificationsEnabled = NotificationAccess.areEnabled(this)
        status = SetupStatus(
            usageAccessGranted = UsageAccess.isGranted(this),
            notificationsEnabled = notificationsEnabled,
            notificationAction = when {
                notificationsEnabled -> NotificationAction.NONE
                NotificationAccess.canRequestPermission(this) && !notificationRequestDenied ->
                    NotificationAction.REQUEST_PERMISSION
                else -> NotificationAction.OPEN_SETTINGS
            }
        )
    }

    private fun requestNotificationPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
    }

    private fun openSettings(intent: Intent) {
        try {
            startActivity(intent)
        } catch (e: ActivityNotFoundException) {
            // Some builds don't expose the specific settings screen; fall back to the main one.
            startActivity(Intent(Settings.ACTION_SETTINGS))
        }
    }
}
