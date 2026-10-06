package com.adapreload.instrumentation.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.adapreload.instrumentation.R
import com.adapreload.instrumentation.collect.TraceUiStatus
import com.adapreload.instrumentation.trace.TraceCounts
import com.adapreload.instrumentation.ui.theme.AdapreloadTheme

/** What the notification row offers when notifications are not enabled. */
enum class NotificationAction { NONE, REQUEST_PERMISSION, OPEN_SETTINGS }

data class SetupStatus(
    val usageAccessGranted: Boolean = false,
    val notificationsEnabled: Boolean = false,
    val notificationAction: NotificationAction = NotificationAction.NONE,
)

@Composable
fun SetupScreen(
    androidVersion: String,
    status: SetupStatus,
    serviceRunning: Boolean,
    onOpenUsageAccessSettings: () -> Unit,
    onRequestNotificationPermission: () -> Unit,
    onOpenNotificationSettings: () -> Unit,
    onStartService: () -> Unit,
    onStopService: () -> Unit,
    trace: TraceUiStatus,
    onExportTrace: () -> Unit,
    onScanInventory: () -> Unit,
    onExportInventory: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text(
            text = stringResource(R.string.app_name),
            style = MaterialTheme.typography.headlineMedium
        )
        Text(
            text = androidVersion,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        StatusCard(
            title = stringResource(R.string.usage_access_title),
            state = stringResource(
                if (status.usageAccessGranted) R.string.usage_access_granted
                else R.string.usage_access_not_granted
            ),
            needsAttention = !status.usageAccessGranted,
            detail = stringResource(R.string.usage_access_detail)
        ) {
            // Shown in both states so access can also be revoked from here while testing.
            OutlinedButton(onClick = onOpenUsageAccessSettings) {
                Text(stringResource(R.string.usage_access_action))
            }
        }

        StatusCard(
            title = stringResource(R.string.notifications_title),
            state = stringResource(
                if (status.notificationsEnabled) R.string.notifications_enabled
                else R.string.notifications_disabled
            ),
            needsAttention = !status.notificationsEnabled,
            detail = stringResource(R.string.notifications_detail)
        ) {
            when (status.notificationAction) {
                NotificationAction.NONE -> Unit
                NotificationAction.REQUEST_PERMISSION ->
                    OutlinedButton(onClick = onRequestNotificationPermission) {
                        Text(stringResource(R.string.notifications_allow))
                    }
                NotificationAction.OPEN_SETTINGS ->
                    OutlinedButton(onClick = onOpenNotificationSettings) {
                        Text(stringResource(R.string.notifications_open_settings))
                    }
            }
        }

        StatusCard(
            title = stringResource(R.string.service_title),
            state = stringResource(
                if (serviceRunning) R.string.service_running else R.string.service_stopped
            ),
            needsAttention = false,
            detail = stringResource(R.string.service_detail)
        ) {
            if (serviceRunning) {
                OutlinedButton(onClick = onStopService) {
                    Text(stringResource(R.string.service_stop))
                }
            } else {
                Button(onClick = onStartService) {
                    Text(stringResource(R.string.service_start))
                }
            }
        }

        TraceCard(
            trace = trace,
            onExport = onExportTrace,
            onScanInventory = onScanInventory,
            onExportInventory = onExportInventory,
        )

        Text(
            text = stringResource(R.string.scope_note),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
private fun StatusCard(
    title: String,
    state: String,
    needsAttention: Boolean,
    detail: String,
    action: @Composable () -> Unit
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(text = title, style = MaterialTheme.typography.titleMedium)
                Text(
                    text = state,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = if (needsAttention) MaterialTheme.colorScheme.error
                    else MaterialTheme.colorScheme.primary
                )
            }
            Text(
                text = detail,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            action()
        }
    }
}

@Preview(showBackground = true)
@Composable
private fun SetupScreenPreview() {
    AdapreloadTheme {
        SetupScreen(
            androidVersion = "Android 16 (API 36)",
            status = SetupStatus(
                usageAccessGranted = false,
                notificationsEnabled = false,
                notificationAction = NotificationAction.REQUEST_PERMISSION
            ),
            serviceRunning = false,
            onOpenUsageAccessSettings = {},
            onRequestNotificationPermission = {},
            onOpenNotificationSettings = {},
            onStartService = {},
            onStopService = {},
            trace = TraceUiStatus(
                observing = true,
                counts = TraceCounts(observedEvents = 120, launchCandidates = 30, supportedLaunches = 12, sequenceLength = 9),
                mappedPackages = 30,
                ambiguousPackages = 39,
            ),
            onExportTrace = {},
            onScanInventory = {},
            onExportInventory = {},
        )
    }
}
