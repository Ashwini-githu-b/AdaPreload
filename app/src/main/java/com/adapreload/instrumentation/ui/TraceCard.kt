package com.adapreload.instrumentation.ui

import androidx.annotation.StringRes
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.adapreload.instrumentation.R
import com.adapreload.instrumentation.collect.TraceUiStatus
import com.adapreload.instrumentation.trace.Classification
import java.text.DateFormat
import java.util.Date

/** Status and debug view of trace collection and the live Layer 2 adapter. */
@Composable
fun TraceCard(trace: TraceUiStatus, onExport: () -> Unit) {
    val c = trace.counts
    val none = stringResource(R.string.trace_none)
    val last = c.lastCandidate
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(text = stringResource(R.string.trace_title), style = MaterialTheme.typography.titleMedium)
                Text(
                    text = stringResource(if (trace.observing) R.string.trace_observing else R.string.trace_not_observing),
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    color = if (trace.observing) MaterialTheme.colorScheme.primary
                    else MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            Text(
                text = stringResource(R.string.trace_detail),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            StatRow(R.string.trace_events_observed, c.observedEvents.toString())
            StatRow(R.string.trace_launch_candidates, c.launchCandidates.toString())
            StatRow(R.string.trace_supported, c.supportedLaunches.toString())
            StatRow(R.string.trace_collapsed, c.collapsed.toString())
            StatRow(R.string.trace_unmapped, c.unmapped.toString())
            StatRow(R.string.trace_excluded, c.excluded.toString())
            StatRow(R.string.trace_sequence_length, c.sequenceLength.toString())
            StatRow(R.string.trace_windows, c.windows.toString())
            StatRow(R.string.trace_mapping, stringResource(R.string.trace_mapping_value, trace.mappedPackages, trace.ambiguousPackages))
            StatRow(R.string.trace_last_package, last?.event?.packageName ?: none)
            StatRow(
                R.string.trace_last_classification,
                when {
                    last == null -> none
                    last.classification == Classification.SUPPORTED -> last.lsappName.orEmpty()
                    else -> last.classification?.name ?: none
                }
            )
            StatRow(
                R.string.trace_last_poll,
                c.lastPollMs?.let { DateFormat.getTimeInstance(DateFormat.MEDIUM).format(Date(it)) } ?: none
            )
            val l2 = trace.layer2
            StatRow(R.string.layer2_updates, l2?.updateCount?.toString() ?: none)
            StatRow(
                R.string.layer2_last_launch,
                l2?.lastPosition?.let { stringResource(R.string.layer2_last_launch_value, it) } ?: none
            )
            StatRow(
                R.string.layer2_last_reveal,
                l2?.lastReveal?.let {
                    stringResource(R.string.layer2_last_reveal_value, it.predictionPosition, it.layer1Rank, it.layer2Rank)
                } ?: none
            )
            StatRow(
                R.string.layer2_next,
                if (l2?.nextLayer1Top1 != null && l2.nextLayer2Top1 != null) {
                    stringResource(R.string.layer2_next_value, l2.nextLayer1Top1, l2.nextLayer2Top1)
                } else none
            )
            trace.message?.let {
                Text(text = it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
            }
            OutlinedButton(onClick = onExport) {
                Text(stringResource(R.string.trace_export))
            }
        }
    }
}

@Composable
private fun StatRow(@StringRes label: Int, value: String) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Text(
            text = stringResource(label),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.weight(1f)
        )
        Text(text = value, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
    }
}
