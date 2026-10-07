package com.adapreload.instrumentation.prewarm

import android.os.Bundle
import android.os.SystemClock
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/**
 * DEBUG-ONLY screen that drives [PrewarmLab] for the E2b behavioural validation. One candidate is
 * bound at a time; the researcher gathers process evidence (pidof / dumpsys / oom_score_adj) and
 * launch timing with adb in parallel. Nothing here launches the target app or its UI.
 *
 * Only approved candidates are offered. Each action logs an E2B_VALIDATE line under "AdaPreloadPrewarm".
 */
class PrewarmLabActivity : ComponentActivity() {
    private val lab by lazy { PrewarmLabHolder.get(applicationContext) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                var last by remember { mutableStateOf("Ready. Approved candidates only. Bind/unbind are logged (tag AdaPreloadPrewarm).") }
                fun trial() = SystemClock.elapsedRealtime()
                Column(
                    modifier = Modifier.fillMaxSize().padding(16.dp).verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    Text("E2b prewarm lab (debug only)", style = MaterialTheme.typography.titleMedium)
                    Text(
                        "Binds one approved service to test process pre-creation. No navigation, no data, no UI launch.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    for (candidate in PrewarmCandidate.approved) {
                        Text(candidate.display + " — " + candidate.service, style = MaterialTheme.typography.bodyMedium)
                        Button(onClick = { lab.bind(candidate, trial()); last = "bind ${candidate.display}" }, modifier = Modifier) {
                            Text("Bind (bind-only)")
                        }
                        if (candidate == PrewarmCandidate.BRAVE_CUSTOMTABS) {
                            Button(onClick = { lab.bindBraveWarmup(trial()); last = "bind+warmup ${candidate.display}" }) {
                                Text("Bind + documented warmup(0)")
                            }
                        }
                    }
                    OutlinedButton(onClick = { lab.unbind(trial()); last = "unbind" }) { Text("Unbind") }
                    Text("Last action: $last", style = MaterialTheme.typography.bodySmall)
                    Text("Bound: ${lab.isBound}", style = MaterialTheme.typography.bodySmall)
                }
            }
        }
    }

    override fun onDestroy() {
        // Never leave a binding open when the lab screen goes away.
        lab.unbind(SystemClock.elapsedRealtime())
        super.onDestroy()
    }
}
