package com.adapreload.instrumentation.model

import android.content.Context

/** Loads the frozen Layer 1 model shipped in the app assets (exported by tools/export_layer1.py). */
object Layer1Assets {
    const val MANIFEST = "layer1/layer1_manifest.json"
    const val WEIGHTS = "layer1/layer1_weights.bin"

    fun load(context: Context): Layer1Model {
        val assets = context.assets
        val manifest = assets.open(MANIFEST).use { String(it.readBytes(), Charsets.UTF_8) }
        val weights = assets.open(WEIGHTS).use { it.readBytes() }
        return Layer1Model.load(manifest, weights)
    }
}
