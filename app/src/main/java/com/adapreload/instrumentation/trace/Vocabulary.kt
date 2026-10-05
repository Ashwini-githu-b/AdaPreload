package com.adapreload.instrumentation.trace

import java.security.MessageDigest

/**
 * The authoritative LSApp vocabulary (assets/lsapp_vocabulary.json), extracted from the frozen
 * Layer 1 checkpoint by tools/audit_vocabulary.py. Ids and names are read, never derived.
 */
class Vocabulary private constructor(
    /** App id (1..appCount) to LSApp display name. */
    val idToName: Map<Int, String>,
    val appsSha256: String,
    val window: Int,
) {
    private val nameToId: Map<String, Int> = idToName.entries.associate { (id, name) -> name to id }

    val appCount: Int get() = idToName.size

    fun idOf(lsappName: String): Int? = nameToId[lsappName]

    fun nameOf(appId: Int): String? = idToName[appId]

    companion object {
        const val PAD_ID = 0
        private const val FORMAT = "adapreload-lsapp-vocabulary"

        /** Parses and verifies the vocabulary file; throws IllegalArgumentException if invalid. */
        fun parse(json: String): Vocabulary {
            val root = Json.parse(json) as? Map<*, *> ?: throw IllegalArgumentException("Vocabulary is not a JSON object")
            require(root["format"] == FORMAT && root["format_version"] == 1L) { "Unknown vocabulary format" }
            val padding = root["padding"] as Map<*, *>
            require(padding["id"] == PAD_ID.toLong()) { "Padding id must be 0" }

            val idToName = LinkedHashMap<Int, String>()
            for (app in root["apps"] as List<*>) {
                app as Map<*, *>
                val id = (app["id"] as Long).toInt()
                require(idToName.put(id, app["name"] as String) == null) { "Duplicate id $id" }
            }
            require(idToName.keys.toList() == (1..idToName.size).toList()) { "Ids must be exactly 1..${idToName.size} in order" }
            require(idToName.values.toSet().size == idToName.size) { "Duplicate app name" }
            require(root["app_count"] == idToName.size.toLong()) { "app_count does not match the app list" }

            val model = root["model"] as Map<*, *>
            require(model["vocab_size"] == idToName.size + 1L) { "vocab_size must be app_count + 1 (padding)" }

            val declared = root["apps_sha256"] as String
            val actual = appsSha256(idToName)
            require(declared == actual) { "apps_sha256 mismatch: file says $declared, content hashes to $actual" }

            return Vocabulary(idToName, actual, (model["window"] as Long).toInt())
        }

        /** SHA-256 of "<id>\t<name>\n" lines in ascending id order (spec section 5.4). */
        fun appsSha256(idToName: Map<Int, String>): String =
            sha256Hex(idToName.keys.sorted().joinToString("") { "$it\t${idToName.getValue(it)}\n" }.toByteArray(Charsets.UTF_8))
    }
}

internal fun sha256Hex(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
