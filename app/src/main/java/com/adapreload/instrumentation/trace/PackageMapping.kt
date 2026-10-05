package com.adapreload.instrumentation.trace

/** Status of one row of the package mapping table (ANDROID_TRACE_SPEC section 3). */
enum class MappingStatus {
    /** Package verified to be the LSApp app named in the row; produces SUPPORTED launches. */
    MAPPED,

    /** Plausibly a vocabulary app, but the correspondence is unresolved; discarded until resolved. */
    AMBIGUOUS,

    /** Explicitly decided not to correspond to any vocabulary app; discarded. */
    UNSUPPORTED,
}

/**
 * One row: Android package → canonical app identity → LSApp display name → checkpoint app id.
 * [lsappName] and [lsappId] are set only for MAPPED rows.
 */
data class MappingEntry(
    val packageName: String,
    val canonicalApp: String,
    val lsappName: String?,
    val lsappId: Int?,
    val status: MappingStatus,
    val confidence: String,
    val source: String,
    val notes: String,
)

/** The researcher-curated mapping table (assets/package_mapping.tsv), validated against the vocabulary. */
class PackageMapping private constructor(
    /** Rows in file order. */
    val entries: List<MappingEntry>,
    /** SHA-256 of the table file, recorded per observation window (AU3). */
    val sha256: String,
) {
    private val byPackage = entries.associateBy { it.packageName }

    fun lookup(packageName: String): MappingEntry? = byPackage[packageName]

    companion object {
        val COLUMNS = listOf("package", "canonical_app", "lsapp_name", "lsapp_id", "status", "confidence", "source", "notes")

        /**
         * Parses the tab-separated table. Lines starting with '#' and blank lines are ignored; the
         * first other line must be the header. Throws IllegalArgumentException listing every
         * invalid row, so a broken table can never silently change the sequence.
         *
         * @param reservedPackages packages that are always excluded (AdaPreload itself and the
         *   explicit system list) and therefore must never be mapped.
         */
        fun parse(tsv: ByteArray, vocabulary: Vocabulary, reservedPackages: Set<String>): PackageMapping {
            val lines = String(tsv, Charsets.UTF_8).split('\n').map { it.removeSuffix("\r") }
            val content = lines.withIndex().filter { (_, l) -> l.isNotBlank() && !l.startsWith("#") }
            require(content.isNotEmpty() && content.first().value.split('\t') == COLUMNS) {
                "Mapping table header must be: ${COLUMNS.joinToString("\\t")}"
            }

            val errors = mutableListOf<String>()
            val entries = mutableListOf<MappingEntry>()
            val seen = HashSet<String>()
            val nameOfCanonical = HashMap<String, String>()

            for ((index, line) in content.drop(1)) {
                val where = "line ${index + 1}"
                val f = line.split('\t')
                if (f.size != COLUMNS.size) {
                    errors += "$where: expected ${COLUMNS.size} tab-separated fields, found ${f.size}"
                    continue
                }
                val (pkg, canonical, name, idText, statusText) = f
                val status = MappingStatus.entries.firstOrNull { it.name == statusText }
                when {
                    pkg.isBlank() -> errors += "$where: empty package"
                    !seen.add(pkg) -> errors += "$where: duplicate package $pkg"
                    pkg in reservedPackages -> errors += "$where: $pkg is always excluded and cannot be mapped"
                    canonical.isBlank() -> errors += "$where: empty canonical_app"
                    status == null -> errors += "$where: unknown status '$statusText'"
                    status == MappingStatus.MAPPED -> {
                        val vocabId = vocabulary.idOf(name)
                        when {
                            vocabId == null -> errors += "$where: '$name' is not an LSApp vocabulary name"
                            idText.toIntOrNull() != vocabId -> errors += "$where: lsapp_id $idText does not match vocabulary id $vocabId for '$name'"
                            nameOfCanonical.getOrPut(canonical) { name } != name ->
                                errors += "$where: canonical_app $canonical already maps to '${nameOfCanonical[canonical]}'"
                            else -> entries += MappingEntry(pkg, canonical, name, vocabId, status, f[5], f[6], f[7])
                        }
                    }
                    name.isNotEmpty() || idText.isNotEmpty() ->
                        errors += "$where: lsapp_name and lsapp_id must be empty unless status is MAPPED"
                    else -> entries += MappingEntry(pkg, canonical, null, null, status, f[5], f[6], f[7])
                }
            }
            require(errors.isEmpty()) { "Invalid package mapping table:\n" + errors.joinToString("\n") }
            return PackageMapping(entries, sha256Hex(tsv))
        }
    }
}
