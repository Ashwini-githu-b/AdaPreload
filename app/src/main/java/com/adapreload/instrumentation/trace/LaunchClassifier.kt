package com.adapreload.instrumentation.trace

/**
 * Classifies an ACTIVITY_RESUMED package (ANDROID_TRACE_SPEC A4). First match wins:
 *
 * 1. AdaPreload itself
 * 2. the current default home app (only that one: Settings also declares a fallback HOME
 *    activity and is a vocabulary app)
 * 3. enabled input methods
 * 4. the explicit system list below
 * 5. the mapping table (MAPPED → SUPPORTED, or AMBIGUOUS / UNSUPPORTED)
 * 6. no launcher entry
 * 7. everything else is OOV
 *
 * There is no blanket system-app rule: many vocabulary apps are preinstalled system apps. The
 * mapping (5) comes before the launcher test (6), so a mapped app without a launcher icon is kept.
 */
class LaunchClassifier(private val mapping: PackageMapping) {

    data class Result(val classification: Classification, val entry: MappingEntry?)

    fun classify(packageName: String, env: EnvironmentSnapshot, hasLauncherEntry: (String) -> Boolean): Result {
        val fixed = when (packageName) {
            env.selfPackage -> Classification.SELF
            env.defaultHomePackage -> Classification.EXCLUDED_HOME
            in env.imePackages -> Classification.EXCLUDED_IME
            in SYSTEM_EXCLUSIONS -> Classification.EXCLUDED_SYSTEM
            else -> null
        }
        if (fixed != null) return Result(fixed, null)

        val entry = mapping.lookup(packageName)
        if (entry != null) {
            val classification = when (entry.status) {
                MappingStatus.MAPPED -> Classification.SUPPORTED
                MappingStatus.AMBIGUOUS -> Classification.AMBIGUOUS
                MappingStatus.UNSUPPORTED -> Classification.UNSUPPORTED
            }
            return Result(classification, entry)
        }
        return Result(if (hasLauncherEntry(packageName)) Classification.OOV else Classification.EXCLUDED_NON_LAUNCHABLE, null)
    }

    companion object {
        /**
         * Explicit system exceptions (spec A4 row 4): SystemUI, permission dialogs and the system
         * chooser/resolver. To be validated on the target device.
         */
        val SYSTEM_EXCLUSIONS: Set<String> = setOf(
            "android",
            "com.android.systemui",
            "com.android.intentresolver",
            "com.google.android.permissioncontroller",
            "com.android.permissioncontroller",
            "com.google.android.packageinstaller",
            "com.android.packageinstaller",
        )
    }
}
