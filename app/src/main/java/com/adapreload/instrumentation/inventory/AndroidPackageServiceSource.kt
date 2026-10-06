package com.adapreload.instrumentation.inventory

import android.content.Context
import android.content.pm.PackageInfo
import android.content.pm.PackageManager
import android.os.Build

/**
 * Reads declared services from the device's `PackageManager` (Phase E2a). This is the only
 * Android-dependent part of the inventory, and it is strictly read-only: the single call it makes
 * is `getPackageManager().getPackageInfo(pkg, GET_SERVICES | MATCH_DISABLED_COMPONENTS)`, which
 * returns manifest metadata. It never binds, starts, stops or invokes any component.
 *
 * API usage and version behaviour:
 * - `GET_SERVICES` asks for each package's declared `<service>` entries (`PackageInfo.services`).
 * - `MATCH_DISABLED_COMPONENTS` (API 24+) includes services disabled via `android:enabled=false`
 *   or a component setting, so the inventory can report COMPONENT_DISABLED rather than omitting
 *   them. minSdk here is 26.
 * - On API 33+ (`TIRAMISU`) the `PackageInfoFlags` overload is used; below it, the deprecated
 *   int-flags overload. Both return the same `ServiceInfo` fields.
 * - A package that is not installed, or not visible under AdaPreload's `<queries>` declaration,
 *   throws `NameNotFoundException`; this is reported as not installed (null), never as an error.
 *   Mapped apps are ordinary launchable apps, so they are visible via the existing LAUNCHER query.
 */
class AndroidPackageServiceSource(context: Context) : PackageServiceSource {
    private val packageManager = context.packageManager

    override fun servicesOf(packageName: String): List<DeclaredService>? {
        val info = packageInfo(packageName) ?: return null
        val services = info.services ?: return emptyList()
        return services.map { s ->
            DeclaredService(
                className = s.name,
                exported = s.exported,
                componentEnabled = s.enabled,
                applicationEnabled = s.applicationInfo?.enabled ?: true,
                permission = s.permission,
            )
        }
    }

    @Suppress("DEPRECATION")
    private fun packageInfo(packageName: String): PackageInfo? {
        val flags = PackageManager.GET_SERVICES or PackageManager.MATCH_DISABLED_COMPONENTS
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                packageManager.getPackageInfo(packageName, PackageManager.PackageInfoFlags.of(flags.toLong()))
            } else {
                packageManager.getPackageInfo(packageName, flags)
            }
        } catch (e: PackageManager.NameNotFoundException) {
            null
        }
    }
}
