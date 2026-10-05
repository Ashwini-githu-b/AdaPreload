package com.adapreload.instrumentation.permission

import android.Manifest
import android.app.AppOpsManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Process
import android.provider.Settings

/**
 * Usage Access (PACKAGE_USAGE_STATS) is an app-op the user grants in Settings, not a runtime
 * permission, so its state is read from AppOpsManager.
 */
object UsageAccess {

    fun isGranted(context: Context): Boolean {
        val appOps = context.getSystemService(Context.APP_OPS_SERVICE) as AppOpsManager
        // API 29 renamed checkOpNoThrow to unsafeCheckOpNoThrow; API 37 deprecates the new name
        // again in favor of checkOpNoThrow. Both call the same implementation, so the original
        // name is correct on every supported API level.
        val mode = appOps.checkOpNoThrow(
            AppOpsManager.OPSTR_GET_USAGE_STATS,
            Process.myUid(),
            context.packageName
        )
        // MODE_DEFAULT means the app-op defers to the grant state of the manifest permission.
        return if (mode == AppOpsManager.MODE_DEFAULT) {
            context.checkSelfPermission(Manifest.permission.PACKAGE_USAGE_STATS) ==
                PackageManager.PERMISSION_GRANTED
        } else {
            mode == AppOpsManager.MODE_ALLOWED
        }
    }

    fun settingsIntent(): Intent = Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS)
}
