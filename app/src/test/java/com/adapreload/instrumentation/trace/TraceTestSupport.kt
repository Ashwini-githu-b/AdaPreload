package com.adapreload.instrumentation.trace

import java.io.File

/** Shared fixtures for the trace unit tests. */
object TraceTestSupport {
    const val SELF = "com.adapreload.instrumentation"
    const val HOME = "com.example.launcher"
    const val IME = "com.example.keyboard"

    /** Unit tests run with the module directory as working directory. */
    fun asset(name: String): File = File("src/main/assets/$name")

    val vocabulary: Vocabulary by lazy { Vocabulary.parse(asset("lsapp_vocabulary.json").readText()) }

    /** A small mapping over the real vocabulary. Settings is mapped to show that a system app stays eligible. */
    val mapping: PackageMapping by lazy {
        mappingOf(
            row("com.whatsapp", "whatsapp", "WhatsApp Messenger"),
            row("com.instagram.android", "instagram", "Instagram"),
            row("com.google.android.youtube", "youtube", "YouTube"),
            row("com.android.settings", "settings", "Settings"),
            "com.google.android.dialer\tgoogle_phone\t\t\tAMBIGUOUS\tunresolved\tspec M1\tPhone vs In Call UI",
            "com.example.decided\texample\t\t\tUNSUPPORTED\tdecided\ttest\tnot a vocabulary app",
        )
    }

    fun row(pkg: String, canonical: String, name: String, id: Int? = vocabulary.idOf(name)) =
        "$pkg\t$canonical\t$name\t$id\tMAPPED\thigh\ttest\t"

    fun mappingOf(vararg rows: String): PackageMapping =
        PackageMapping.parse(
            (PackageMapping.COLUMNS.joinToString("\t") + "\n" + rows.joinToString("\n") + "\n").toByteArray(),
            vocabulary,
            LaunchClassifier.SYSTEM_EXCLUSIONS + SELF,
        )

    fun id(name: String): Int = vocabulary.idOf(name)!!

    val env = EnvironmentSnapshot(selfPackage = SELF, defaultHomePackage = HOME, imePackages = setOf(IME))

    /** Packages without a launcher entry in the test environment. */
    val nonLaunchable = setOf("com.example.service.only", "com.android.settings")

    val hasLauncherEntry: (String) -> Boolean = { it !in nonLaunchable }

    fun pipeline(m: PackageMapping = mapping) = TracePipeline(LaunchClassifier(m))

    fun resumed(ts: Long, pkg: String) = RawUsageEvent(ts, UsageEventTypes.ACTIVITY_RESUMED, pkg, "$pkg.Main")

    fun other(ts: Long, pkg: String, type: Int = 2) = RawUsageEvent(ts, type, pkg, "$pkg.Main")

    fun windowInfo(startMs: Long, e: EnvironmentSnapshot = env) =
        WindowInfo(startMs, "UTC", e, 1, mapping.sha256, vocabulary.appsSha256, 36, "test device")

    fun sequenceOf(records: List<TraceRecord>): List<Int> =
        records.filter { it.outcome == Outcome.APPENDED }.sortedBy { it.sequencePosition }.map { it.appId!! }
}

/** Returns the events of a scripted timeline that fall in [begin, end), in timeline order. */
class FakeEventSource(private val timeline: List<RawUsageEvent>) : UsageEventSource {
    val queried = mutableListOf<Pair<Long, Long>>()

    override fun query(beginMs: Long, endMs: Long): List<RawUsageEvent> {
        queried += beginMs to endMs
        return timeline.filter { it.timestampMs in beginMs until endMs }
    }
}

/** TraceStore in memory, deriving the sequence state from the records as the SQLite store does. */
class InMemoryTraceStore : TraceStore {
    val windows = mutableListOf<ObservationWindow>()
    val records = mutableListOf<StoredRecord>()
    private var cursorMs: Long? = null
    private var openWindowId: Long? = null
    private var experimentStartMs: Long? = null
    private var lastPollMs: Long? = null

    override fun loadState(): PersistedState {
        val appended = records.map { it.record }.filter { it.outcome == Outcome.APPENDED }
        val trace = TraceState(
            nextPosition = appended.size,
            lastRetainedAppId = appended.maxByOrNull { it.sequencePosition!! }?.appId,
            lastEventTimestampMs = records.lastOrNull()?.record?.event?.timestampMs,
        )
        return PersistedState(trace, cursorMs, openWindowId, experimentStartMs, lastPollMs)
    }

    override fun openWindow(info: WindowInfo): Long {
        val id = windows.size + 1L
        windows += ObservationWindow(id, info, null, null)
        openWindowId = id
        cursorMs = info.startMs
        if (experimentStartMs == null) experimentStartMs = info.startMs
        return id
    }

    override fun closeWindow(windowId: Long, endMs: Long, reason: CloseReason) {
        val i = windows.indexOfFirst { it.id == windowId }
        windows[i] = windows[i].copy(endMs = endMs, closeReason = reason)
        if (openWindowId == windowId) openWindowId = null
    }

    override fun commitBatch(records: List<TraceRecord>, cursorMs: Long, polledAtMs: Long) {
        records.forEach { this.records += StoredRecord(this.records.size + 1L, it) }
        this.cursorMs = cursorMs
        lastPollMs = polledAtMs
    }

    override fun snapshot() = TraceSnapshot(experimentStartMs, cursorMs, lastPollMs, windows.toList(), records.toList())

    override fun counts() = TraceCounts.of(records.map { it.record }, windows.size, lastPollMs)
}
