package com.adapreload.instrumentation.trace

import com.adapreload.instrumentation.trace.TraceTestSupport.asset
import com.adapreload.instrumentation.trace.TraceTestSupport.mapping
import com.adapreload.instrumentation.trace.TraceTestSupport.mappingOf
import com.adapreload.instrumentation.trace.TraceTestSupport.row
import com.adapreload.instrumentation.trace.TraceTestSupport.vocabulary
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class MappingAndVocabularyTest {

    @Test
    fun shippedVocabularyMatchesTheAudit() {
        assertEquals(87, vocabulary.appCount)
        assertEquals("d0801f3b2e1ece7558989fa2fc70c86560aebb36a2e441f2ffa9b865a2147d78", vocabulary.appsSha256)
        assertEquals(20, vocabulary.window)
        assertEquals("AOL", vocabulary.nameOf(1))
        assertEquals("imo", vocabulary.nameOf(87))
        assertEquals(72, vocabulary.idOf("S’more"))
        assertNull(vocabulary.nameOf(Vocabulary.PAD_ID))
    }

    @Test
    fun tamperedVocabularyIsRejected() {
        val text = asset("lsapp_vocabulary.json").readText().replace("\"name\": \"AOL\"", "\"name\": \"AOL2\"")
        val error = runCatching { Vocabulary.parse(text) }.exceptionOrNull()
        assertTrue(error?.message.orEmpty().contains("apps_sha256 mismatch"))
    }

    @Test
    fun mappingLookupReturnsTheVocabularyIdForMappedPackages() {
        val entry = mapping.lookup("com.whatsapp")!!
        assertEquals(MappingStatus.MAPPED, entry.status)
        assertEquals("WhatsApp Messenger", entry.lsappName)
        assertEquals(vocabulary.idOf("WhatsApp Messenger"), entry.lsappId)
        assertEquals(MappingStatus.AMBIGUOUS, mapping.lookup("com.google.android.dialer")!!.status)
        assertNull(mapping.lookup("com.google.android.dialer")!!.lsappId)
        assertNull(mapping.lookup("com.example.not.listed"))
    }

    @Test
    fun invalidRowsAreRejectedWithReasons() {
        fun error(vararg rows: String) = runCatching { mappingOf(*rows) }.exceptionOrNull()?.message.orEmpty()

        assertTrue(error(row("com.a", "a", "Not An LSApp App", 5)).contains("not an LSApp vocabulary name"))
        assertTrue(error(row("com.a", "a", "YouTube", 1)).contains("does not match vocabulary id"))
        assertTrue(error(row("com.a", "a", "YouTube"), row("com.a", "b", "Gmail")).contains("duplicate package"))
        assertTrue(error(row("com.a", "a", "YouTube"), row("com.b", "a", "Gmail")).contains("already maps to"))
        assertTrue(error(row("com.android.systemui", "ui", "YouTube")).contains("cannot be mapped"))
        assertTrue(error("com.a\ta\tYouTube\t85\tAMBIGUOUS\tx\tx\tx").contains("must be empty unless status is MAPPED"))
        assertTrue(error("com.a\ta\t\t\tGUESSED\tx\tx\tx").contains("unknown status"))
    }

    @Test
    fun shippedMappingTableValidatesAgainstShippedVocabulary() {
        val table = PackageMapping.parse(
            asset("package_mapping.tsv").readBytes(),
            vocabulary,
            LaunchClassifier.SYSTEM_EXCLUSIONS + TraceTestSupport.SELF,
        )
        val mapped = table.entries.filter { it.status == MappingStatus.MAPPED }
        assertTrue(mapped.isNotEmpty())
        mapped.forEach { assertEquals(vocabulary.idOf(it.lsappName!!), it.lsappId) }
        // Unresolved cases stay unmapped.
        listOf("com.google.android.dialer", "com.google.android.apps.messaging", "com.sec.android.gallery3d", "com.android.settings")
            .forEach { assertEquals(it, MappingStatus.AMBIGUOUS, table.lookup(it)!!.status) }
    }

    @Test
    fun jsonRoundTripsEscapes() {
        val value = linkedMapOf<String, Any?>("s" to "a\"b\\c\n’", "n" to 12L, "l" to listOf(1L, null, true))
        assertEquals(value, Json.parse(Json.write(value)))
        assertTrue(Json.write(value).all { it.code < 128 })
    }
}
