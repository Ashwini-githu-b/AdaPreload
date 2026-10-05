package com.adapreload.instrumentation.trace

/**
 * Minimal JSON reader and writer, so the trace core needs neither Android (org.json) nor an
 * extra library and runs unchanged in JVM unit tests.
 *
 * Reading yields Map<String, Any?> (insertion-ordered), List<Any?>, String, Long (integers),
 * Double (other numbers), Boolean or null. Writing accepts the same types plus Int; maps keep
 * insertion order, so output is deterministic for deterministic input.
 */
object Json {

    fun parse(text: String): Any? {
        val parser = Parser(text)
        val value = parser.readValue()
        parser.skipWhitespace()
        require(parser.atEnd()) { "Unexpected trailing content at offset ${parser.pos}" }
        return value
    }

    /**
     * Objects whose values are all scalars are written on one line; other containers are
     * indented by two spaces per level. Lists of scalars stay on one line.
     */
    fun write(value: Any?): String = StringBuilder().also { writeValue(it, value, 0) }.append('\n').toString()

    fun quote(s: String): String = StringBuilder().also { writeString(it, s) }.toString()

    private fun isScalar(v: Any?) = v !is Map<*, *> && v !is List<*>

    private fun writeValue(out: StringBuilder, value: Any?, level: Int) {
        when (value) {
            null -> out.append("null")
            is String -> writeString(out, value)
            is Boolean, is Int, is Long -> out.append(value.toString())
            is Map<*, *> -> writeMap(out, value, level)
            is List<*> -> writeList(out, value, level)
            else -> throw IllegalArgumentException("Unsupported JSON value: ${value::class}")
        }
    }

    private fun writeMap(out: StringBuilder, map: Map<*, *>, level: Int) {
        if (map.isEmpty()) {
            out.append("{}")
            return
        }
        val inline = map.values.all(::isScalar)
        out.append('{')
        map.entries.forEachIndexed { i, (k, v) ->
            if (i > 0) out.append(',')
            if (inline) {
                if (i > 0) out.append(' ')
            } else {
                newline(out, level + 1)
            }
            writeString(out, k as String)
            out.append(": ")
            writeValue(out, v, level + 1)
        }
        if (!inline) newline(out, level)
        out.append('}')
    }

    private fun writeList(out: StringBuilder, list: List<*>, level: Int) {
        if (list.isEmpty()) {
            out.append("[]")
            return
        }
        val inline = list.all(::isScalar)
        out.append('[')
        list.forEachIndexed { i, v ->
            if (i > 0) out.append(',')
            if (inline) {
                if (i > 0) out.append(' ')
            } else {
                newline(out, level + 1)
            }
            writeValue(out, v, level + 1)
        }
        if (!inline) newline(out, level)
        out.append(']')
    }

    private fun newline(out: StringBuilder, level: Int) {
        out.append('\n')
        repeat(level) { out.append("  ") }
    }

    private fun writeString(out: StringBuilder, s: String) {
        out.append('"')
        for (c in s) {
            when {
                c == '"' -> out.append("\\\"")
                c == '\\' -> out.append("\\\\")
                c == '\n' -> out.append("\\n")
                c == '\r' -> out.append("\\r")
                c == '\t' -> out.append("\\t")
                // Escape control and non-ASCII characters so the output is pure ASCII.
                c < ' ' || c > '~' -> out.append("\\u").append(String.format("%04x", c.code))
                else -> out.append(c)
            }
        }
        out.append('"')
    }

    private class Parser(private val s: String) {
        var pos = 0

        fun atEnd() = pos >= s.length

        fun skipWhitespace() {
            while (pos < s.length && s[pos] in " \t\r\n") pos++
        }

        fun readValue(): Any? {
            skipWhitespace()
            require(pos < s.length) { "Unexpected end of JSON" }
            return when (val c = s[pos]) {
                '{' -> readObject()
                '[' -> readArray()
                '"' -> readString()
                't' -> literal("true", true)
                'f' -> literal("false", false)
                'n' -> literal("null", null)
                else -> if (c == '-' || c.isDigit()) readNumber() else error("Unexpected '$c' at offset $pos")
            }
        }

        private fun literal(word: String, value: Any?): Any? {
            require(s.startsWith(word, pos)) { "Invalid literal at offset $pos" }
            pos += word.length
            return value
        }

        private fun readObject(): Map<String, Any?> {
            val map = LinkedHashMap<String, Any?>()
            pos++
            skipWhitespace()
            if (s[pos] == '}') {
                pos++
                return map
            }
            while (true) {
                skipWhitespace()
                require(s[pos] == '"') { "Expected object key at offset $pos" }
                val key = readString()
                skipWhitespace()
                require(s[pos] == ':') { "Expected ':' at offset $pos" }
                pos++
                require(key !in map) { "Duplicate key '$key'" }
                map[key] = readValue()
                skipWhitespace()
                when (s[pos++]) {
                    ',' -> continue
                    '}' -> return map
                    else -> error("Expected ',' or '}' at offset ${pos - 1}")
                }
            }
        }

        private fun readArray(): List<Any?> {
            val list = ArrayList<Any?>()
            pos++
            skipWhitespace()
            if (s[pos] == ']') {
                pos++
                return list
            }
            while (true) {
                list += readValue()
                skipWhitespace()
                when (s[pos++]) {
                    ',' -> continue
                    ']' -> return list
                    else -> error("Expected ',' or ']' at offset ${pos - 1}")
                }
            }
        }

        private fun readString(): String {
            val out = StringBuilder()
            pos++
            while (true) {
                require(pos < s.length) { "Unterminated string" }
                when (val c = s[pos++]) {
                    '"' -> return out.toString()
                    '\\' -> {
                        when (val e = s[pos++]) {
                            '"', '\\', '/' -> out.append(e)
                            'b' -> out.append('\b')
                            'f' -> out.append('\u000c')
                            'n' -> out.append('\n')
                            'r' -> out.append('\r')
                            't' -> out.append('\t')
                            'u' -> {
                                out.append(s.substring(pos, pos + 4).toInt(16).toChar())
                                pos += 4
                            }
                            else -> error("Invalid escape '\\$e' at offset ${pos - 1}")
                        }
                    }
                    else -> {
                        require(c >= ' ') { "Control character in string at offset ${pos - 1}" }
                        out.append(c)
                    }
                }
            }
        }

        private fun readNumber(): Any {
            val start = pos
            if (s[pos] == '-') pos++
            while (pos < s.length && (s[pos].isDigit() || s[pos] in ".eE+-")) pos++
            val token = s.substring(start, pos)
            return token.toLongOrNull() ?: token.toDouble()
        }
    }
}
