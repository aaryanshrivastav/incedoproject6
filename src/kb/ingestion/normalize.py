"""Text normalization and deterministic document content hashing."""

import hashlib
import json
import re
import unicodedata

from kb.domain.documents import ParsedDocument

# Matches control characters other than newline, carriage return, and tab
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Matches horizontal whitespace (spaces, tabs)
_HORIZONTAL_WHITESPACE_RE = re.compile(r"[^\S\r\n]+")

# Matches 3 or more consecutive newlines
_EXCESSIVE_NEWLINES_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Normalize text into canonical Unicode NFC, strip control characters, and collapse whitespace.

    Preserves paragraph breaks (single/double newlines) and tabs if relevant.
    """
    if not text:
        return ""

    # 1. Unicode NFC normalization
    normalized = unicodedata.normalize("NFC", text)

    # 2. Strip non-printable control characters
    normalized = _CONTROL_CHAR_RE.sub("", normalized)

    # 3. Normalize CRLF to LF
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # 4. Collapse multiple horizontal whitespaces on each line
    lines = [_HORIZONTAL_WHITESPACE_RE.sub(" ", line).strip() for line in normalized.split("\n")]

    # 5. Rejoin and collapse excessive blank lines to max 2 newlines (paragraph boundary)
    cleaned = "\n".join(lines)
    cleaned = _EXCESSIVE_NEWLINES_RE.sub("\n\n", cleaned)

    return cleaned.strip()


def document_content_hash(parsed: ParsedDocument) -> str:
    """Compute deterministic SHA-256 hash over document content.

    Includes block types, normalized block texts, and table data.
    Excludes unstable or non-content attributes like page numbers and metadata.
    """
    hasher = hashlib.sha256()

    # Document title
    hasher.update(normalize_text(parsed.title).encode("utf-8"))

    for block in parsed.blocks:
        hasher.update(block.type.value.encode("utf-8"))
        hasher.update(normalize_text(block.text).encode("utf-8"))

        if block.table is not None:
            # Deterministic serialization of table header and rows
            table_dict = {
                "name": block.table.name,
                "header": block.table.header,
                "rows": block.table.rows,
            }
            hasher.update(json.dumps(table_dict, sort_keys=True, ensure_ascii=True).encode("utf-8"))

    return hasher.hexdigest()
