"""Unit tests for text normalization and document content hashing."""

from kb.domain.documents import Block, BlockType, ParsedDocument, TableData
from kb.ingestion.normalize import document_content_hash, normalize_text


def test_normalize_text_unicode_nfc() -> None:
    """Verify normalize_text converts decomposed Unicode into canonical NFC."""
    # Decomposed: 'e' + combining acute accent
    decomposed = "e\u0301cole"
    # Precomposed NFC
    precomposed = "\u00e9cole"

    assert decomposed != precomposed
    assert normalize_text(decomposed) == precomposed


def test_normalize_text_control_chars_and_whitespace() -> None:
    """Verify control characters are stripped and horizontal whitespace is collapsed."""
    raw = "Hello \x00\x05World!\tThis   is  a\r\n\r\ntest   with\n\n\n\nextra lines."
    normalized = normalize_text(raw)

    assert "\x00" not in normalized
    assert "\x05" not in normalized
    assert "Hello World! This is a" in normalized
    assert "\n\n" in normalized
    assert "\n\n\n" not in normalized


def test_document_content_hash_stability() -> None:
    """Verify that document content hash is stable across page numbers and metadata changes."""
    doc1 = ParsedDocument(
        title="Report 2026",
        blocks=[
            Block(
                type=BlockType.HEADING,
                text="Introduction",
                page=1,
                section_path=["Introduction"],
            ),
            Block(
                type=BlockType.PARAGRAPH,
                text="This is the first paragraph.   With some extra spacing.",
                page=1,
                section_path=["Introduction"],
            ),
        ],
        page_count=1,
        used_ocr=False,
        metadata={"created_at": "2026-01-01"},
    )

    doc2 = ParsedDocument(
        title="Report 2026",
        blocks=[
            Block(
                type=BlockType.HEADING,
                text="Introduction",
                page=5,  # Different page number
                section_path=["Introduction"],
            ),
            Block(
                type=BlockType.PARAGRAPH,
                text="This is the first paragraph. With some extra spacing.",
                page=5,
                section_path=["Introduction"],
            ),
        ],
        page_count=10,  # Different page count
        used_ocr=True,  # Different OCR flag
        metadata={"created_at": "2026-11-23"},  # Different metadata
    )

    hash1 = document_content_hash(doc1)
    hash2 = document_content_hash(doc2)

    assert hash1 == hash2


def test_document_content_hash_changes_with_content() -> None:
    """Verify that document content hash changes when content or tables change."""
    base_doc = ParsedDocument(
        title="Report",
        blocks=[
            Block(type=BlockType.PARAGRAPH, text="Base content"),
        ],
    )

    modified_text_doc = ParsedDocument(
        title="Report",
        blocks=[
            Block(type=BlockType.PARAGRAPH, text="Modified content"),
        ],
    )

    table_doc = ParsedDocument(
        title="Report",
        blocks=[
            Block(
                type=BlockType.TABLE,
                text="Col: val",
                table=TableData(header=["Col"], rows=[["val"]]),
            ),
        ],
    )

    base_hash = document_content_hash(base_doc)
    mod_hash = document_content_hash(modified_text_doc)
    tbl_hash = document_content_hash(table_doc)

    assert base_hash != mod_hash
    assert base_hash != tbl_hash
    assert mod_hash != tbl_hash
