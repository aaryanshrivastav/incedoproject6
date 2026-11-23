"""Unit tests verifying all five ParseError failure reasons without crashing."""

from pathlib import Path
from unittest.mock import patch

import pytest
from reportlab.lib.pdfencrypt import StandardEncryption
from reportlab.pdfgen import canvas

from kb.domain.documents import FailureReason, ParseError
from kb.ingestion.parsers.pdf import PDFParser
from kb.ingestion.parsers.registry import get_parser


def test_corrupt_file_raises_corrupt(tmp_path: Path) -> None:
    """Verify that corrupt data or malformed files raise ParseError(CORRUPT)."""
    # 1. Text file with .pdf extension
    fake_pdf = tmp_path / "not_a_pdf.pdf"
    fake_pdf.write_text("This is plain text and not a PDF", encoding="utf-8")

    with pytest.raises(ParseError) as exc_info:
        get_parser(fake_pdf)
    assert exc_info.value.reason == FailureReason.CORRUPT

    # 2. Corrupt PDF bytes starting with %PDF- but truncated/garbled
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"%PDF-1.4\nGarbage data that is not a valid PDF trailer")

    parser = PDFParser()
    with pytest.raises(ParseError) as exc_info:
        parser.parse(corrupt_pdf)
    assert exc_info.value.reason == FailureReason.CORRUPT

    # 3. Corrupt DOCX with invalid ZIP bytes
    corrupt_docx = tmp_path / "corrupt.docx"
    corrupt_docx.write_bytes(b"PK\x03\x04\x00\x00Invalid zip structure")

    with pytest.raises(ParseError) as exc_info:
        get_parser(corrupt_docx)
    assert exc_info.value.reason == FailureReason.CORRUPT


def test_empty_file_raises_empty(tmp_path: Path) -> None:
    """Verify that zero-byte or contentless files raise ParseError(EMPTY)."""
    # 1. Zero-byte file
    empty_file = tmp_path / "empty.txt"
    empty_file.write_bytes(b"")

    with pytest.raises(ParseError) as exc_info:
        get_parser(empty_file)
    assert exc_info.value.reason == FailureReason.EMPTY

    # 2. Empty CSV
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("\n\n   \n", encoding="utf-8")

    parser = get_parser(empty_csv)
    with pytest.raises(ParseError) as exc_info:
        parser.parse(empty_csv)
    assert exc_info.value.reason == FailureReason.EMPTY


def test_unsupported_file_raises_unsupported(tmp_path: Path) -> None:
    """Verify that unsupported extensions or formats raise ParseError(UNSUPPORTED)."""
    binary_file = tmp_path / "program.exe"
    binary_file.write_bytes(b"\x4d\x5a\x90\x00\x03\x00\x00\x00")

    with pytest.raises(ParseError) as exc_info:
        get_parser(binary_file)
    assert exc_info.value.reason == FailureReason.UNSUPPORTED


def test_encrypted_pdf_raises_encrypted(tmp_path: Path) -> None:
    """Verify that password-protected PDFs raise ParseError(ENCRYPTED)."""
    encrypted_path = tmp_path / "protected.pdf"
    enc = StandardEncryption("secret_password", canPrint=0)
    c = canvas.Canvas(str(encrypted_path), encrypt=enc)
    c.drawString(100, 100, "Secret content")
    c.showPage()
    c.save()

    parser = PDFParser()
    with pytest.raises(ParseError) as exc_info:
        parser.parse(encrypted_path)
    assert exc_info.value.reason == FailureReason.ENCRYPTED


def test_missing_tesseract_when_ocr_needed_raises_parse_error(tmp_path: Path) -> None:
    """Verify missing Tesseract when OCR is required raises ParseError(PARSE_ERROR)."""
    # Create empty PDF page requiring OCR
    from reportlab.pdfgen import canvas

    empty_pdf = tmp_path / "empty_scanned.pdf"
    c = canvas.Canvas(str(empty_pdf))
    c.showPage()
    c.save()

    parser = PDFParser(ocr_char_threshold=50)

    # Mock pytesseract.get_tesseract_version to simulate missing binary
    with patch(
        "pytesseract.get_tesseract_version", side_effect=FileNotFoundError("tesseract not found")
    ):
        with pytest.raises(ParseError) as exc_info:
            parser.parse(empty_pdf)
        assert exc_info.value.reason == FailureReason.PARSE_ERROR
        assert "Tesseract OCR binary not available" in exc_info.value.message
