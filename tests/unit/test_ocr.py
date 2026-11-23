"""Unit tests verifying OCR fallback for scanned PDF documents without a text layer."""

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from kb.ingestion.parsers.pdf import PDFParser


@pytest.mark.requires_tesseract
def test_scanned_pdf_triggers_ocr(tmp_path: Path) -> None:
    """Verify that a scanned PDF without a text layer falls back to OCR and recovers text."""
    # 1. Create a pure bitmap image with text
    img = Image.new("RGB", (800, 400), color="white")
    draw = ImageDraw.Draw(img)

    # Draw clear text onto the image
    draw.text((50, 80), "CONFIDENTIAL INVOICE #9821", fill="black")
    draw.text((50, 150), "Client: Acme Corporation", fill="black")
    draw.text((50, 220), "Total Balance Due: $45000", fill="black")

    # 2. Save the image directly as a PDF (creates a pure raster PDF without a font/text layer)
    scanned_pdf = tmp_path / "scanned_invoice.pdf"
    img.save(str(scanned_pdf), "PDF", resolution=300.0)

    # 3. Parse with PDFParser (zero selectable text triggers OCR)
    parser = PDFParser(ocr_char_threshold=25)
    parsed = parser.parse(scanned_pdf)

    # 4. Verify OCR was executed and extracted key keywords
    assert parsed.used_ocr is True
    assert len(parsed.blocks) >= 1

    extracted_text = " ".join(b.text for b in parsed.blocks).upper()
    assert "INVOICE" in extracted_text or "CONFIDENTIAL" in extracted_text
    assert "ACME" in extracted_text or "CORPORATION" in extracted_text or "45000" in extracted_text
