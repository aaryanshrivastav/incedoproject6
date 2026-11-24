"""PDF parser using pdfplumber with OCR fallback for scanned pages."""

import statistics
from pathlib import Path

import pdfplumber
import pytesseract

from kb.config import get_settings
from kb.domain.documents import (
    Block,
    BlockType,
    FailureReason,
    ParsedDocument,
    ParseError,
    TableData,
)
from kb.ingestion.normalize import normalize_text


class PDFParser:
    """Parser for PDF documents extracting text, headings, tables, and handling OCR fallback."""

    def __init__(
        self,
        ocr_char_threshold: int | None = None,
        ocr_lang: str | None = None,
    ) -> None:
        try:
            settings = get_settings()
            default_threshold = settings.OCR_CHAR_THRESHOLD
            default_lang = settings.OCR_LANG
        except Exception:
            default_threshold = 25
            default_lang = "eng"

        self.ocr_char_threshold = (
            ocr_char_threshold if ocr_char_threshold is not None else default_threshold
        )
        self.ocr_lang = ocr_lang if ocr_lang is not None else default_lang

    def parse(self, path: Path) -> ParsedDocument:
        """Parse a PDF file into a ParsedDocument."""
        if not path.exists():
            raise ParseError(FailureReason.CORRUPT, f"File not found: {path}")

        try:
            with open(path, "rb") as f:
                header = f.read(5)
                if header != b"%PDF-":
                    raise ParseError(
                        FailureReason.CORRUPT, "File is not a valid PDF (invalid magic bytes)"
                    )
        except OSError as e:
            raise ParseError(FailureReason.CORRUPT, f"Unable to read file: {e}") from e

        try:
            pdf = pdfplumber.open(path)
        except Exception as e:
            err_msg = str(e).lower()
            if "password" in err_msg or "encrypt" in err_msg:
                raise ParseError(FailureReason.ENCRYPTED, f"PDF is password-protected: {e}") from e
            raise ParseError(FailureReason.CORRUPT, f"Corrupted PDF file: {e}") from e

        with pdf:
            # Check for encryption
            try:
                # Access pages to trigger decryption check
                pages = pdf.pages
                page_count = len(pages)
            except Exception as e:
                err_msg = str(e).lower()
                if "password" in err_msg or "encrypt" in err_msg:
                    raise ParseError(
                        FailureReason.ENCRYPTED, f"PDF is password-protected: {e}"
                    ) from e
                raise ParseError(FailureReason.CORRUPT, f"Error reading PDF pages: {e}") from e

            if page_count == 0:
                raise ParseError(FailureReason.EMPTY, "PDF has no pages")

            blocks: list[Block] = []
            used_ocr = False
            active_section_path: list[str] = []
            inferred_title: str | None = None

            for page_idx, page in enumerate(pages, start=1):
                raw_text = page.extract_text() or ""
                cleaned_page_text = raw_text.strip()

                # Extract tables first
                page_tables: list[list[list[str | None]]] = []
                try:
                    page_tables = page.extract_tables() or []
                    if not page_tables:
                        page_tables = (
                            page.extract_tables(
                                {"vertical_strategy": "text", "horizontal_strategy": "text"}
                            )
                            or []
                        )
                except Exception:
                    page_tables = []

                # Trigger OCR fallback if extracted text is below threshold
                if len(cleaned_page_text) < self.ocr_char_threshold:
                    try:
                        # Verify tesseract is available
                        pytesseract.get_tesseract_version()
                    except Exception as e:
                        raise ParseError(
                            FailureReason.PARSE_ERROR,
                            f"Tesseract OCR binary not available or failed: {e}",
                        ) from e

                    try:
                        # Render page at 300 DPI for OCR
                        page_image = page.to_image(resolution=300).original
                        ocr_text = pytesseract.image_to_string(page_image, lang=self.ocr_lang)
                        if ocr_text.strip():
                            cleaned_page_text = ocr_text.strip()
                            used_ocr = True
                    except Exception as e:
                        raise ParseError(
                            FailureReason.PARSE_ERROR,
                            f"OCR processing failed on page {page_idx}: {e}",
                        ) from e

                # Detect font size statistics for heading detection if words available
                words = page.extract_words() if not used_ocr else []
                font_sizes = [
                    float(w["bottom"] - w["top"]) for w in words if "top" in w and "bottom" in w
                ]
                median_font_size = statistics.median(font_sizes) if font_sizes else 10.0

                # Process text lines
                if cleaned_page_text:
                    lines = [line.strip() for line in cleaned_page_text.split("\n") if line.strip()]
                    for line in lines:
                        normalized_line = normalize_text(line)
                        if not normalized_line:
                            continue

                        # Check if line looks like a heading (short line, not ending in punctuation,
                        # font size substantially above median)
                        is_heading = False
                        heading_level = 1

                        if words:
                            # Check word heights for this line
                            matching_words = [w for w in words if w.get("text", "") in line]
                            if matching_words:
                                avg_line_size = sum(
                                    float(w["bottom"] - w["top"]) for w in matching_words
                                ) / len(matching_words)
                                if (
                                    avg_line_size >= 1.25 * median_font_size
                                    and len(line) <= 80
                                    and not line.endswith((".", ",", ";", ":"))
                                ):
                                    is_heading = True
                                    heading_level = (
                                        1 if avg_line_size >= 1.45 * median_font_size else 2
                                    )

                        if is_heading:
                            if inferred_title is None:
                                inferred_title = normalized_line

                            # Maintain heading hierarchy
                            if heading_level == 1:
                                active_section_path = [normalized_line]
                            elif len(active_section_path) >= 1:
                                active_section_path = [active_section_path[0], normalized_line]
                            else:
                                active_section_path = [normalized_line]

                            blocks.append(
                                Block(
                                    type=BlockType.HEADING,
                                    text=normalized_line,
                                    level=heading_level,
                                    page=page_idx,
                                    section_path=list(active_section_path),
                                )
                            )
                        else:
                            blocks.append(
                                Block(
                                    type=BlockType.PARAGRAPH,
                                    text=normalized_line,
                                    page=page_idx,
                                    section_path=list(active_section_path),
                                )
                            )

                # Process tables on page
                for table_idx, raw_table in enumerate(page_tables, start=1):
                    if not raw_table or len(raw_table) == 0:
                        continue

                    # Filter out completely empty rows
                    cleaned_rows: list[list[str]] = [
                        [normalize_text(cell or "") for cell in row]
                        for row in raw_table
                        if any(cell and str(cell).strip() for cell in row)
                    ]

                    if not cleaned_rows:
                        continue

                    headers = cleaned_rows[0]
                    data_rows = cleaned_rows[1:] if len(cleaned_rows) > 1 else []

                    # Serialize table into self-describing text
                    serialized_lines: list[str] = []
                    for row in data_rows:
                        row_parts: list[str] = []
                        for col_idx, cell_val in enumerate(row):
                            col_header = (
                                headers[col_idx]
                                if col_idx < len(headers) and headers[col_idx]
                                else f"Col{col_idx + 1}"
                            )
                            row_parts.append(f"{col_header}: {cell_val}")
                        if row_parts:
                            serialized_lines.append(" | ".join(row_parts))

                    table_text = (
                        "\n".join(serialized_lines) if serialized_lines else " | ".join(headers)
                    )

                    table_data = TableData(
                        name=f"Table {table_idx} (Page {page_idx})",
                        header=headers,
                        rows=data_rows,
                    )

                    blocks.append(
                        Block(
                            type=BlockType.TABLE,
                            text=table_text,
                            page=page_idx,
                            section_path=list(active_section_path),
                            table=table_data,
                        )
                    )

            if not blocks:
                raise ParseError(
                    FailureReason.EMPTY, "PDF contains no extractable content or blocks"
                )

            doc_title = inferred_title or path.stem

            return ParsedDocument(
                title=doc_title,
                blocks=blocks,
                page_count=page_count,
                used_ocr=used_ocr,
                metadata={"source_file": path.name, "format": "pdf"},
            )
