"""Parser registry detecting file formats by magic bytes and content first."""

import zipfile
from pathlib import Path

from kb.domain.documents import FailureReason, ParseError
from kb.ingestion.parsers.base import Parser
from kb.ingestion.parsers.docx import DOCXParser
from kb.ingestion.parsers.html import HTMLParser
from kb.ingestion.parsers.pdf import PDFParser
from kb.ingestion.parsers.tabular import CSVParser, XLSXParser


def get_parser(path: Path) -> Parser:
    """Return the appropriate parser for the given file path.

    Inspects magic bytes and file content first, falling back to extension.
    A file with mismatched extension (e.g., text file named .pdf) raises ParseError.
    """
    if not path.exists():
        raise ParseError(FailureReason.CORRUPT, f"File not found: {path}")

    # Check for empty file upfront
    try:
        if path.stat().st_size == 0:
            raise ParseError(FailureReason.EMPTY, f"File is empty (0 bytes): {path.name}")
    except OSError as e:
        raise ParseError(FailureReason.CORRUPT, f"Unable to access file: {e}") from e

    # Read first 1024 bytes for magic byte inspection
    header = b""
    try:
        with open(path, "rb") as f:
            header = f.read(1024)
    except OSError as e:
        raise ParseError(FailureReason.CORRUPT, f"Unable to read file header: {e}") from e

    suffix = path.suffix.lower()

    # 1. PDF magic detection
    if header.startswith(b"%PDF-"):
        return PDFParser()
    if suffix == ".pdf":
        raise ParseError(
            FailureReason.CORRUPT,
            f"File '{path.name}' has .pdf extension but lacks PDF magic bytes (%PDF-)",
        )

    # 2. ZIP container detection (DOCX or XLSX)
    if header.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path, "r") as zf:
                namelist = zf.namelist()
                if any(name.startswith("word/") for name in namelist):
                    return DOCXParser()
                if any(name.startswith("xl/") for name in namelist):
                    return XLSXParser()
        except Exception as e:
            raise ParseError(
                FailureReason.CORRUPT, f"Corrupted or invalid ZIP archive for '{path.name}': {e}"
            ) from e

        # Fallback to extension if valid zip but non-standard structure
        if suffix == ".docx":
            return DOCXParser()
        if suffix == ".xlsx":
            return XLSXParser()

        raise ParseError(
            FailureReason.UNSUPPORTED,
            f"ZIP archive '{path.name}' is neither a Word document nor an Excel workbook",
        )

    if suffix in (".docx", ".xlsx"):
        raise ParseError(
            FailureReason.CORRUPT,
            f"File '{path.name}' has {suffix} extension but lacks valid ZIP magic bytes",
        )

    # 3. HTML detection
    stripped_header = header.lstrip().lower()
    is_html = (
        stripped_header.startswith(b"<!doctype html")
        or stripped_header.startswith(b"<html")
        or stripped_header.startswith(b"<?xml")
        or b"<body" in stripped_header
        or b"<p>" in stripped_header
        or b"<div>" in stripped_header
    )
    if is_html or suffix in (".html", ".htm"):
        return HTMLParser()

    # 4. CSV detection
    if suffix == ".csv":
        return CSVParser()

    # 5. Unsupported format
    raise ParseError(
        FailureReason.UNSUPPORTED,
        f"Unsupported file format or unrecognized magic bytes for '{path.name}' (suffix: {suffix})",
    )
