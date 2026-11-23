"""Tabular parsers for CSV and Excel (XLSX) files."""

import csv
from pathlib import Path

import openpyxl

from kb.domain.documents import (
    Block,
    BlockType,
    FailureReason,
    ParsedDocument,
    ParseError,
    TableData,
)
from kb.ingestion.normalize import normalize_text


class CSVParser:
    """Parser for comma-separated value (CSV) files."""

    def parse(self, path: Path) -> ParsedDocument:
        """Parse a CSV file into a single-table ParsedDocument."""
        if not path.exists():
            raise ParseError(FailureReason.CORRUPT, f"File not found: {path}")

        raw_bytes = b""
        try:
            raw_bytes = path.read_bytes()
        except OSError as e:
            raise ParseError(FailureReason.CORRUPT, f"Unable to read file: {e}") from e

        if not raw_bytes.strip():
            raise ParseError(FailureReason.EMPTY, "CSV file is empty")

        # Decode content with encoding fallback
        text_content = ""
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                text_content = raw_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if not text_content:
            raise ParseError(FailureReason.PARSE_ERROR, "Failed to decode CSV content")

        lines = [line for line in text_content.splitlines() if line.strip()]
        if not lines:
            raise ParseError(FailureReason.EMPTY, "CSV file has no non-empty rows")

        # Sniff delimiter or default to comma
        sample = "\n".join(lines[:10])
        delimiter = ","
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            delimiter = dialect.delimiter
        except Exception:
            delimiter = ","

        reader = csv.reader(lines, delimiter=delimiter)
        raw_rows: list[list[str]] = []
        for row in reader:
            normalized_row = [normalize_text(cell) for cell in row]
            if any(cell for cell in normalized_row):
                raw_rows.append(normalized_row)

        if not raw_rows:
            raise ParseError(FailureReason.EMPTY, "CSV file has no valid data rows")

        headers = raw_rows[0]
        data_rows = raw_rows[1:] if len(raw_rows) > 1 else []

        serialized_lines: list[str] = []
        for r in data_rows:
            row_parts: list[str] = []
            for col_idx, cell_val in enumerate(r):
                col_header = (
                    headers[col_idx]
                    if col_idx < len(headers) and headers[col_idx]
                    else f"Col{col_idx + 1}"
                )
                row_parts.append(f"{col_header}: {cell_val}")
            if row_parts:
                serialized_lines.append(" | ".join(row_parts))

        table_text = "\n".join(serialized_lines) if serialized_lines else " | ".join(headers)
        sheet_name = path.stem

        table_data = TableData(
            name=sheet_name,
            header=headers,
            rows=data_rows,
        )

        block = Block(
            type=BlockType.TABLE,
            text=table_text,
            section_path=[sheet_name],
            table=table_data,
        )

        return ParsedDocument(
            title=sheet_name,
            blocks=[block],
            page_count=1,
            used_ocr=False,
            metadata={"source_file": path.name, "format": "csv"},
        )


class XLSXParser:
    """Parser for Excel (.xlsx) workbooks."""

    def parse(self, path: Path) -> ParsedDocument:
        """Parse an XLSX workbook into a multi-table ParsedDocument."""
        if not path.exists():
            raise ParseError(FailureReason.CORRUPT, f"File not found: {path}")

        try:
            with open(path, "rb") as f:
                magic = f.read(4)
                if magic != b"PK\x03\x04":
                    raise ParseError(
                        FailureReason.CORRUPT,
                        "File is not a valid XLSX archive (invalid magic bytes)",
                    )
        except OSError as e:
            raise ParseError(FailureReason.CORRUPT, f"Unable to read file: {e}") from e

        try:
            wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        except Exception as e:
            raise ParseError(
                FailureReason.CORRUPT, f"Corrupted or invalid XLSX workbook: {e}"
            ) from e

        blocks: list[Block] = []

        try:
            for sheetname in wb.sheetnames:
                sheet = wb[sheetname]
                raw_rows: list[list[str]] = []

                for row in sheet.iter_rows(values_only=True):
                    row_cells = [
                        normalize_text(str(cell)) if cell is not None else "" for cell in row
                    ]
                    if any(cell for cell in row_cells):
                        raw_rows.append(row_cells)

                if not raw_rows:
                    continue

                headers = raw_rows[0]
                data_rows = raw_rows[1:] if len(raw_rows) > 1 else []

                serialized_lines: list[str] = []
                for r in data_rows:
                    row_parts: list[str] = []
                    for col_idx, cell_val in enumerate(r):
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
                    name=sheetname,
                    header=headers,
                    rows=data_rows,
                )

                blocks.append(
                    Block(
                        type=BlockType.TABLE,
                        text=table_text,
                        section_path=[sheetname],
                        table=table_data,
                    )
                )
        finally:
            wb.close()

        if not blocks:
            raise ParseError(
                FailureReason.EMPTY, "XLSX workbook contains no non-empty sheets or rows"
            )

        return ParsedDocument(
            title=path.stem,
            blocks=blocks,
            page_count=1,
            used_ocr=False,
            metadata={"source_file": path.name, "format": "xlsx"},
        )
