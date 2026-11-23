"""DOCX parser using python-docx with body element order preservation."""

from pathlib import Path

from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from kb.domain.documents import (
    Block,
    BlockType,
    FailureReason,
    ParsedDocument,
    ParseError,
    TableData,
)
from kb.ingestion.normalize import normalize_text


class DOCXParser:
    """Parser for Microsoft Word (.docx) documents."""

    def parse(self, path: Path) -> ParsedDocument:
        """Parse a .docx document into a ParsedDocument."""
        if not path.exists():
            raise ParseError(FailureReason.CORRUPT, f"File not found: {path}")

        try:
            with open(path, "rb") as f:
                magic = f.read(4)
                if magic != b"PK\x03\x04":
                    raise ParseError(
                        FailureReason.CORRUPT,
                        "File is not a valid DOCX archive (invalid magic bytes)",
                    )
        except OSError as e:
            raise ParseError(FailureReason.CORRUPT, f"Unable to read file: {e}") from e

        try:
            doc = Document(str(path))
        except Exception as e:
            raise ParseError(
                FailureReason.CORRUPT, f"Corrupted or invalid DOCX document: {e}"
            ) from e

        blocks: list[Block] = []
        heading_stack: list[tuple[int, str]] = []
        inferred_title: str | None = None

        # Iterate all body elements in document order
        for child in doc.element.body:
            if isinstance(child, CT_P):
                p = Paragraph(child, doc)
                text = normalize_text(p.text)
                if not text:
                    continue

                style_name = p.style.name if p.style else ""
                style_lower = style_name.lower()

                # Check for heading
                if style_lower.startswith("heading"):
                    level = 1
                    try:
                        level = int(style_name.split()[-1])
                    except (ValueError, IndexError):
                        level = 1

                    if inferred_title is None and level == 1:
                        inferred_title = text

                    # Update heading stack based on level
                    while heading_stack and heading_stack[-1][0] >= level:
                        heading_stack.pop()
                    heading_stack.append((level, text))
                    section_path = [title for _, title in heading_stack]

                    blocks.append(
                        Block(
                            type=BlockType.HEADING,
                            text=text,
                            level=level,
                            section_path=section_path,
                        )
                    )
                elif "list" in style_lower or "bullet" in style_lower:
                    section_path = [title for _, title in heading_stack]
                    blocks.append(
                        Block(
                            type=BlockType.LIST_ITEM,
                            text=text,
                            section_path=section_path,
                        )
                    )
                else:
                    section_path = [title for _, title in heading_stack]
                    blocks.append(
                        Block(
                            type=BlockType.PARAGRAPH,
                            text=text,
                            section_path=section_path,
                        )
                    )

            elif isinstance(child, CT_Tbl):
                tbl = Table(child, doc)
                raw_rows: list[list[str]] = []
                for row in tbl.rows:
                    row_cells = [normalize_text(cell.text) for cell in row.cells]
                    if any(cell for cell in row_cells):
                        raw_rows.append(row_cells)

                if not raw_rows:
                    continue

                headers = raw_rows[0]
                data_rows = raw_rows[1:] if len(raw_rows) > 1 else []

                # Serialize table into self-describing text
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
                section_path = [title for _, title in heading_stack]

                table_data = TableData(
                    name=None,
                    header=headers,
                    rows=data_rows,
                )

                blocks.append(
                    Block(
                        type=BlockType.TABLE,
                        text=table_text,
                        section_path=section_path,
                        table=table_data,
                    )
                )

        if not blocks:
            raise ParseError(FailureReason.EMPTY, "DOCX document contains no content")

        doc_title = inferred_title or path.stem

        return ParsedDocument(
            title=doc_title,
            blocks=blocks,
            page_count=1,
            used_ocr=False,
            metadata={"source_file": path.name, "format": "docx"},
        )
