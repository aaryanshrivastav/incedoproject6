"""HTML parser using BeautifulSoup and lxml."""

from pathlib import Path

from bs4 import BeautifulSoup

from kb.domain.documents import (
    Block,
    BlockType,
    FailureReason,
    ParsedDocument,
    ParseError,
    TableData,
)
from kb.ingestion.normalize import normalize_text


class HTMLParser:
    """Parser for HTML documents stripping non-content tags and preserving structure."""

    def parse(self, path: Path) -> ParsedDocument:
        """Parse an HTML file into a ParsedDocument."""
        if not path.exists():
            raise ParseError(FailureReason.CORRUPT, f"File not found: {path}")

        raw_bytes = b""
        try:
            raw_bytes = path.read_bytes()
        except OSError as e:
            raise ParseError(FailureReason.CORRUPT, f"Unable to read file: {e}") from e

        if not raw_bytes.strip():
            raise ParseError(FailureReason.EMPTY, "HTML file is empty")

        # Decode content with fallback
        html_str = ""
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                html_str = raw_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if not html_str:
            raise ParseError(FailureReason.PARSE_ERROR, "Failed to decode HTML content")

        try:
            soup = BeautifulSoup(html_str, "lxml")
        except Exception as e:
            raise ParseError(FailureReason.PARSE_ERROR, f"HTML parsing error: {e}") from e

        # Extract title from <title> tag if present
        title_tag = soup.find("title")
        inferred_title = normalize_text(title_tag.get_text()) if title_tag else None

        # Remove non-content elements
        for unwanted in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
            unwanted.decompose()

        # Find body or use whole document
        container = soup.body if soup.body else soup

        blocks: list[Block] = []
        heading_stack: list[tuple[int, str]] = []

        # Walk through top-level elements or direct tags of interest
        for elem in container.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table"]):
            # Check if this element is nested inside a table (we process tables as whole units)
            if elem.name != "table" and elem.find_parent("table"):
                continue

            tag_name = elem.name.lower()
            text = normalize_text(elem.get_text())
            if not text:
                continue

            if tag_name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(tag_name[1])
                if inferred_title is None and level == 1:
                    inferred_title = text

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

            elif tag_name == "li":
                section_path = [title for _, title in heading_stack]
                blocks.append(
                    Block(
                        type=BlockType.LIST_ITEM,
                        text=text,
                        section_path=section_path,
                    )
                )

            elif tag_name == "p":
                section_path = [title for _, title in heading_stack]
                blocks.append(
                    Block(
                        type=BlockType.PARAGRAPH,
                        text=text,
                        section_path=section_path,
                    )
                )

            elif tag_name == "table":
                raw_rows: list[list[str]] = []
                for tr in elem.find_all("tr"):
                    cells = [normalize_text(td.get_text()) for td in tr.find_all(["th", "td"])]
                    if any(c for c in cells):
                        raw_rows.append(cells)

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
            raise ParseError(FailureReason.EMPTY, "HTML document contains no extractable content")

        doc_title = inferred_title or path.stem

        return ParsedDocument(
            title=doc_title,
            blocks=blocks,
            page_count=1,
            used_ocr=False,
            metadata={"source_file": path.name, "format": "html"},
        )
