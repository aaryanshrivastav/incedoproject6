"""Unit tests for document parsers (PDF, DOCX, HTML, CSV, XLSX)."""

import csv
from pathlib import Path

import openpyxl
from docx import Document
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table

from kb.domain.documents import BlockType
from kb.ingestion.parsers.docx import DOCXParser
from kb.ingestion.parsers.html import HTMLParser
from kb.ingestion.parsers.pdf import PDFParser
from kb.ingestion.parsers.tabular import CSVParser, XLSXParser


def test_pdf_parser_with_text_and_table(tmp_path: Path) -> None:
    """Verify that PDFParser extracts text, headings, and tables with page numbers."""
    pdf_path = tmp_path / "sample.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()

    elements = [
        Paragraph("Overview", styles["Title"]),
        Paragraph("This is the main introductory paragraph of the document.", styles["Normal"]),
        Table(
            [
                ["Metric", "Value", "Status"],
                ["Revenue", "$100M", "Growing"],
                ["EBITDA", "$25M", "Stable"],
            ],
            style=[("GRID", (0, 0), (-1, -1), 1, colors.black)],
        ),
    ]
    doc.build(elements)

    parser = PDFParser(ocr_char_threshold=5)
    parsed = parser.parse(pdf_path)

    assert parsed.page_count >= 1
    assert len(parsed.blocks) >= 2
    assert not parsed.used_ocr

    # Verify table block was extracted
    table_blocks = [b for b in parsed.blocks if b.type == BlockType.TABLE]
    assert len(table_blocks) == 1
    tbl = table_blocks[0]
    assert tbl.table is not None
    assert tbl.table.header == ["Metric", "Value", "Status"]
    assert len(tbl.table.rows) == 2
    assert "Value: $100M" in tbl.text
    assert "Metric: Revenue" in tbl.text


def test_docx_parser(tmp_path: Path) -> None:
    """Verify that DOCXParser extracts headings, paragraphs, lists, and tables in visual order."""
    docx_path = tmp_path / "sample.docx"
    doc = Document()

    doc.add_heading("Executive Summary", level=1)
    doc.add_paragraph("This document outlines corporate strategy for Q3.")
    doc.add_heading("Financial Metrics", level=2)
    doc.add_paragraph("First key item", style="List Bullet")
    doc.add_paragraph("Second key item", style="List Bullet")

    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Quarter"
    table.cell(0, 1).text = "Target"
    table.cell(1, 0).text = "Q3"
    table.cell(1, 1).text = "$50M"

    doc.save(str(docx_path))

    parser = DOCXParser()
    parsed = parser.parse(docx_path)

    assert parsed.title == "Executive Summary"
    assert len(parsed.blocks) == 6

    # Heading 1
    h1 = parsed.blocks[0]
    assert h1.type == BlockType.HEADING
    assert h1.level == 1
    assert h1.text == "Executive Summary"
    assert h1.section_path == ["Executive Summary"]

    # Paragraph under H1
    p1 = parsed.blocks[1]
    assert p1.type == BlockType.PARAGRAPH
    assert p1.section_path == ["Executive Summary"]

    # Heading 2
    h2 = parsed.blocks[2]
    assert h2.type == BlockType.HEADING
    assert h2.level == 2
    assert h2.text == "Financial Metrics"
    assert h2.section_path == ["Executive Summary", "Financial Metrics"]

    # List items under H2
    li1 = parsed.blocks[3]
    assert li1.type == BlockType.LIST_ITEM
    assert li1.section_path == ["Executive Summary", "Financial Metrics"]

    # Table under H2
    tbl_block = parsed.blocks[5]
    assert tbl_block.type == BlockType.TABLE
    assert tbl_block.section_path == ["Executive Summary", "Financial Metrics"]
    assert tbl_block.table is not None
    assert tbl_block.table.header == ["Quarter", "Target"]
    assert "Quarter: Q3 | Target: $50M" in tbl_block.text


def test_html_parser(tmp_path: Path) -> None:
    """Verify that HTMLParser strips nav/script/footer and extracts structured blocks."""
    html_content = """<!DOCTYPE html>
    <html>
      <head><title>System Architecture</title></head>
      <body>
        <nav><a href="#">Ignored Link</a></nav>
        <script>console.log("ignore me");</script>
        <h1>Core Architecture</h1>
        <p>The system uses modular pipelines for RAG.</p>
        <h2>Component List</h2>
        <ul>
          <li>Ingestion Engine</li>
          <li>Retrieval Gateway</li>
        </ul>
        <table>
          <tr><th>Component</th><th>Tech</th></tr>
          <tr><td>Database</td><td>PostgreSQL</td></tr>
        </table>
        <footer>Footer to be stripped</footer>
      </body>
    </html>
    """
    html_path = tmp_path / "arch.html"
    html_path.write_text(html_content, encoding="utf-8")

    parser = HTMLParser()
    parsed = parser.parse(html_path)

    assert parsed.title == "System Architecture"
    texts = [b.text for b in parsed.blocks]

    # Verify unwanted tags are decomposed
    assert not any("Ignored Link" in t for t in texts)
    assert not any("console.log" in t for t in texts)
    assert not any("Footer to be stripped" in t for t in texts)

    # Verify structured elements
    h1 = parsed.blocks[0]
    assert h1.type == BlockType.HEADING
    assert h1.text == "Core Architecture"
    assert h1.section_path == ["Core Architecture"]

    h2 = parsed.blocks[2]
    assert h2.type == BlockType.HEADING
    assert h2.text == "Component List"
    assert h2.section_path == ["Core Architecture", "Component List"]

    # Table
    tbl = parsed.blocks[5]
    assert tbl.type == BlockType.TABLE
    assert tbl.table is not None
    assert tbl.table.header == ["Component", "Tech"]
    assert "Component: Database | Tech: PostgreSQL" in tbl.text


def test_csv_parser(tmp_path: Path) -> None:
    """Verify CSVParser handles headers, data rows, and section naming."""
    csv_path = tmp_path / "sales_2026.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Region", "Sales", "Rep"])
        writer.writerow(["North", "150000", "Alice"])
        writer.writerow(["South", "220000", "Bob"])

    parser = CSVParser()
    parsed = parser.parse(csv_path)

    assert parsed.title == "sales_2026"
    assert len(parsed.blocks) == 1

    tbl = parsed.blocks[0]
    assert tbl.type == BlockType.TABLE
    assert tbl.section_path == ["sales_2026"]
    assert tbl.table is not None
    assert tbl.table.name == "sales_2026"
    assert tbl.table.header == ["Region", "Sales", "Rep"]
    assert len(tbl.table.rows) == 2
    assert "Region: North | Sales: 150000 | Rep: Alice" in tbl.text


def test_xlsx_parser(tmp_path: Path) -> None:
    """Verify XLSXParser handles multiple sheets as separate table blocks."""
    xlsx_path = tmp_path / "financials.xlsx"
    wb = openpyxl.Workbook()

    # Sheet 1: Income
    ws1 = wb.active
    ws1.title = "Income"
    ws1.append(["Category", "Amount"])
    ws1.append(["Services", "500000"])

    # Sheet 2: Expenses
    ws2 = wb.create_sheet(title="Expenses")
    ws2.append(["Category", "Cost"])
    ws2.append(["Payroll", "300000"])

    wb.save(str(xlsx_path))

    parser = XLSXParser()
    parsed = parser.parse(xlsx_path)

    assert parsed.title == "financials"
    assert len(parsed.blocks) == 2

    # Sheet 1
    t1 = parsed.blocks[0]
    assert t1.type == BlockType.TABLE
    assert t1.section_path == ["Income"]
    assert t1.table is not None
    assert t1.table.name == "Income"
    assert "Category: Services | Amount: 500000" in t1.text

    # Sheet 2
    t2 = parsed.blocks[1]
    assert t2.type == BlockType.TABLE
    assert t2.section_path == ["Expenses"]
    assert t2.table is not None
    assert t2.table.name == "Expenses"
    assert "Category: Payroll | Cost: 300000" in t2.text
