"""Document parsers package."""

from kb.ingestion.parsers.base import Parser
from kb.ingestion.parsers.docx import DOCXParser
from kb.ingestion.parsers.html import HTMLParser
from kb.ingestion.parsers.pdf import PDFParser
from kb.ingestion.parsers.registry import get_parser
from kb.ingestion.parsers.tabular import CSVParser, XLSXParser

__all__ = [
    "CSVParser",
    "DOCXParser",
    "HTMLParser",
    "PDFParser",
    "Parser",
    "XLSXParser",
    "get_parser",
]
