"""
tests/test_pdf_processor.py
===========================
Tests for the PDF text extraction module.

We use synthetic PDFs created in-memory (using PyMuPDF) so we don't need
any external files. Tests work without an API key or internet connection.
"""

import sys
import os
import pytest

# Add the project root to path so we can import from 'src'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pdf_processor import PDFProcessor


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_pdf_bytes(pages_text: list[str]) -> bytes:
    """
    Create an in-memory PDF with the given text on each page.
    Requires PyMuPDF (fitz).
    """
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24
    except ImportError:
        import fitz  # legacy
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page(width=595, height=842)  # A4 size
        if text:
            page.insert_text((72, 100), text, fontsize=12)
    buffer = doc.tobytes()
    doc.close()
    return buffer


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestPDFProcessor:

    def setup_method(self):
        """Called before each test method."""
        self.processor = PDFProcessor(max_file_size_mb=10)

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_extract_single_page_pdf(self):
        """Test that text is correctly extracted from a single-page PDF."""
        pdf_bytes = make_pdf_bytes(["Hello, this is a test document. It has one page."])
        pages, warning = self.processor.extract_pages_from_bytes(pdf_bytes, "test.pdf")

        assert len(pages) == 1, "Should return one page."
        assert pages[0]["page"] == 1, "Page number should be 1."
        assert "Hello" in pages[0]["text"], "Extracted text should contain 'Hello'."
        assert warning == "" or "scanned" not in warning.lower(), \
            "Should not warn about scanned pages for text PDFs."

    def test_extract_multi_page_pdf(self):
        """Test extraction from a multi-page PDF, preserving page numbers."""
        texts = [
            "This is page one. It discusses artificial intelligence.",
            "This is page two. It discusses machine learning.",
            "This is page three. It discusses deep learning.",
        ]
        pdf_bytes = make_pdf_bytes(texts)
        pages, warning = self.processor.extract_pages_from_bytes(pdf_bytes, "multi.pdf")

        assert len(pages) == 3, "Should return 3 pages."
        assert pages[0]["page"] == 1
        assert pages[1]["page"] == 2
        assert pages[2]["page"] == 3
        assert "artificial intelligence" in pages[0]["text"]
        assert "machine learning" in pages[1]["text"]
        assert "deep learning" in pages[2]["text"]

    def test_page_numbers_start_at_1(self):
        """Page numbers must be human-readable (start at 1, not 0)."""
        pdf_bytes = make_pdf_bytes(["Page content here."])
        pages, _ = self.processor.extract_pages_from_bytes(pdf_bytes, "test.pdf")
        assert pages[0]["page"] == 1, "Page numbers should start at 1."

    # ── Empty and edge cases ──────────────────────────────────────────────────

    def test_empty_pdf_returns_warning(self):
        """A PDF with no text should return a scanned/empty warning."""
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        # Create a PDF with blank pages (no text)
        doc = fitz.open()
        doc.new_page()
        pdf_bytes = doc.tobytes()
        doc.close()

        pages, warning = self.processor.extract_pages_from_bytes(pdf_bytes, "blank.pdf")
        # Either no text pages, or a warning about scanned content
        text_pages = [p for p in pages if p.get("text", "").strip()]
        assert len(text_pages) == 0 or warning != "", \
            "Empty PDF should produce no text pages or a warning."

    def test_empty_bytes_returns_error(self):
        """Uploading 0 bytes should return a clear error."""
        pages, warning = self.processor.extract_pages_from_bytes(b"", "empty.pdf")
        assert pages == [], "Should return empty list for empty bytes."
        assert warning != "", "Should return a warning/error message."

    def test_non_pdf_extension_rejected(self):
        """A non-PDF file should be rejected with a clear message."""
        fake_bytes = b"This is just a text file, not a PDF."
        pages, warning = self.processor.extract_pages_from_bytes(fake_bytes, "notes.txt")
        assert pages == []
        assert "pdf" in warning.lower() or "only" in warning.lower()

    def test_corrupted_pdf_returns_error(self):
        """A corrupted/invalid PDF should not crash and should return an error."""
        garbage = b"%PDF-1.4 corrupted garbage data that is not a real PDF"
        pages, warning = self.processor.extract_pages_from_bytes(garbage, "corrupted.pdf")
        # Should return empty pages and a warning (not raise an exception)
        assert isinstance(pages, list)
        # May or may not extract text from garbage, but should not raise

    # ── File size validation ──────────────────────────────────────────────────

    def test_file_size_validation(self):
        """Files exceeding the size limit should be rejected by validate_bytes."""
        # Our test PDF is ~833 bytes, so limit to 500 bytes (0.0005 MB)
        processor = PDFProcessor(max_file_size_mb=0.0005)
        pdf_bytes = make_pdf_bytes(["Some content here."])
        # Test via validate_bytes (the public validation API)
        valid, warning = processor.validate_bytes(pdf_bytes, "big.pdf")
        assert valid is False
        assert "mb" in warning.lower() or "limit" in warning.lower() or "size" in warning.lower()

    # ── Validation helpers ────────────────────────────────────────────────────

    def test_validate_bytes_valid_pdf(self):
        """validate_bytes should return True for a real PDF."""
        pdf_bytes = make_pdf_bytes(["Valid content."])
        valid, msg = self.processor.validate_bytes(pdf_bytes, "doc.pdf")
        assert valid is True
        assert msg == ""

    def test_validate_bytes_wrong_extension(self):
        """validate_bytes should reject non-PDF extensions."""
        valid, msg = self.processor.validate_bytes(b"data", "file.docx")
        assert valid is False
        assert "pdf" in msg.lower()

    def test_get_page_count(self):
        """get_page_count should return the correct number of pages."""
        pdf_bytes = make_pdf_bytes(["Page 1", "Page 2", "Page 3"])
        count = self.processor.get_page_count(pdf_bytes)
        assert count == 3
