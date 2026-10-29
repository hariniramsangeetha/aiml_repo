"""
src/pdf_processor.py
====================
PURPOSE: Extract text from PDF files, page by page.

WHY WE NEED THIS:
  A PDF is a visual document – it's not just plain text. We need a special
  library (PyMuPDF, also called "fitz") to open the PDF and read the text
  from each page. We also keep track of the page number so we can show
  users EXACTLY which page their answer came from.

HOW IT WORKS:
  1. Open the PDF file using PyMuPDF.
  2. Loop through each page.
  3. Extract the text from each page.
  4. Return a list of {"page": number, "text": "..."} dictionaries.
"""

import logging
from pathlib import Path
from typing import Optional

# PyMuPDF is imported as 'pymupdf' in newer versions, 'fitz' in older ones.
# We try the new name first, then fall back to the old name.
try:
    import pymupdf as fitz  # PyMuPDF >= 1.24
except ImportError:
    try:
        import fitz  # PyMuPDF < 1.24 (legacy name)
    except ImportError:
        fitz = None

# Set up a logger – this prints helpful debug messages to the console
# instead of crashing the whole app when something goes wrong.
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

# Minimum characters on a page to consider it "has text"
MIN_TEXT_LENGTH = 10


# ─── Main Class ───────────────────────────────────────────────────────────────

class PDFProcessor:
    """
    Handles all PDF-related operations: opening, validating, and extracting text.

    Usage example:
        processor = PDFProcessor()
        pages = processor.extract_pages("my_document.pdf")
        # pages = [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]
    """

    def __init__(self, max_file_size_mb: float = 50.0):
        """
        Initialize the processor.

        Args:
            max_file_size_mb: Maximum allowed file size. Files larger than this
                              will be rejected to protect memory.
        """
        # Convert MB to bytes for easy comparison
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)

        if fitz is None:
            logger.error("PyMuPDF (fitz) is not installed. Run: pip install PyMuPDF")

    # ──────────────────────────────────────────────────────────────────────────

    def validate_file(self, file_path: str) -> tuple[bool, str]:
        """
        Check that a file exists, is a PDF, and is not too large.

        Returns:
            (True, "") if the file is valid
            (False, "reason") if there is a problem
        """
        path = Path(file_path)

        # Does the file exist?
        if not path.exists():
            return False, f"File not found: {file_path}"

        # Is it a PDF? (Check the extension)
        if path.suffix.lower() != ".pdf":
            return False, "Only PDF files are supported."

        # Is it within the size limit?
        file_size = path.stat().st_size
        if file_size > self.max_file_size_bytes:
            size_mb = file_size / (1024 * 1024)
            limit_mb = self.max_file_size_bytes / (1024 * 1024)
            return False, f"File is {size_mb:.1f} MB, which exceeds the {limit_mb:.0f} MB limit."

        # Is it empty?
        if file_size == 0:
            return False, "The file is empty."

        return True, ""

    # ──────────────────────────────────────────────────────────────────────────

    def validate_bytes(self, file_bytes: bytes, filename: str) -> tuple[bool, str]:
        """
        Validate a PDF that comes as raw bytes (e.g., uploaded through Streamlit).

        Args:
            file_bytes: The raw content of the uploaded file.
            filename:   The original filename (used for extension check).

        Returns:
            (True, "") if valid, (False, "reason") if not.
        """
        if not filename.lower().endswith(".pdf"):
            return False, "Only PDF files are supported."

        if len(file_bytes) == 0:
            return False, "The uploaded file is empty."

        max_bytes = self.max_file_size_bytes
        if len(file_bytes) > max_bytes:
            size_mb = len(file_bytes) / (1024 * 1024)
            limit_mb = max_bytes / (1024 * 1024)
            return False, f"File is {size_mb:.1f} MB, limit is {limit_mb:.0f} MB."

        return True, ""

    # ──────────────────────────────────────────────────────────────────────────

    def extract_pages(self, file_path: str) -> list[dict]:
        """
        Open a PDF from disk and return a list of page dictionaries.

        Args:
            file_path: Path to the PDF file on disk.

        Returns:
            List of dicts: [{"page": 1, "text": "page content..."}, ...]
            Returns an empty list if extraction fails.
        """
        if fitz is None:
            logger.error("PyMuPDF not available.")
            return []

        valid, reason = self.validate_file(file_path)
        if not valid:
            logger.warning(f"Invalid file: {reason}")
            return []

        return self._extract_from_path(file_path)

    # ──────────────────────────────────────────────────────────────────────────

    def extract_pages_from_bytes(self, file_bytes: bytes, filename: str) -> tuple[list[dict], str]:
        """
        Extract pages from a PDF provided as raw bytes (from Streamlit's uploader).

        Args:
            file_bytes: Raw bytes of the PDF file.
            filename:   Original filename (for display and validation).

        Returns:
            (pages_list, warning_message)
            - pages_list: List of {"page": int, "text": str} dicts.
            - warning_message: Empty string if OK, or a user-friendly warning.
        """
        if fitz is None:
            return [], "PyMuPDF is not installed. Cannot extract text."

        valid, reason = self.validate_bytes(file_bytes, filename)
        if not valid:
            return [], reason

        try:
            # Open the PDF directly from bytes – no need to save to disk first
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            logger.error(f"Failed to open PDF '{filename}': {e}")
            return [], "Could not open the PDF. It may be corrupted."

        return self._extract_from_doc(doc, filename)

    # ──────────────────────────────────────────────────────────────────────────

    def _extract_from_path(self, file_path: str) -> list[dict]:
        """Internal: open a PDF from a file path and extract pages."""
        try:
            doc = fitz.open(file_path)
            pages, _ = self._extract_from_doc(doc, Path(file_path).name)
            return pages
        except Exception as e:
            logger.error(f"Error reading '{file_path}': {e}")
            return []

    # ──────────────────────────────────────────────────────────────────────────

    def _extract_from_doc(self, doc, filename: str) -> tuple[list[dict], str]:
        """
        Internal: Loop through all pages of an open PyMuPDF document.

        Returns:
            (pages, warning)
        """
        # Check if the PDF is encrypted (password-protected)
        if doc.is_encrypted:
            doc.close()
            return [], "This PDF is password-protected. Please upload an unlocked PDF."

        pages = []
        total_pages = len(doc)
        scanned_page_count = 0

        for page_num in range(total_pages):
            page = doc[page_num]

            # get_text() extracts all readable text from the page
            raw_text = page.get_text()

            # Clean up extra whitespace
            text = raw_text.strip()

            # Count pages that appear to have no text (likely scanned images)
            if len(text) < MIN_TEXT_LENGTH:
                scanned_page_count += 1
                logger.debug(f"Page {page_num + 1} has little or no text.")
                # Still add it (with empty text) so page numbers stay consistent
                pages.append({
                    "page": page_num + 1,  # Human-readable: start from 1
                    "text": ""
                })
            else:
                pages.append({
                    "page": page_num + 1,
                    "text": text
                })

        doc.close()

        # Build a warning if many pages appear to be scanned
        warning = ""
        if total_pages > 0 and scanned_page_count == total_pages:
            warning = (
                "⚠️ This PDF appears to be a scanned document (image-based). "
                "No text could be extracted. RAG requires text-based PDFs. "
                "OCR (Optical Character Recognition) is not supported in this version."
            )
        elif scanned_page_count > 0:
            warning = (
                f"⚠️ {scanned_page_count} of {total_pages} pages had no extractable text "
                f"(possibly scanned images). Those pages will be skipped."
            )

        logger.info(
            f"Extracted {total_pages} pages from '{filename}'. "
            f"Scanned/empty: {scanned_page_count}."
        )
        return pages, warning

    # ──────────────────────────────────────────────────────────────────────────

    def get_page_count(self, file_bytes: bytes) -> Optional[int]:
        """Return the number of pages in a PDF without fully extracting text."""
        if fitz is None:
            return None
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            count = len(doc)
            doc.close()
            return count
        except Exception:
            return None
