"""
src/text_chunker.py
===================
PURPOSE: Split long document pages into smaller, overlapping text "chunks".

WHY DO WE NEED CHUNKING?
  Imagine you have a 50-page textbook. If you try to compare your question
  against the ENTIRE book at once, it would be very slow and inaccurate.

  Instead, we cut the book into many short paragraphs (chunks). Then we
  compare the question against EACH small chunk. This is much faster and
  finds the RIGHT paragraph instead of the whole chapter.

WHAT IS OVERLAP?
  If a chunk ends in the middle of a sentence, the next chunk starts a bit
  earlier so the sentence is complete. This "overlap" prevents losing
  important information at the boundary between chunks.

  Example (chunk_size=20, overlap=5):
    Text:  "The quick brown fox jumps over the lazy dog"
    Chunk1: "The quick brown fox "     (characters 0-19)
    Chunk2: "fox jumps over the la"   (characters 15-34)  ← overlaps by 5
    Chunk3: "the lazy dog"            (characters 30-end) ← overlaps by 5

METADATA:
  Each chunk also stores WHERE it came from:
  - filename: which PDF file
  - page:     which page number in that PDF
  This lets us show the user "Your answer is from page 3 of document.pdf"
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Main Class ───────────────────────────────────────────────────────────────

class TextChunker:
    """
    Splits document text into overlapping chunks with source metadata.

    Usage example:
        chunker = TextChunker(chunk_size=500, overlap=100)
        pages = [{"page": 1, "text": "Long text here..."}, ...]
        chunks = chunker.chunk_pages(pages, filename="document.pdf")
        # chunks = [
        #   {"text": "Long text", "page": 1, "filename": "document.pdf", "chunk_id": 0},
        #   ...
        # ]
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        """
        Set up chunking parameters.

        Args:
            chunk_size: Target number of characters per chunk.
                        500 chars ≈ 75-100 words – a good paragraph size.
            overlap:    Number of characters shared between consecutive chunks.
                        100 chars gives enough context at boundaries.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if overlap < 0:
            raise ValueError("overlap cannot be negative.")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size.")

        self.chunk_size = chunk_size
        self.overlap = overlap

        logger.info(
            f"TextChunker initialized: chunk_size={chunk_size}, overlap={overlap}"
        )

    # ──────────────────────────────────────────────────────────────────────────

    def chunk_text(self, text: str) -> list[str]:
        """
        Split a single string of text into overlapping chunks.

        This is a simple character-based chunker. It tries to break at
        sentence boundaries (periods, newlines) to avoid cutting words.

        Args:
            text: The full text to split.

        Returns:
            List of text strings, each approximately chunk_size characters.
        """
        if not text or not text.strip():
            return []

        text = text.strip()
        chunks = []
        start = 0  # Starting character position of the current chunk
        text_len = len(text)

        while start < text_len:
            # Calculate where this chunk should end
            end = min(start + self.chunk_size, text_len)

            if end >= text_len:
                # We've reached the end – add the remaining text as the last chunk
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            # Try to find a good "break point" near the end of the chunk.
            # We prefer to break at a newline or period rather than mid-word.
            break_pos = self._find_break_point(text, end)

            # Safety: break_pos must always advance past start
            if break_pos <= start:
                break_pos = end  # Force progress

            chunk = text[start:break_pos].strip()
            if chunk:
                chunks.append(chunk)

            # Move the start forward, but go back by 'overlap' chars
            # so the next chunk slightly overlaps with this one.
            next_start = break_pos - self.overlap

            # CRITICAL: always advance forward to prevent infinite loop
            if next_start <= start:
                next_start = break_pos

            start = next_start

        return chunks

    # ──────────────────────────────────────────────────────────────────────────

    def _find_break_point(self, text: str, ideal_end: int) -> int:
        """
        Find the best place to break the text near 'ideal_end'.

        Prefers to break at:
          1. A double newline (paragraph boundary) – best
          2. A single newline
          3. A period followed by a space
          4. A space (word boundary)
          5. The ideal_end position (last resort)
        """
        # Search backward up to 100 characters from the ideal end
        search_start = max(0, ideal_end - 100)
        window = text[search_start:ideal_end]

        # 1. Double newline (paragraph break)
        pos = window.rfind("\n\n")
        if pos != -1:
            return search_start + pos + 2  # +2 to skip past the newlines

        # 2. Single newline
        pos = window.rfind("\n")
        if pos != -1:
            return search_start + pos + 1

        # 3. Period + space (end of sentence)
        pos = window.rfind(". ")
        if pos != -1:
            return search_start + pos + 2

        # 4. Space (word boundary)
        pos = window.rfind(" ")
        if pos != -1:
            return search_start + pos + 1

        # 5. No good break found – just cut at ideal_end
        return ideal_end

    # ──────────────────────────────────────────────────────────────────────────

    def chunk_pages(
        self,
        pages: list[dict],
        filename: str = "unknown.pdf"
    ) -> list[dict]:
        """
        Convert a list of page dicts (from PDFProcessor) into a flat list of chunks,
        each tagged with its source filename and page number.

        Args:
            pages:    List of {"page": int, "text": str} dicts.
            filename: The name of the PDF file (for source citations).

        Returns:
            List of chunk dicts:
            [
              {
                "text":      "The chunk content...",
                "page":      3,
                "filename":  "document.pdf",
                "chunk_id":  12,         # global index across all pages
                "char_start": 500        # character offset within the page
              },
              ...
            ]
        """
        all_chunks = []
        chunk_id = 0  # A running counter across all pages

        for page_data in pages:
            page_num = page_data.get("page", 0)
            page_text = page_data.get("text", "")

            if not page_text or not page_text.strip():
                # Skip pages with no text (scanned pages etc.)
                logger.debug(f"Skipping empty page {page_num}")
                continue

            # Split this page's text into chunks
            page_chunks = self.chunk_text(page_text)

            for chunk_text in page_chunks:
                all_chunks.append({
                    "text":     chunk_text,
                    "page":     page_num,
                    "filename": filename,
                    "chunk_id": chunk_id,
                })
                chunk_id += 1

        logger.info(
            f"Chunked '{filename}': {len(pages)} pages → {len(all_chunks)} chunks"
        )
        return all_chunks

    # ──────────────────────────────────────────────────────────────────────────

    def get_stats(self, chunks: list[dict]) -> dict:
        """
        Return basic statistics about a list of chunks.
        Useful for displaying in the UI (e.g., "Document split into 42 chunks").
        """
        if not chunks:
            return {"total_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}

        lengths = [len(c["text"]) for c in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_length":   sum(lengths) // len(lengths),
            "min_length":   min(lengths),
            "max_length":   max(lengths),
        }
