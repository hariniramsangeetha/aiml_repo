"""
tests/test_text_chunker.py
==========================
Tests for the text chunking module.

These tests verify that:
- Text is split into the right sizes
- Overlap works correctly
- Source metadata is preserved
- Edge cases (empty text, tiny text) are handled
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.text_chunker import TextChunker


class TestTextChunker:

    def setup_method(self):
        """Create a chunker with small sizes so we can test with short text."""
        self.chunker = TextChunker(chunk_size=100, overlap=20)

    # ── Basic chunking ────────────────────────────────────────────────────────

    def test_short_text_single_chunk(self):
        """Text shorter than chunk_size should produce exactly one chunk."""
        text = "This is a short sentence."
        chunks = self.chunker.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self):
        """Text longer than chunk_size should be split into multiple chunks."""
        # Create text that's clearly longer than 100 chars
        text = "A" * 50 + " " + "B" * 50 + " " + "C" * 50 + " " + "D" * 50
        chunks = self.chunker.chunk_text(text)
        assert len(chunks) > 1, "Long text should produce more than one chunk."

    def test_chunks_cover_all_content(self):
        """All words from the original text should appear in at least one chunk."""
        text = "The quick brown fox jumps over the lazy dog. " * 5
        chunks = self.chunker.chunk_text(text)
        all_chunk_text = " ".join(chunks)
        # All unique words should be present somewhere
        for word in ["quick", "brown", "fox", "lazy", "dog"]:
            assert word in all_chunk_text, f"Word '{word}' missing from chunks."

    def test_overlap_means_content_appears_twice(self):
        """With overlap > 0, some text should appear in consecutive chunks."""
        chunker = TextChunker(chunk_size=50, overlap=25)
        text = "AAAAAAAAAAAAAAAAAAAAAAAAA BBBBBBBBBBBBBBBBBBBBBBBBB CCCCCCCCCCCCCCCCCCCCCCCCC"
        chunks = chunker.chunk_text(text)
        if len(chunks) >= 2:
            # The end of chunk[0] and the beginning of chunk[1] should overlap
            end_of_first = chunks[0][-20:]
            start_of_second = chunks[1][:20]
            # There should be some shared characters
            assert len(set(end_of_first) & set(start_of_second)) > 0

    def test_empty_text_returns_empty_list(self):
        """Empty or whitespace-only text should return an empty list."""
        assert self.chunker.chunk_text("") == []
        assert self.chunker.chunk_text("   ") == []
        assert self.chunker.chunk_text("\n\n\n") == []

    def test_no_chunk_larger_than_chunk_size_plus_margin(self):
        """No chunk should be significantly larger than the configured chunk_size."""
        text = " ".join([f"word{i}" for i in range(200)])
        chunks = self.chunker.chunk_text(text)
        for chunk in chunks:
            # Allow a 20% margin because we try to break at word boundaries
            assert len(chunk) <= self.chunker.chunk_size * 1.3, \
                f"Chunk too long: {len(chunk)} > {self.chunker.chunk_size * 1.3}"

    # ── Metadata preservation ─────────────────────────────────────────────────

    def test_chunk_pages_preserves_filename(self):
        """Each chunk dict must include the source filename."""
        pages = [{"page": 1, "text": "A" * 300}]
        chunks = self.chunker.chunk_pages(pages, filename="myfile.pdf")
        for chunk in chunks:
            assert chunk.get("filename") == "myfile.pdf", \
                "Filename not preserved in chunk metadata."

    def test_chunk_pages_preserves_page_number(self):
        """Each chunk must know which page of the PDF it came from."""
        pages = [
            {"page": 1, "text": "Content from page one. " * 10},
            {"page": 5, "text": "Content from page five. " * 10},
        ]
        chunks = self.chunker.chunk_pages(pages, filename="doc.pdf")

        page_1_chunks = [c for c in chunks if c["page"] == 1]
        page_5_chunks = [c for c in chunks if c["page"] == 5]

        assert len(page_1_chunks) > 0, "Should have chunks from page 1."
        assert len(page_5_chunks) > 0, "Should have chunks from page 5."

    def test_chunk_pages_assigns_chunk_ids(self):
        """Each chunk must have a unique chunk_id."""
        pages = [{"page": 1, "text": "Word " * 100}]
        chunks = self.chunker.chunk_pages(pages, filename="doc.pdf")
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "chunk_ids must be unique."

    def test_chunk_pages_skips_empty_pages(self):
        """Pages with no text should be skipped (not produce empty chunks)."""
        pages = [
            {"page": 1, "text": "Real content here. " * 10},
            {"page": 2, "text": ""},  # Empty page
            {"page": 3, "text": "   "},  # Whitespace only
            {"page": 4, "text": "More real content here. " * 10},
        ]
        chunks = self.chunker.chunk_pages(pages, filename="doc.pdf")
        page_nums = {c["page"] for c in chunks}
        assert 2 not in page_nums, "Empty page 2 should be skipped."
        assert 3 not in page_nums, "Whitespace page 3 should be skipped."
        assert 1 in page_nums
        assert 4 in page_nums

    # ── Statistics ────────────────────────────────────────────────────────────

    def test_get_stats_returns_correct_count(self):
        """get_stats should return the correct total chunk count."""
        pages = [{"page": 1, "text": "Word " * 200}]
        chunks = self.chunker.chunk_pages(pages, filename="doc.pdf")
        stats = self.chunker.get_stats(chunks)
        assert stats["total_chunks"] == len(chunks)

    def test_get_stats_on_empty_list(self):
        """get_stats should handle an empty chunk list."""
        stats = self.chunker.get_stats([])
        assert stats["total_chunks"] == 0

    # ── Configuration validation ──────────────────────────────────────────────

    def test_invalid_chunk_size_raises_error(self):
        """Negative or zero chunk_size should raise ValueError."""
        with pytest.raises(ValueError):
            TextChunker(chunk_size=0, overlap=0)
        with pytest.raises(ValueError):
            TextChunker(chunk_size=-100, overlap=0)

    def test_overlap_exceeds_chunk_size_raises_error(self):
        """overlap >= chunk_size should raise ValueError."""
        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, overlap=100)
        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, overlap=150)
