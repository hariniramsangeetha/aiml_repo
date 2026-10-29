"""
src/retriever.py
================
PURPOSE: Find the most relevant document chunks for a user's question.

HOW RETRIEVAL WORKS (simple explanation):
  1. Take the user's question: "What is photosynthesis?"
  2. Convert it to a vector (embedding) using the same model we used for chunks.
  3. Search the FAISS index for the chunks whose vectors are closest to the question.
  4. Return the top matching chunks with their similarity scores.

WHY SAME EMBEDDING MODEL?
  The embedding model maps both questions and document text to the same
  "meaning space". So similar meanings → similar vectors → easy to find.
  If you use a different model for questions vs. documents, the vectors
  won't be comparable and search will fail.

WHAT THIS MODULE ADDS:
  - Deduplication: removes near-identical chunks from results.
  - Logging: records what was retrieved for debugging.
  - Clean interface: hides FAISS details from the rest of the app.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class Retriever:
    """
    Connects the EmbeddingModel and VectorStore to answer retrieval queries.

    Usage:
        retriever = Retriever(embedding_model, vector_store)
        results = retriever.retrieve("What is the capital of France?")
        # results = [{"text": "...", "page": 2, "filename": "geo.pdf", "similarity": 0.89}, ...]
    """

    def __init__(
        self,
        embedding_model,
        vector_store,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
    ):
        """
        Args:
            embedding_model:     An EmbeddingModel instance (must be loaded).
            vector_store:        A VectorStore instance (must have data).
            top_k:               Max chunks to retrieve per question.
            similarity_threshold: Minimum similarity score (0-1) to include a chunk.
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = int(os.getenv("TOP_K_RESULTS", top_k))
        self.similarity_threshold = float(
            os.getenv("SIMILARITY_THRESHOLD", similarity_threshold)
        )

    # ──────────────────────────────────────────────────────────────────────────

    def retrieve(self, question: str, top_k: Optional[int] = None) -> list[dict]:
        """
        Retrieve the most relevant chunks for the given question.

        Args:
            question: The user's natural language question.
            top_k:    Override the default top_k for this call.

        Returns:
            List of relevant chunk dicts (may be empty if nothing matches).
            Each dict has: text, page, filename, similarity, chunk_id.
        """
        if not question or not question.strip():
            logger.warning("Empty question received.")
            return []

        if not self.vector_store.is_ready():
            logger.warning("Vector store is not ready. No documents indexed yet.")
            return []

        if not self.embedding_model.is_loaded():
            logger.error("Embedding model is not loaded.")
            return []

        k = top_k if top_k is not None else self.top_k

        # Step 1: Convert the question into a vector
        question_embedding = self.embedding_model.encode_single(question.strip())
        if question_embedding is None:
            logger.error("Failed to encode question.")
            return []

        # Step 2: Search the FAISS index for similar chunks
        raw_results = self.vector_store.search(
            query_embedding=question_embedding,
            top_k=k,
            similarity_threshold=self.similarity_threshold,
        )

        # Step 3: Remove near-duplicate chunks (same text from different chunk IDs)
        results = self._deduplicate(raw_results)

        logger.info(
            f"Retrieval for '{question[:60]}...': "
            f"{len(raw_results)} raw → {len(results)} after dedup"
        )

        return results

    # ──────────────────────────────────────────────────────────────────────────

    def _deduplicate(self, results: list[dict]) -> list[dict]:
        """
        Remove chunks that are nearly identical to an already-included chunk.

        This can happen when two overlapping chunks cover almost the same text.
        We keep only one of them (the one with higher similarity).

        Method: if 80% of a chunk's words are in an already-kept chunk, skip it.
        """
        if not results:
            return []

        kept = []
        kept_texts = []  # Store lowercased words for comparison

        for result in results:
            text = result.get("text", "")
            words = set(text.lower().split())

            is_duplicate = False
            for kept_words in kept_texts:
                if not words:
                    is_duplicate = True
                    break
                # Overlap ratio: how many words from this chunk appear in a kept chunk?
                overlap = len(words & kept_words) / len(words)
                if overlap > 0.8:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(result)
                kept_texts.append(words)

        return kept

    # ──────────────────────────────────────────────────────────────────────────

    def format_context(self, results: list[dict]) -> str:
        """
        Format retrieved chunks into a single context string for the LLM.

        The LLM will read this formatted text when generating an answer.
        Each chunk is labeled with its source so the LLM knows where it came from.

        Args:
            results: List of chunk dicts from retrieve().

        Returns:
            A multi-line string ready to be inserted into the LLM prompt.
        """
        if not results:
            return ""

        lines = []
        for i, chunk in enumerate(results, start=1):
            filename = chunk.get("filename", "Unknown")
            page = chunk.get("page", "?")
            text = chunk.get("text", "")
            similarity = chunk.get("similarity", 0.0)

            lines.append(
                f"[Source {i}: {filename}, Page {page} | Relevance: {similarity:.2f}]\n"
                f"{text}"
            )

        return "\n\n---\n\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        """True if the retriever can handle questions right now."""
        return (
            self.embedding_model.is_loaded()
            and self.vector_store.is_ready()
        )
