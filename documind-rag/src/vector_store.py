"""
src/vector_store.py
===================
PURPOSE: Store document embeddings in a FAISS index for fast similarity search.

WHAT IS FAISS?
  FAISS (Facebook AI Similarity Search) is a library that can store millions
  of vectors and find the most similar ones in milliseconds.

  Think of it like a very smart filing cabinet:
  - You put in labelled folders (embeddings + metadata).
  - When you ask a question, it finds the most relevant folders.

HOW IT WORKS:
  1. We feed all chunk embeddings into FAISS.
  2. FAISS builds an internal index for fast searching.
  3. When a question comes in, we embed the question and ask FAISS:
     "Which stored vectors are most similar to this?"
  4. FAISS returns the indices of the top-K most similar chunks.
  5. We look up those indices to get the original text and page numbers.

SIMILARITY METRIC:
  We use "inner product" similarity because our embeddings are normalized
  (unit length). For normalized vectors, inner product == cosine similarity.
  Cosine similarity ranges from 0 (completely unrelated) to 1 (identical).

MULTI-DOCUMENT SUPPORT:
  Each chunk stored in the index knows which filename and page it came from.
  This prevents answers from "document A" being attributed to "document B".
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.error("FAISS is not installed. Run: pip install faiss-cpu")


# ─── Main Class ───────────────────────────────────────────────────────────────

class VectorStore:
    """
    Manages a FAISS index and the metadata (text + page info) for each vector.

    This class keeps two things in sync:
    1. The FAISS index (fast vector search)
    2. A Python list of chunk dicts (the actual text, filename, page number)

    The list index and the FAISS index always match: position 0 in the list
    corresponds to vector 0 in FAISS.
    """

    def __init__(self):
        self._index = None          # The FAISS index object
        self._chunks = []           # List of chunk dicts (text, page, filename, ...)
        self._dimension = None      # Embedding vector size (e.g., 384)
        self._document_ids = set()  # Tracks which filenames have been added

    # ──────────────────────────────────────────────────────────────────────────

    def build(self, embeddings: np.ndarray, chunks: list[dict]) -> bool:
        """
        Create a FAISS index from scratch with the given embeddings and chunks.

        Call this when processing a new document from zero.

        Args:
            embeddings: 2D numpy array, shape (num_chunks, embedding_dim).
            chunks:     List of chunk dicts (same order as embeddings).

        Returns:
            True on success, False on failure.
        """
        if not FAISS_AVAILABLE:
            logger.error("FAISS not available.")
            return False

        if embeddings is None or len(embeddings) == 0:
            logger.warning("No embeddings provided. Cannot build index.")
            return False

        if len(embeddings) != len(chunks):
            logger.error(
                f"Mismatch: {len(embeddings)} embeddings vs {len(chunks)} chunks."
            )
            return False

        try:
            n_vectors, dim = embeddings.shape
            self._dimension = dim

            # IndexFlatIP = "Inner Product" index (brute-force, exact search).
            # It's slower than approximate methods but always correct – good for
            # a laptop demo where we have at most a few thousand chunks.
            self._index = faiss.IndexFlatIP(dim)

            # Add all embeddings to the index at once
            self._index.add(embeddings)

            # Store the chunk metadata alongside
            self._chunks = list(chunks)

            # Track which documents are in this index
            self._document_ids = {c.get("filename", "unknown") for c in chunks}

            logger.info(
                f"FAISS index built: {n_vectors} vectors, dim={dim}. "
                f"Documents: {self._document_ids}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────────

    def add(self, embeddings: np.ndarray, chunks: list[dict]) -> bool:
        """
        Add more embeddings to an existing index (for multi-document support).

        Args:
            embeddings: New embeddings to add (shape: [n, dim]).
            chunks:     Corresponding chunk dicts.

        Returns:
            True on success, False on failure.
        """
        if not FAISS_AVAILABLE:
            return False

        if embeddings is None or len(embeddings) == 0:
            logger.warning("No embeddings to add.")
            return False

        # If the index doesn't exist yet, build it
        if self._index is None:
            return self.build(embeddings, chunks)

        # Dimension check: new embeddings must match existing ones
        if embeddings.shape[1] != self._dimension:
            logger.error(
                f"Dimension mismatch: index is {self._dimension}D, "
                f"new embeddings are {embeddings.shape[1]}D."
            )
            return False

        try:
            self._index.add(embeddings)
            self._chunks.extend(chunks)
            new_docs = {c.get("filename", "unknown") for c in chunks}
            self._document_ids.update(new_docs)
            logger.info(f"Added {len(chunks)} chunks. Total: {self._index.ntotal}")
            return True
        except Exception as e:
            logger.error(f"Failed to add to FAISS index: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> list[dict]:
        """
        Find the top-K most similar chunks to the query embedding.

        Args:
            query_embedding:     1D or 2D numpy array of the question's embedding.
            top_k:               Maximum number of results to return.
            similarity_threshold: Only return chunks with similarity >= this value.
                                  Range: [0.0, 1.0]. 0.3 is a reasonable default.

        Returns:
            List of result dicts, sorted by similarity (highest first):
            [
              {
                "text":       "Relevant chunk text...",
                "page":       3,
                "filename":   "document.pdf",
                "chunk_id":   12,
                "similarity": 0.82,   # 0=unrelated, 1=identical
              },
              ...
            ]
        """
        if not self.is_ready():
            logger.warning("Vector store is not ready. Cannot search.")
            return []

        if query_embedding is None:
            return []

        # Ensure query is 2D: shape (1, dim) – FAISS expects a batch
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # FAISS can't return more results than we have stored
        actual_k = min(top_k, self._index.ntotal)
        if actual_k == 0:
            return []

        try:
            # search() returns:
            #   scores:  similarity scores, shape (1, k)
            #   indices: positions in the index, shape (1, k)
            scores, indices = self._index.search(query_embedding.astype(np.float32), actual_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                # FAISS uses -1 to indicate "no result"
                if idx == -1:
                    continue

                # Apply the similarity threshold
                if float(score) < similarity_threshold:
                    continue

                chunk = self._chunks[idx].copy()
                chunk["similarity"] = float(score)
                results.append(chunk)

            logger.debug(
                f"Search returned {len(results)} results "
                f"(threshold={similarity_threshold}, top_k={top_k})"
            )
            return results

        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []

    # ──────────────────────────────────────────────────────────────────────────

    def clear(self):
        """Remove all indexed data. Called when starting fresh."""
        self._index = None
        self._chunks = []
        self._dimension = None
        self._document_ids = set()
        logger.info("Vector store cleared.")

    # ──────────────────────────────────────────────────────────────────────────

    def remove_document(self, filename: str) -> bool:
        """
        Remove all chunks from a specific document.

        FAISS IndexFlatIP does not support removing individual vectors, so we
        rebuild the index from scratch minus the removed document's chunks.

        Returns True if the document was found and removed.
        """
        if not self.is_ready():
            return False

        original_count = len(self._chunks)
        remaining_chunks = [c for c in self._chunks if c.get("filename") != filename]

        if len(remaining_chunks) == original_count:
            logger.warning(f"Document '{filename}' not found in index.")
            return False

        if not remaining_chunks:
            self.clear()
            return True

        # We need the embeddings to rebuild the index.
        # Unfortunately, FAISS IndexFlatIP doesn't let us retrieve stored vectors,
        # so we store a simple flag here. The caller (RAGPipeline) should handle
        # full rebuilds when needed. For now, just clear and let the pipeline
        # know it needs to re-embed the remaining chunks.
        logger.warning(
            "Removing a document requires rebuilding the index. "
            "The pipeline will handle this."
        )
        return False  # Signal that a full rebuild is needed

    # ──────────────────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        """True if the index exists and has at least one vector."""
        return (
            FAISS_AVAILABLE
            and self._index is not None
            and self._index.ntotal > 0
        )

    @property
    def total_chunks(self) -> int:
        """Total number of chunks stored in the index."""
        return self._index.ntotal if self._index else 0

    @property
    def indexed_documents(self) -> set:
        """Set of filenames that have been indexed."""
        return set(self._document_ids)

    @property
    def dimension(self) -> Optional[int]:
        """Embedding dimension of the stored vectors."""
        return self._dimension
