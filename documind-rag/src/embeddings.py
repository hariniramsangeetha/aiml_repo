"""
src/embeddings.py
=================
PURPOSE: Convert text into numerical vectors (called "embeddings").

WHAT ARE EMBEDDINGS?
  An embedding is a list of numbers (e.g., [0.12, -0.45, 0.78, ...]) that
  represents the MEANING of a piece of text.

  The key idea: texts with similar meanings get similar numbers.

  Example:
    "A dog is running" → [0.12, -0.45, 0.78, ...]
    "A puppy is jogging" → [0.11, -0.43, 0.79, ...]   ← very similar!
    "The stock market fell" → [-0.88, 0.23, -0.12, ...] ← very different

  So when a user asks "What is the main topic?", we convert that question
  into an embedding and find document chunks with similar embeddings.
  This is MUCH smarter than just searching for exact keywords.

MODEL USED:
  We use "all-MiniLM-L6-v2" from the sentence-transformers library.
  - It's tiny (22 MB) and runs fast on a laptop CPU.
  - It produces 384-dimensional embeddings (384 numbers per text).
  - It's accurate enough for document Q&A tasks.

HOW IT CONNECTS:
  PDFProcessor → TextChunker → [THIS FILE] → VectorStore → Retriever
"""

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default model name (can be overridden via .env file)
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ─── Main Class ───────────────────────────────────────────────────────────────

class EmbeddingModel:
    """
    Wraps a Sentence Transformer model and provides text-to-vector conversion.

    Usage example:
        embedder = EmbeddingModel()
        embedder.load()
        vectors = embedder.encode(["Hello world", "Another sentence"])
        # vectors.shape → (2, 384)  – 2 texts, each with 384 numbers
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Args:
            model_name: HuggingFace model ID. Falls back to the .env value
                        or the built-in default.
        """
        self.model_name = (
            model_name
            or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
        )
        self._model = None  # The actual model – loaded lazily when first needed
        self._dimension = None  # Size of each embedding vector

    # ──────────────────────────────────────────────────────────────────────────

    def load(self) -> bool:
        """
        Download and load the embedding model into memory.

        The model is downloaded from HuggingFace the first time (requires
        internet). After that it is cached locally (~22 MB for MiniLM-L6-v2).

        Returns:
            True if the model loaded successfully, False on error.
        """
        if self._model is not None:
            return True  # Already loaded

        try:
            # Import here so the app can start even if sentence-transformers
            # is not installed – we'll show a clear error message instead.
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            logger.info("(First run will download ~22 MB – this is normal)")

            # device="cpu" ensures it works on any laptop without a GPU
            self._model = SentenceTransformer(self.model_name, device="cpu")

            # Find out the embedding dimension by encoding a test string
            test_embedding = self._model.encode(["test"])
            self._dimension = test_embedding.shape[1]

            logger.info(
                f"Model '{self.model_name}' loaded. "
                f"Embedding dimension: {self._dimension}"
            )
            return True

        except ImportError:
            logger.error(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to load embedding model '{self.model_name}': {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────────

    def encode(self, texts: list[str], batch_size: int = 32) -> Optional[np.ndarray]:
        """
        Convert a list of texts into a 2D numpy array of embeddings.

        Args:
            texts:      List of strings to embed.
            batch_size: Process this many texts at once. Lower = less memory.

        Returns:
            numpy array of shape (len(texts), embedding_dimension)
            Returns None if the model isn't loaded or encoding fails.
        """
        if not self.is_loaded():
            logger.error("Embedding model is not loaded. Call .load() first.")
            return None

        if not texts:
            return np.array([])

        # Filter out empty strings (they cause issues)
        valid_texts = [t.strip() if t else "" for t in texts]
        valid_texts = [t if t else "empty" for t in valid_texts]

        try:
            # show_progress_bar=False keeps the console clean during web use
            embeddings = self._model.encode(
                valid_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,  # Return as numpy array (needed by FAISS)
                normalize_embeddings=True,  # Normalize to unit length for cosine similarity
            )
            return embeddings.astype(np.float32)  # FAISS needs float32

        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────────────

    def encode_single(self, text: str) -> Optional[np.ndarray]:
        """
        Encode a single string and return a 1D numpy array.

        This is used for encoding the user's question at query time.

        Args:
            text: The question or text to embed.

        Returns:
            1D numpy array of shape (embedding_dimension,)
            Returns None on failure.
        """
        result = self.encode([text])
        if result is not None and len(result) > 0:
            return result[0]  # Return just the first (and only) row
        return None

    # ──────────────────────────────────────────────────────────────────────────

    def is_loaded(self) -> bool:
        """Returns True if the model is ready to use."""
        return self._model is not None

    @property
    def dimension(self) -> Optional[int]:
        """The number of dimensions in each embedding vector (e.g., 384)."""
        return self._dimension

    @property
    def model_name_display(self) -> str:
        """Human-readable model name for UI display."""
        return self.model_name.split("/")[-1]  # "sentence-transformers/all-MiniLM-L6-v2" → "all-MiniLM-L6-v2"
