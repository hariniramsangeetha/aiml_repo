"""
src/rag_pipeline.py
===================
PURPOSE: The central "brain" that connects all modules into one RAG pipeline.

THE FULL PIPELINE:
  User uploads PDF
      ↓
  PDFProcessor  →  extract pages (text + page numbers)
      ↓
  TextChunker   →  split pages into small overlapping chunks
      ↓
  EmbeddingModel → convert chunks to vectors (numbers)
      ↓
  VectorStore   →  store vectors in FAISS index
      ↓
  (User asks a question)
      ↓
  EmbeddingModel → convert question to vector
      ↓
  VectorStore (FAISS) → find most similar chunk vectors
      ↓
  Retriever     →  get the matching text chunks + deduplicate
      ↓
  LLMService    →  generate an answer from question + retrieved context
      ↓
  Display       →  show answer + source references to user

This file is the "conductor" that makes all these steps happen in order.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv

# Import all our modules
from src.pdf_processor  import PDFProcessor
from src.text_chunker   import TextChunker
from src.embeddings     import EmbeddingModel
from src.vector_store   import VectorStore
from src.retriever      import Retriever
from src.llm_service    import LLMService

logger = logging.getLogger(__name__)

# Load environment variables from .env file as early as possible
load_dotenv()


# ─── Result Classes ───────────────────────────────────────────────────────────

class DocumentInfo:
    """Stores info about a processed document (for display in the UI)."""
    def __init__(self, filename: str, page_count: int, chunk_count: int, warning: str = ""):
        self.filename    = filename
        self.page_count  = page_count
        self.chunk_count = chunk_count
        self.warning     = warning  # e.g., scanned PDF warning

    def __repr__(self):
        return f"DocumentInfo('{self.filename}', pages={self.page_count}, chunks={self.chunk_count})"


class QueryResult:
    """Stores the result of a question-answering query."""
    def __init__(
        self,
        question:        str,
        answer:          str,
        sources:         list[dict],
        llm_answered:    bool,
        no_context_found: bool = False,
    ):
        self.question         = question
        self.answer           = answer          # The generated or fallback answer text
        self.sources          = sources         # List of retrieved chunk dicts
        self.llm_answered     = llm_answered    # True if LLM generated the answer
        self.no_context_found = no_context_found  # True if FAISS found nothing relevant

    def has_sources(self) -> bool:
        return len(self.sources) > 0


# ─── Main Pipeline Class ──────────────────────────────────────────────────────

class RAGPipeline:
    """
    Orchestrates the complete Retrieval-Augmented Generation workflow.

    One instance of this class is created per Streamlit session and stored
    in st.session_state so that the FAISS index is preserved between questions.
    """

    def __init__(self):
        """Initialize all pipeline components."""
        # Read configuration from environment variables
        chunk_size    = int(os.getenv("CHUNK_SIZE",    500))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 100))
        top_k         = int(os.getenv("TOP_K_RESULTS",  5))
        threshold     = float(os.getenv("SIMILARITY_THRESHOLD", 0.3))
        max_mb        = float(os.getenv("MAX_UPLOAD_SIZE_MB",   50))

        # Create each component
        self.pdf_processor  = PDFProcessor(max_file_size_mb=max_mb)
        self.text_chunker   = TextChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.embedding_model = EmbeddingModel()
        self.vector_store   = VectorStore()
        self.llm_service    = LLMService()

        # Retriever is created after embedding_model and vector_store are set up
        self.retriever = Retriever(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            top_k=top_k,
            similarity_threshold=threshold,
        )

        # Track processed documents (filename → DocumentInfo)
        self._documents: dict[str, DocumentInfo] = {}

        # Track whether the embedding model has been loaded
        self._embedding_model_loaded = False

    # ──────────────────────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        """
        Load the embedding model into memory.

        This is called once when the app starts (or on first document upload).
        It downloads the model on first run (~22 MB, needs internet once).

        Returns True if the model is ready.
        """
        if self._embedding_model_loaded:
            return True

        success = self.embedding_model.load()
        self._embedding_model_loaded = success

        if not success:
            logger.error("Failed to load embedding model.")
        else:
            logger.info("Embedding model loaded and ready.")

        return success

    # ──────────────────────────────────────────────────────────────────────────

    def process_document(
        self,
        file_bytes: bytes,
        filename:   str,
    ) -> tuple[Optional[DocumentInfo], str]:
        """
        Process an uploaded PDF: extract → chunk → embed → index.

        Args:
            file_bytes: Raw bytes of the uploaded PDF.
            filename:   Original filename (shown in UI and stored in metadata).

        Returns:
            (DocumentInfo, error_message)
            - DocumentInfo is None if processing failed.
            - error_message is empty string on success.
        """
        logger.info(f"Processing document: '{filename}'")

        # ── Step 1: Make sure the embedding model is loaded ──────────────────
        if not self.initialize():
            return None, "Failed to load the embedding model. Please check the logs."

        # ── Step 2: Extract text from the PDF ────────────────────────────────
        pages, warning = self.pdf_processor.extract_pages_from_bytes(file_bytes, filename)

        if not pages:
            return None, warning or "Could not extract any text from the PDF."

        # Filter out pages with no text
        text_pages = [p for p in pages if p.get("text", "").strip()]
        if not text_pages:
            return None, (
                warning or "The PDF has no extractable text. "
                "It may be a scanned image document. OCR is not supported."
            )

        page_count = len(pages)

        # ── Step 3: Split text into chunks ────────────────────────────────────
        chunks = self.text_chunker.chunk_pages(text_pages, filename=filename)
        if not chunks:
            return None, "Document processed but produced no text chunks. It may be too short."

        # ── Step 4: Generate embeddings for all chunks ────────────────────────
        chunk_texts = [c["text"] for c in chunks]
        embeddings = self.embedding_model.encode(chunk_texts)

        if embeddings is None or len(embeddings) == 0:
            return None, "Failed to generate embeddings for the document."

        # ── Step 5: Add to the FAISS vector index ────────────────────────────
        if self.vector_store.is_ready():
            # If we already have documents indexed, add to the existing index
            success = self.vector_store.add(embeddings, chunks)
        else:
            # First document – build a fresh index
            success = self.vector_store.build(embeddings, chunks)

        if not success:
            return None, "Failed to build the search index."

        # ── Step 6: Store document info for UI display ───────────────────────
        doc_info = DocumentInfo(
            filename=filename,
            page_count=page_count,
            chunk_count=len(chunks),
            warning=warning,
        )
        self._documents[filename] = doc_info

        logger.info(
            f"'{filename}' indexed: {page_count} pages, {len(chunks)} chunks."
        )
        return doc_info, ""

    # ──────────────────────────────────────────────────────────────────────────

    def answer_question(self, question: str) -> QueryResult:
        """
        Run the full RAG query pipeline for a user question.

        Args:
            question: The user's natural language question.

        Returns:
            A QueryResult containing the answer, sources, and metadata.
        """
        if not question or not question.strip():
            return QueryResult(
                question=question,
                answer="Please enter a question.",
                sources=[],
                llm_answered=False,
            )

        if not self.vector_store.is_ready():
            return QueryResult(
                question=question,
                answer="Please upload a document first before asking questions.",
                sources=[],
                llm_answered=False,
            )

        # ── Step 1: Retrieve relevant chunks ─────────────────────────────────
        sources = self.retriever.retrieve(question)

        if not sources:
            return QueryResult(
                question=question,
                answer=(
                    "I couldn't find enough relevant information in the document "
                    "to answer that question confidently. "
                    "Try rephrasing your question or asking about a different topic covered in the document."
                ),
                sources=[],
                llm_answered=False,
                no_context_found=True,
            )

        # ── Step 2: Format context for the LLM ───────────────────────────────
        context = self.retriever.format_context(sources)

        # ── Step 3: Generate answer with the LLM ─────────────────────────────
        answer, llm_success = self.llm_service.generate_answer(
            question=question,
            context=context,
        )

        return QueryResult(
            question=question,
            answer=answer,
            sources=sources,
            llm_answered=llm_success,
        )

    # ──────────────────────────────────────────────────────────────────────────

    def remove_document(self, filename: str) -> tuple[bool, str]:
        """
        Remove a document from the index and re-build from remaining docs.

        Returns (success, message).
        """
        if filename not in self._documents:
            return False, f"Document '{filename}' is not indexed."

        # Remove from our tracking dict
        del self._documents[filename]

        if not self._documents:
            # No documents left – just clear everything
            self.vector_store.clear()
            return True, "All documents removed. Index cleared."

        # We need to rebuild the index from scratch with the remaining documents.
        # This is a limitation of FAISS IndexFlatIP (no delete support).
        # In practice, re-indexing a few documents is fast.
        return False, (
            "Rebuilding the index after deletion is not supported in this version. "
            "Please refresh the page to start fresh."
        )

    # ──────────────────────────────────────────────────────────────────────────

    def get_documents(self) -> dict[str, DocumentInfo]:
        """Return the dict of all indexed documents."""
        return self._documents

    def has_documents(self) -> bool:
        """True if at least one document has been successfully indexed."""
        return bool(self._documents) and self.vector_store.is_ready()

    def get_llm_info(self) -> dict:
        """Return display-safe LLM configuration info."""
        return self.llm_service.get_configuration_info()

    def is_embedding_model_loaded(self) -> bool:
        return self._embedding_model_loaded
