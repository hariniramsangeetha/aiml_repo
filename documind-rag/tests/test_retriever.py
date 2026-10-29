"""
tests/test_retriever.py
========================
Tests for the semantic retrieval pipeline.

These tests cover:
- Retrieving relevant chunks for a question
- Returning nothing when no relevant chunks exist
- Handling edge cases (empty index, unloaded model)
- LLM configuration detection

NO API KEY IS NEEDED to run these tests.
The embedding model runs locally on your CPU.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.text_chunker  import TextChunker
from src.vector_store  import VectorStore
from src.retriever     import Retriever
from src.llm_service   import LLMService


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def embedding_model():
    """
    Load the embedding model once for all tests in this module.
    scope="module" means it's loaded only once, not before every single test.
    This saves time since loading takes a few seconds.
    """
    from src.embeddings import EmbeddingModel
    model = EmbeddingModel()
    success = model.load()
    if not success:
        pytest.skip("Could not load embedding model – skipping retrieval tests.")
    return model


@pytest.fixture
def sample_chunks():
    """Return a list of sample chunks about different topics."""
    return [
        {
            "text": "Photosynthesis is the process by which plants convert sunlight into food. "
                    "Chlorophyll in leaves absorbs light energy.",
            "page": 1,
            "filename": "biology.pdf",
            "chunk_id": 0,
        },
        {
            "text": "The water cycle describes the continuous movement of water on Earth. "
                    "It includes evaporation, condensation, and precipitation.",
            "page": 2,
            "filename": "biology.pdf",
            "chunk_id": 1,
        },
        {
            "text": "Machine learning is a subset of artificial intelligence. "
                    "It allows computers to learn from data without being explicitly programmed.",
            "page": 1,
            "filename": "cs.pdf",
            "chunk_id": 2,
        },
        {
            "text": "Neural networks are inspired by the human brain. "
                    "They consist of layers of interconnected nodes called neurons.",
            "page": 2,
            "filename": "cs.pdf",
            "chunk_id": 3,
        },
        {
            "text": "The French Revolution began in 1789. It fundamentally changed "
                    "the political and social structure of France.",
            "page": 5,
            "filename": "history.pdf",
            "chunk_id": 4,
        },
    ]


@pytest.fixture
def loaded_vector_store(embedding_model, sample_chunks):
    """Create a VectorStore pre-loaded with sample chunks."""
    store = VectorStore()
    texts = [c["text"] for c in sample_chunks]
    embeddings = embedding_model.encode(texts)
    assert embeddings is not None, "Embeddings should not be None."
    store.build(embeddings, sample_chunks)
    return store


@pytest.fixture
def retriever(embedding_model, loaded_vector_store):
    """Create a Retriever connected to the loaded vector store."""
    return Retriever(
        embedding_model=embedding_model,
        vector_store=loaded_vector_store,
        top_k=3,
        similarity_threshold=0.2,
    )


# ─── VectorStore Tests ────────────────────────────────────────────────────────

class TestVectorStore:

    def test_build_creates_index(self, embedding_model, sample_chunks):
        """Building a vector store should create a searchable FAISS index."""
        store = VectorStore()
        texts = [c["text"] for c in sample_chunks]
        embeddings = embedding_model.encode(texts)
        success = store.build(embeddings, sample_chunks)
        assert success is True
        assert store.is_ready() is True
        assert store.total_chunks == len(sample_chunks)

    def test_empty_store_not_ready(self):
        """A fresh VectorStore with no data should not be ready."""
        store = VectorStore()
        assert store.is_ready() is False

    def test_search_returns_results(self, loaded_vector_store, embedding_model):
        """Searching should return results when the index has data."""
        query_emb = embedding_model.encode_single("What is photosynthesis?")
        results = loaded_vector_store.search(query_emb, top_k=3, similarity_threshold=0.1)
        assert len(results) > 0, "Should find at least one result."

    def test_search_returns_correct_metadata(self, loaded_vector_store, embedding_model):
        """Each result should have text, page, and filename fields."""
        query_emb = embedding_model.encode_single("plants and sunlight")
        results = loaded_vector_store.search(query_emb, top_k=1, similarity_threshold=0.1)
        if results:
            assert "text"     in results[0]
            assert "page"     in results[0]
            assert "filename" in results[0]
            assert "similarity" in results[0]

    def test_high_threshold_filters_results(self, loaded_vector_store, embedding_model):
        """A very high similarity threshold should return fewer results."""
        query_emb = embedding_model.encode_single("random unrelated xyz query abc")
        results_low  = loaded_vector_store.search(query_emb, top_k=5, similarity_threshold=0.0)
        results_high = loaded_vector_store.search(query_emb, top_k=5, similarity_threshold=0.9)
        assert len(results_high) <= len(results_low), \
            "Higher threshold should return fewer or equal results."

    def test_clear_resets_index(self, embedding_model, sample_chunks):
        """Clearing the store should make it not ready."""
        store = VectorStore()
        texts = [c["text"] for c in sample_chunks]
        embeddings = embedding_model.encode(texts)
        store.build(embeddings, sample_chunks)
        store.clear()
        assert store.is_ready() is False
        assert store.total_chunks == 0


# ─── Retriever Tests ──────────────────────────────────────────────────────────

class TestRetriever:

    def test_retrieve_relevant_chunks(self, retriever):
        """Asking about photosynthesis should return biology-related chunks."""
        results = retriever.retrieve("How do plants make food from sunlight?")
        assert len(results) > 0, "Should find relevant chunks."
        # The top result should be about photosynthesis (check for key word)
        top_text = results[0]["text"].lower()
        assert "photosynthesis" in top_text or "plant" in top_text or "sunlight" in top_text, \
            "Top result should be about plants/photosynthesis."

    def test_retrieve_different_topics_for_different_questions(self, retriever):
        """Different questions should retrieve chunks from different topics."""
        bio_results = retriever.retrieve("What is photosynthesis?")
        cs_results  = retriever.retrieve("What is machine learning?")

        if bio_results and cs_results:
            bio_top = bio_results[0]["text"].lower()
            cs_top  = cs_results[0]["text"].lower()
            # They should come from different content areas
            assert bio_top != cs_top, \
                "Different questions should retrieve different top results."

    def test_retrieve_returns_empty_for_empty_question(self, retriever):
        """An empty question should return an empty list (not an error)."""
        assert retriever.retrieve("") == []
        assert retriever.retrieve("   ") == []

    def test_retrieve_respects_top_k(self, retriever):
        """Retriever should return at most top_k results."""
        results = retriever.retrieve("What is learning?", top_k=2)
        assert len(results) <= 2

    def test_not_ready_without_documents(self, embedding_model):
        """Retriever should not be ready if the vector store is empty."""
        empty_store = VectorStore()
        r = Retriever(embedding_model, empty_store)
        assert r.is_ready() is False
        # Retrieving from an empty store should return empty list (not crash)
        result = r.retrieve("What is DNA?")
        assert result == []

    def test_format_context_contains_source_info(self, retriever):
        """Formatted context should include page and filename info."""
        results = retriever.retrieve("machine learning and AI")
        if results:
            context = retriever.format_context(results)
            assert "Page" in context, "Context should mention page numbers."
            assert ".pdf" in context, "Context should mention the filename."

    def test_deduplicate_removes_near_duplicates(self, embedding_model):
        """Deduplication should remove chunks with >80% word overlap."""
        text_a = "The quick brown fox jumps over the lazy dog"
        text_b = "The quick brown fox jumps over the lazy dog and runs away"  # very similar
        text_c = "Machine learning is a powerful technology for data analysis"

        chunks = [
            {"text": text_a, "page": 1, "filename": "doc.pdf", "chunk_id": 0},
            {"text": text_b, "page": 1, "filename": "doc.pdf", "chunk_id": 1},
            {"text": text_c, "page": 2, "filename": "doc.pdf", "chunk_id": 2},
        ]
        embeddings = embedding_model.encode([c["text"] for c in chunks])
        store = VectorStore()
        store.build(embeddings, chunks)

        r = Retriever(embedding_model, store, top_k=5, similarity_threshold=0.0)
        results = r.retrieve("fox jumps dog")
        # text_a and text_b are very similar – one should be deduplicated
        assert len(results) <= 2, \
            "Near-duplicate chunks should be deduplicated."


# ─── LLM Service Tests ───────────────────────────────────────────────────────

class TestLLMService:

    def test_not_configured_without_api_key(self, monkeypatch):
        """LLMService should report not configured when no API key is set."""
        monkeypatch.setenv("LLM_API_KEY", "")
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        service = LLMService()
        assert service.is_configured() is False

    def test_not_configured_with_placeholder_key(self, monkeypatch):
        """LLMService should reject the placeholder example key."""
        monkeypatch.setenv("LLM_API_KEY", "your_api_key_here")
        service = LLMService()
        assert service.is_configured() is False

    def test_not_configured_when_provider_is_none(self, monkeypatch):
        """Setting LLM_PROVIDER=none should disable the LLM."""
        monkeypatch.setenv("LLM_PROVIDER", "none")
        monkeypatch.setenv("LLM_API_KEY", "sk-real-key-here")
        service = LLMService()
        assert service.is_configured() is False

    def test_generate_answer_without_config_returns_friendly_message(self, monkeypatch):
        """Without config, generate_answer should return a helpful message, not crash."""
        monkeypatch.setenv("LLM_API_KEY", "")
        service = LLMService()
        answer, success = service.generate_answer(
            question="What is photosynthesis?",
            context="Photosynthesis is how plants make food.",
        )
        assert success is False
        assert isinstance(answer, str)
        assert len(answer) > 10, "Should return a non-empty message."
        assert "api key" in answer.lower() or "configure" in answer.lower() or "llm" in answer.lower()

    def test_generate_answer_with_empty_context_returns_message(self, monkeypatch):
        """No context should return a 'not found' message."""
        monkeypatch.setenv("LLM_API_KEY", "")
        service = LLMService()
        answer, success = service.generate_answer(
            question="What is the meaning of life?",
            context="",
        )
        assert success is False

    def test_configuration_info_does_not_expose_api_key(self, monkeypatch):
        """get_configuration_info should NEVER include the API key."""
        monkeypatch.setenv("LLM_API_KEY", "sk-super-secret-key-12345")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        service = LLMService()
        info = service.get_configuration_info()
        info_str = str(info)
        assert "sk-super-secret-key-12345" not in info_str, \
            "API key must not appear in configuration info!"
