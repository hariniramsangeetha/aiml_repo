"""
app.py
======
DocuMind AI – Ask Your Documents
Main Streamlit Web Application

This is the entry point for the web interface.
Run it with: streamlit run app.py

HOW STREAMLIT WORKS:
  Streamlit runs this entire Python file from top to bottom every time
  the user interacts with the page (uploads a file, clicks a button, etc.).
  To avoid re-processing documents on every interaction, we store state
  in `st.session_state` – a dictionary that persists between runs.
"""

import logging
import sys
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── Load environment variables first (before any imports that need them) ──────
load_dotenv()

# ── Set up logging so errors go to the console (not to the user's screen) ────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Add the project root to Python path so 'src' imports work correctly ───────
sys.path.insert(0, str(Path(__file__).parent))

# ── Import our RAG pipeline ───────────────────────────────────────────────────
try:
    from src.rag_pipeline import RAGPipeline
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    IMPORT_ERROR = str(e)
    logger.error(f"Failed to import RAGPipeline: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# This must be the first Streamlit command called.
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="DocuMind AI – Ask Your Documents",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "DocuMind AI – A RAG-powered document Q&A tool built for learning.",
    },
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS – Makes the interface look professional
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── General typography ────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── App title ─────────────────────────────────────────────────── */
.app-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}

.app-subtitle {
    font-size: 1.1rem;
    color: #6b7280;
    margin-bottom: 1.5rem;
}

/* ── Step badges ───────────────────────────────────────────────── */
.step-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #f3f4f6;
    border-radius: 24px;
    padding: 6px 16px;
    font-size: 0.9rem;
    font-weight: 500;
    color: #374151;
    margin-right: 8px;
    margin-bottom: 8px;
}

/* ── Answer box ────────────────────────────────────────────────── */
.answer-box {
    background: linear-gradient(135deg, #f0f9ff 0%, #f5f3ff 100%);
    border: 1px solid #c7d2fe;
    border-left: 4px solid #4f46e5;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin: 1rem 0;
    line-height: 1.7;
}

/* ── Source card ───────────────────────────────────────────────── */
.source-card {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.88rem;
}

.source-meta {
    font-weight: 600;
    color: #4f46e5;
    font-size: 0.8rem;
    margin-bottom: 0.3rem;
}

.source-text {
    color: #374151;
    line-height: 1.5;
    font-style: italic;
}

/* ── Status indicator ──────────────────────────────────────────── */
.status-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 600;
}
.status-ok   { background: #d1fae5; color: #065f46; }
.status-warn { background: #fef3c7; color: #92400e; }
.status-err  { background: #fee2e2; color: #991b1b; }

/* ── Question history ──────────────────────────────────────────── */
.question-bubble {
    background: #4f46e5;
    color: white;
    padding: 0.6rem 1rem;
    border-radius: 18px 18px 4px 18px;
    display: inline-block;
    max-width: 85%;
    font-weight: 500;
    margin: 0.3rem 0;
}

/* ── Upload area ───────────────────────────────────────────────── */
.upload-hint {
    text-align: center;
    padding: 2rem;
    background: #f8fafc;
    border: 2px dashed #c7d2fe;
    border-radius: 12px;
    color: #6b7280;
}

/* ── Sidebar ───────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #fafafa;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# session_state variables persist between Streamlit re-runs (page interactions).
# ══════════════════════════════════════════════════════════════════════════════

def init_session_state():
    """Initialize all session state variables with safe defaults."""
    defaults = {
        "pipeline":        None,      # The RAGPipeline instance
        "chat_history":    [],        # List of (question, QueryResult) tuples
        "processing":      False,     # True while a document is being processed
        "error_message":   "",        # Last error to display to the user
        "uploaded_docs":   [],        # List of filenames uploaded this session
        "active_doc":      None,      # Currently selected document (for display)
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def get_pipeline() -> "RAGPipeline":
    """Get or create the RAGPipeline (one per session)."""
    if st.session_state.pipeline is None:
        if not PIPELINE_AVAILABLE:
            st.error(
                f"❌ **Import Error:** Could not load the RAG pipeline. "
                f"Please run `pip install -r requirements.txt`.\n\n"
                f"Error: `{IMPORT_ERROR}`"
            )
            st.stop()
        st.session_state.pipeline = RAGPipeline()
    return st.session_state.pipeline


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    """Render the left sidebar with document list and info panels."""
    with st.sidebar:
        st.markdown("## 📂 Documents")

        # ── Upload widget ─────────────────────────────────────────────────────
        uploaded_file = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            accept_multiple_files=False,
            help="Upload any text-based PDF (not scanned/image PDFs).",
            key="pdf_uploader",
        )

        if uploaded_file is not None:
            handle_document_upload(uploaded_file)

        # ── Document list ─────────────────────────────────────────────────────
        pipeline = get_pipeline()
        docs = pipeline.get_documents()

        if docs:
            st.markdown("### Indexed Documents")
            for filename, doc_info in docs.items():
                col1, col2 = st.columns([4, 1])
                with col1:
                    is_active = (st.session_state.active_doc == filename)
                    label = f"✅ {filename}" if is_active else f"📄 {filename}"
                    if st.button(label, key=f"select_{filename}", use_container_width=True):
                        st.session_state.active_doc = filename
                with col2:
                    if st.button("🗑️", key=f"remove_{filename}", help=f"Remove {filename}"):
                        handle_remove_document(filename, pipeline)

                st.caption(
                    f"Pages: {doc_info.page_count} | Chunks: {doc_info.chunk_count}"
                )
                if doc_info.warning:
                    st.warning(doc_info.warning, icon="⚠️")
        else:
            st.info("No documents uploaded yet.")

        # ── AI Status ─────────────────────────────────────────────────────────
        st.divider()
        st.markdown("### ⚙️ AI Configuration")
        llm_info = pipeline.get_llm_info()

        if llm_info["configured"]:
            st.markdown(
                f'<span class="status-pill status-ok">✅ LLM Ready</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"Provider: **{llm_info['provider']}**")
            st.caption(f"Model: `{llm_info['model']}`")
        else:
            st.markdown(
                '<span class="status-pill status-warn">⚠️ LLM Not Configured</span>',
                unsafe_allow_html=True,
            )
            st.caption(
                "AI answers disabled. Copy `.env.example` → `.env` "
                "and add your API key to enable them."
            )

        # ── How RAG works ─────────────────────────────────────────────────────
        st.divider()
        with st.expander("❓ How does RAG work?", expanded=False):
            st.markdown("""
**RAG = Retrieval-Augmented Generation**

Instead of sending your entire document to an AI (slow & expensive),
RAG works in two steps:

1. **Retrieve** – Find the 3-5 most relevant passages from your document
   using semantic search (meaning-based, not just keywords).

2. **Generate** – Send only those passages + your question to an LLM.
   The LLM writes a clear answer based on what's in those passages.

This makes answers:
- ✅ Grounded in your actual document
- ✅ More accurate (no hallucinations)
- ✅ Faster and cheaper than sending the whole document
            """)

        # ── Clear chat ────────────────────────────────────────────────────────
        st.divider()
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    return uploaded_file


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT UPLOAD HANDLER
# ══════════════════════════════════════════════════════════════════════════════

def handle_document_upload(uploaded_file):
    """Process a newly uploaded PDF file."""
    filename = uploaded_file.name
    pipeline = get_pipeline()

    # Skip if this file is already indexed
    if filename in pipeline.get_documents():
        st.sidebar.success(f"✅ '{filename}' is already indexed.")
        return

    # Show a spinner while processing
    with st.sidebar:
        with st.spinner(f"Processing '{filename}'..."):
            try:
                file_bytes = uploaded_file.read()
                doc_info, error = pipeline.process_document(file_bytes, filename)

                if error:
                    st.error(f"❌ {error}")
                    logger.error(f"Document processing error for '{filename}': {error}")
                    return

                st.success(
                    f"✅ '{filename}' ready!\n"
                    f"{doc_info.page_count} pages, {doc_info.chunk_count} chunks."
                )
                st.session_state.active_doc = filename
                if filename not in st.session_state.uploaded_docs:
                    st.session_state.uploaded_docs.append(filename)

                if doc_info.warning:
                    st.warning(doc_info.warning)

            except Exception as e:
                logger.exception(f"Unexpected error processing '{filename}'")
                st.error(
                    "An unexpected error occurred while processing the document. "
                    "Please try again with a different file."
                )


def handle_remove_document(filename: str, pipeline):
    """Handle the remove document button."""
    success, msg = pipeline.remove_document(filename)
    if success:
        st.session_state.active_doc = None
        st.success(msg)
    else:
        # Clear everything and ask the user to refresh
        pipeline.vector_store.clear()
        del pipeline._documents[filename]
        if filename in st.session_state.uploaded_docs:
            st.session_state.uploaded_docs.remove(filename)
        if st.session_state.active_doc == filename:
            remaining = list(pipeline.get_documents().keys())
            st.session_state.active_doc = remaining[0] if remaining else None
        st.warning("Document removed. The index has been updated.")
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# WELCOME / HEADER SECTION
# ══════════════════════════════════════════════════════════════════════════════

def render_header():
    """Render the top header section."""
    st.markdown(
        '<div class="app-title">🧠 DocuMind AI</div>'
        '<div class="app-subtitle">Ask Your Documents – Powered by RAG</div>',
        unsafe_allow_html=True,
    )

    # Three-step guide
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<span class="step-badge">📤 Step 1: Upload a PDF</span>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<span class="step-badge">💬 Step 2: Ask a Question</span>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<span class="step-badge">🔍 Step 3: Get a Sourced Answer</span>',
            unsafe_allow_html=True,
        )

    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# NO DOCUMENTS STATE
# ══════════════════════════════════════════════════════════════════════════════

def render_empty_state():
    """Show a friendly message when no documents are loaded."""
    st.markdown("""
<div class="upload-hint">
    <h3>📄 Upload a PDF to get started</h3>
    <p>Use the <strong>Upload a PDF</strong> button in the left sidebar.<br>
    Then ask any question about your document in the chat below.</p>
    <br>
    <p>💡 <em>Tip: Works best with text-based PDFs like research papers, textbooks, and reports.<br>
    Scanned/image PDFs are not supported.</em></p>
</div>
    """, unsafe_allow_html=True)

    with st.expander("📚 Example use cases", expanded=True):
        st.markdown("""
- **Research paper**: "What is the main contribution of this paper?"
- **Textbook chapter**: "Explain the concept of photosynthesis."
- **Legal document**: "What are the termination conditions?"
- **Company report**: "What was the revenue in Q3?"
- **Study notes**: "List all the key terms mentioned."
        """)


# ══════════════════════════════════════════════════════════════════════════════
# QUESTION & ANSWER SECTION
# ══════════════════════════════════════════════════════════════════════════════

def render_qa_section(pipeline):
    """Render the question input and chat history."""

    # ── Chat history (show existing Q&A pairs) ────────────────────────────────
    if st.session_state.chat_history:
        st.markdown("### 💬 Conversation")
        for i, (q, result) in enumerate(st.session_state.chat_history):
            render_qa_pair(q, result, index=i)
        st.divider()

    # ── Question input ────────────────────────────────────────────────────────
    st.markdown("### ❓ Ask a Question")

    # Show example questions for first-time users
    if not st.session_state.chat_history:
        with st.expander("💡 Example questions to try", expanded=True):
            example_cols = st.columns(2)
            examples = [
                "What is the main topic of this document?",
                "Summarize the key points.",
                "What are the main conclusions?",
                "List the important terms or definitions.",
                "What problem does this document address?",
                "What methods or approaches are described?",
            ]
            for i, ex in enumerate(examples):
                col = example_cols[i % 2]
                with col:
                    if st.button(f"💬 {ex}", key=f"example_{i}", use_container_width=True):
                        run_question(ex, pipeline)
                        st.rerun()

    # ── Text input ────────────────────────────────────────────────────────────
    with st.form(key="question_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            question = st.text_input(
                "Your question",
                placeholder="Ask anything about your document...",
                label_visibility="collapsed",
                key="question_input",
            )
        with col_btn:
            submit = st.form_submit_button(
                "Ask ➤",
                use_container_width=True,
                type="primary",
            )

    if submit and question and question.strip():
        with st.spinner("🔍 Searching document and generating answer..."):
            run_question(question.strip(), pipeline)
        st.rerun()
    elif submit and not (question and question.strip()):
        st.warning("Please type a question before clicking Ask.")


# ──────────────────────────────────────────────────────────────────────────────

def run_question(question: str, pipeline):
    """Execute the RAG pipeline for a question and store the result."""
    try:
        result = pipeline.answer_question(question)
        st.session_state.chat_history.append((question, result))
    except Exception as e:
        logger.exception(f"Error answering question: '{question}'")
        st.error(
            "An unexpected error occurred while processing your question. "
            "Please try again."
        )


# ──────────────────────────────────────────────────────────────────────────────

def render_qa_pair(question: str, result, index: int):
    """Render a single question-answer pair with sources."""

    # Question bubble
    st.markdown(
        f'<div style="text-align:right; margin-bottom:0.5rem;">'
        f'<span class="question-bubble">❓ {question}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Answer box
    if result.llm_answered:
        icon = "🤖"
        label = "AI Answer (based on your document)"
    elif result.no_context_found:
        icon = "🔍"
        label = "No Matching Content Found"
    else:
        icon = "⚙️"
        label = "System Message"

    st.markdown(
        f'<div class="answer-box">'
        f'<strong>{icon} {label}</strong><br><br>'
        f'{result.answer}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Source references
    if result.has_sources():
        with st.expander(
            f"📖 View {len(result.sources)} source excerpt(s)",
            expanded=(index == len(st.session_state.chat_history) - 1),
        ):
            for j, source in enumerate(result.sources):
                filename = source.get("filename", "Unknown")
                page     = source.get("page", "?")
                text     = source.get("text", "")
                sim      = source.get("similarity", 0.0)

                # Truncate long excerpts for readability
                display_text = text[:350] + "..." if len(text) > 350 else text

                st.markdown(
                    f'<div class="source-card">'
                    f'<div class="source-meta">'
                    f'📄 {filename} — Page {page} &nbsp;|&nbsp; '
                    f'Relevance: {sim:.0%}'
                    f'</div>'
                    f'<div class="source-text">{display_text}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    elif not result.no_context_found and not result.llm_answered:
        pass  # System messages don't need source references


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT INFO PANEL
# ══════════════════════════════════════════════════════════════════════════════

def render_document_info(pipeline):
    """Show a summary of the currently active document."""
    docs = pipeline.get_documents()
    if not docs:
        return

    active = st.session_state.active_doc
    if active and active in docs:
        doc = docs[active]
        with st.expander(f"📄 Active Document: {active}", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("Pages", doc.page_count)
            c2.metric("Text Chunks", doc.chunk_count)
            c3.metric("Indexed Docs", len(docs))
            if doc.warning:
                st.warning(doc.warning)
    elif docs:
        # Auto-select the first document
        st.session_state.active_doc = list(docs.keys())[0]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Main application entry point."""
    # Render the sidebar (also handles file upload logic inside)
    render_sidebar()

    # Render the main content area
    render_header()

    pipeline = get_pipeline()

    if not pipeline.has_documents():
        render_empty_state()
    else:
        render_document_info(pipeline)
        render_qa_section(pipeline)


# ─── Run the app ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
else:
    # Streamlit calls this file as a module – run main() either way
    main()
