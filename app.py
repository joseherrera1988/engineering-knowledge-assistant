"""
Streamlit UI for RAG System
Interactive interface for document search, Q&A, and summarization
"""

import os
import sys
from pathlib import Path

import streamlit as st

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion import DocumentIngester, create_sample_documents
from rag_agent import RAGAgent
from uploads import ingest_uploaded_files
from vector_store import FAISSVectorStore

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="RAG System - Document Search & Analysis",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
        font-weight: 600;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .answer-box {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================


def init_session_state():
    """Initialize session state variables"""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "agent" not in st.session_state:
        st.session_state.agent = None

    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    if "index_stats" not in st.session_state:
        st.session_state.index_stats = {}

    if not st.session_state.documents_loaded:
        try_load_existing_index()


def try_load_existing_index() -> bool:
    """Load a previously saved FAISS index if one exists on disk."""
    import paths

    index_file = paths.INDEX_DIR / "index.faiss"
    if not index_file.exists():
        return False

    try:
        vector_store = FAISSVectorStore()
        vector_store.load(str(paths.INDEX_DIR))
    except Exception:
        return False

    st.session_state.vector_store = vector_store
    st.session_state.agent = RAGAgent(vector_store)
    st.session_state.documents_loaded = True
    st.session_state.index_stats = vector_store.get_document_stats()
    return True


init_session_state()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def load_documents():
    """Load and index documents"""
    with st.spinner("Loading and indexing documents..."):
        # Create sample documents if they don't exist
        from paths import SAMPLE_DOCS_DIR

        docs_dir = str(SAMPLE_DOCS_DIR)
        if not os.path.exists(docs_dir) or not os.listdir(docs_dir):
            create_sample_documents(docs_dir)

        # Ingest documents
        ingester = DocumentIngester(chunk_size=512, overlap=100)
        documents = ingester.ingest_directory(docs_dir)

        if not documents:
            st.error("No documents found to index.")
            return False

        # Build vector store
        vector_store = FAISSVectorStore()
        vector_store.add_documents(documents)

        # Initialize agent
        agent = RAGAgent(vector_store)

        # Store in session
        st.session_state.vector_store = vector_store
        st.session_state.agent = agent
        st.session_state.documents_loaded = True
        st.session_state.index_stats = vector_store.get_document_stats()

        st.success(
            f"✓ Loaded {len(documents)} document chunks from {len(st.session_state.index_stats['sources'])} sources"
        )
        return True


def format_response(response):
    """Format agent response for display"""
    st.markdown("### Answer")
    st.markdown(f'<div class="answer-box">{response.answer}</div>', unsafe_allow_html=True)

    # Display sources
    if response.sources:
        st.markdown("### 📎 Sources")

        cols = st.columns([1, 1])

        with cols[0]:
            st.metric("Confidence", f"{response.confidence:.1%}")

        with cols[1]:
            st.metric("Tool Used", response.tool_used.replace("_", " ").title())

        st.markdown("#### Retrieved Documents")
        for i, source in enumerate(response.sources, 1):
            with st.expander(
                f"Source {i}: {Path(source['file']).name} (Similarity: {source['similarity']})"
            ):
                st.markdown(f"**File:** `{source['file']}`")
                st.markdown(f"**Chunk ID:** {source['chunk_id']}")
                st.markdown("**Excerpt:**")
                st.text(source["excerpt"])


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("⚙️ Configuration")

    # System Status
    st.markdown("### System Status")

    if st.session_state.documents_loaded:
        st.success("✓ Documents indexed")

        with st.expander("📊 Index Statistics"):
            stats = st.session_state.index_stats
            st.metric("Total Documents", stats["total_documents"])
            st.metric("Indexed Chunks", stats["total_indexed"])
            st.metric("Unique Sources", stats["unique_sources"])
            st.metric("Embedding Dimension", stats["embedding_dimension"])

            if stats["sources"]:
                st.markdown("**Sources:**")
                for source in stats["sources"]:
                    st.write(f"- {Path(source).name}")
    else:
        st.warning("⚠️ No documents indexed yet")

    # Document Loading
    st.markdown("### 📚 Document Management")

    if st.button("🔄 Load/Reload Documents", use_container_width=True):
        load_documents()

    uploaded = st.file_uploader(
        "Upload your own documents",
        type=["md", "txt", "py", "js", "sql"],
        accept_multiple_files=True,
        key="file_uploader",
    )
    if uploaded:
        from paths import INDEX_DIR, UPLOADS_DIR

        store, added = ingest_uploaded_files(uploaded, UPLOADS_DIR, st.session_state.vector_store)
        if added:
            store.save(str(INDEX_DIR))
            st.session_state.vector_store = store
            st.session_state.agent = RAGAgent(store)
            st.session_state.documents_loaded = True
            st.session_state.index_stats = store.get_document_stats()
            st.success(f"✓ Added {added} chunks from {len(uploaded)} file(s)")
        else:
            st.warning("No content could be extracted from the uploaded files.")

    # Search Parameters
    st.markdown("### 🔍 Search Parameters")

    retrieval_k = st.slider(
        "Number of documents to retrieve:",
        min_value=1,
        max_value=10,
        value=5,
        help="Higher values retrieve more context but may reduce precision",
    )

    confidence_threshold = st.slider(
        "Confidence threshold:",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Minimum similarity score for results",
    )

    # Clear History
    st.markdown("### 💬 Conversation")

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.conversation_history = []
        st.session_state.agent.clear_history() if st.session_state.agent else None
        st.success("History cleared!")


# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("📚 RAG System - Document Search & Analysis")

st.markdown("""
This system combines **semantic search** with **LLM-powered reasoning** to answer questions about your documents.

**Features:**
- 🔍 Semantic search using embeddings and FAISS
- 💡 Intelligent Q&A with source attribution
- 📝 Document summarization
- 🗄️ SQL query generation
- 💬 Multi-turn conversations
""")

# Load documents if not already loaded
if not st.session_state.documents_loaded:
    if st.button("📥 Initialize System - Load Documents", use_container_width=True, type="primary"):
        load_documents()
else:
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["❓ Q&A", "📋 Summarize", "🗄️ SQL Query", "💬 Chat"])

    # ============================================================================
    # TAB 1: QUESTION ANSWERING
    # ============================================================================

    with tab1:
        st.header("Question Answering")
        st.markdown(
            "Ask questions about your documents. The system will retrieve relevant passages and answer based on the content."
        )

        question = st.text_input(
            "Ask a question:",
            placeholder="e.g., What authentication methods are supported?",
            key="qa_question",
        )

        if st.button("Search & Answer", use_container_width=True, type="primary"):
            if question:
                response = st.session_state.agent.query(question, k=retrieval_k)
                format_response(response)
            else:
                st.warning("Please enter a question")

    # ============================================================================
    # TAB 2: SUMMARIZATION
    # ============================================================================

    with tab2:
        st.header("Document Summarization")
        st.markdown("Get concise summaries of documents or specific topics.")

        col1, col2 = st.columns([3, 1])

        with col1:
            summary_query = st.text_input(
                "What would you like summarized?",
                placeholder="e.g., Summarize the API documentation",
                key="summary_query",
            )

        with col2:
            st.markdown("###")
            if st.button("Summarize", use_container_width=True, type="primary"):
                if summary_query:
                    full_query = f"Summarize the following: {summary_query}"
                    response = st.session_state.agent.query(full_query, k=retrieval_k)
                    format_response(response)
                else:
                    st.warning("Please enter what to summarize")

    # ============================================================================
    # TAB 3: SQL GENERATION
    # ============================================================================

    with tab3:
        st.header("SQL Query Generation")
        st.markdown(
            "Describe what data you want, and the system will generate SQL queries based on your schema documentation."
        )

        sql_request = st.text_input(
            "What query do you need?",
            placeholder="e.g., Get recent documents with their chunk count",
            key="sql_request",
        )

        if st.button("Generate SQL", use_container_width=True, type="primary"):
            if sql_request:
                full_query = f"Generate SQL for: {sql_request}"
                response = st.session_state.agent.query(full_query, k=retrieval_k)
                format_response(response)
            else:
                st.warning("Please describe your query")

    # ============================================================================
    # TAB 4: MULTI-TURN CONVERSATION
    # ============================================================================

    with tab4:
        st.header("Conversation")
        st.markdown("Have a multi-turn conversation about your documents.")

        # Display conversation history
        if st.session_state.conversation_history:
            for message in st.session_state.conversation_history:
                if message["role"] == "user":
                    st.write(f"**You:** {message['content']}")
                else:
                    st.write(f"**Assistant:** {message['content']}")

            st.divider()

        # New message input
        user_message = st.text_input(
            "Your message:",
            placeholder="Ask a follow-up question or continue the conversation...",
            key="chat_message",
        )

        if st.button("Send", use_container_width=True, type="primary"):
            if user_message:
                response = st.session_state.agent.multi_turn_conversation(user_message)
                st.rerun()
            else:
                st.warning("Please enter a message")


# ============================================================================
# FOOTER
# ============================================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🎯 Features")
    st.markdown("""
    - FAISS vector indexing
    - Semantic search
    - Claude API integration
    - Source attribution
    """)

with col2:
    st.markdown("### 🔧 Technology Stack")
    st.markdown("""
    - sentence-transformers
    - FAISS
    - OpenAI SDK
    - Streamlit
    """)

with col3:
    st.markdown("### 📖 Use Cases")
    st.markdown("""
    - Document Q&A
    - Knowledge base search
    - Code documentation
    - Technical analysis
    """)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>RAG System v1.0 | Powered by Claude + FAISS</p>",
    unsafe_allow_html=True,
)
