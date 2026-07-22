# RAG System - Quick Start Guide

## Installation (2 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key (optional, for Claude integration)
export OPENAI_API_KEY="your-key-here"
# OR
export ANTHROPIC_API_KEY="your-key-here"

# 3. Run the system
python main.py --mode full
```

## What You Get

**4 Python Modules** - Production-ready, modular code;
**Document Ingestion** - Smart chunking for multiple file types;
**Vector Search** - FAISS-based semantic retrieval;
**RAG Agent** - LLM-powered reasoning with multiple tools; and a
**Streamlit UI** - Interactive web interface

## Running Different Modes

### Option 1: Full System (Recommended for first time)
```bash
python main.py --mode full
# Initializes documents + runs demo queries + launches UI
```

### Option 2: Initialize Only
```bash
python main.py --mode init
# Creates sample documents and builds the FAISS index
```

### Option 3: Demo Mode
```bash
python main.py --mode demo
# Runs example queries to test the system
```

### Option 4: Just the UI
```bash
streamlit run app.py
# Launches web interface (documents must be pre-indexed)
```

## Using in Python Code

### Basic Question Answering

```python
from ingestion import DocumentIngester, create_sample_documents
from vector_store import FAISSVectorStore
from rag_agent import RAGAgent

# Create sample documents
create_sample_documents("./docs")

# Ingest and index
ingester = DocumentIngester()
documents = ingester.ingest_directory("./docs")

# Build vector store
vector_store = FAISSVectorStore()
vector_store.add_documents(documents)

# Create agent and query
agent = RAGAgent(vector_store)
response = agent.query("What is the API authentication?")

print(response.answer)
print(f"Confidence: {response.confidence:.1%}")
```

### Advanced: Custom Document Processing

```python
# Process specific files
ingester = DocumentIngester(chunk_size=256, overlap=50)

# Single file
documents = ingester.ingest_file("api_docs.md")

# Directory
documents = ingester.ingest_directory("./technical_docs")

# Access document chunks
for doc in documents:
    print(f"{doc.source} - Chunk {doc.chunk_id}:")
    print(doc.content[:100])
```

### Advanced: Vector Store Management

```python
# Create and build
vs = FAISSVectorStore()
vs.add_documents(documents)

# Search with customization
results = vs.search("your query", k=10)
results = vs.semantic_search_with_reranking("your query", k=10, rerank_top_k=5)

# Persistence
vs.save("./my_index")
vs.load("./my_index")

# Statistics
stats = vs.get_document_stats()
print(f"Documents: {stats['total_documents']}")
print(f"Sources: {stats['sources']}")
```

### Advanced: Multi-turn Conversations

```python
agent = RAGAgent(vector_store)

# First query
resp1 = agent.multi_turn_conversation("What is FAISS?")
print(resp1.answer)

# Follow-up (maintains context)
resp2 = agent.multi_turn_conversation("How is it different from ES?")
print(resp2.answer)

# View history
for msg in agent.conversation_history:
    print(f"{msg['role']}: {msg['content']}")

# Clear for new conversation
agent.clear_history()
```

## Web UI Features

### Q&A Tab
- Ask questions about documents
- Get answers with source attribution
- Adjust number of retrieved documents
- Set confidence threshold

### Summarization Tab
- Summarize documents
- Summarize specific topics
- Generate quick overviews

### SQL Query Tab
- Ask for database queries
- Get SQL with explanations
- Based on schema documentation

### Chat Tab
- Multi-turn conversations
- Maintains conversation context
- Clear history anytime

### Sidebar Controls
- Load/reload documents
- View index statistics
- Adjust search parameters
- Clear conversation history

## Common Use Cases

### Use Case 1: Internal Documentation Search
```python
# Ingest your company docs
docs = ingester.ingest_directory("./company_docs")
vector_store.add_documents(docs)
agent = RAGAgent(vector_store)

# Employees can ask questions
response = agent.query("How do I request time off?")
```

### Use Case 2: Code Documentation Q&A
```python
# Index source code and comments
docs = ingester.ingest_directory("./codebase")
vector_store.add_documents(docs)

# Get code explanations
response = agent.query("How does the auth middleware work?")
```

### Use Case 3: SQL Query Generator
```python
# Index database schema docs
schema_docs = ingester.ingest_file("schema.sql")
schema_docs += ingester.ingest_file("schema_docs.md")
vector_store.add_documents(schema_docs)

# Generate queries
response = agent.query("SQL: Get users who logged in today")
```

### Use Case 4: Knowledge Base
```python
# Index multiple document sources
docs = []
docs += ingester.ingest_directory("./api_docs")
docs += ingester.ingest_directory("./tutorials")
docs += ingester.ingest_directory("./faqs")
vector_store.add_documents(docs)

# Users search for anything
response = agent.query("How to integrate with Stripe?")
```

## Performance Tips

### For Better Retrieval Quality
1. Use larger embedding models (but slower)
   ```python
   vs = FAISSVectorStore("all-mpnet-base-v2")
   ```

2. Increase retrieved context
   ```python
   response = agent.query(text, k=10)  # Get top 10 docs
   ```

3. Use reranking
   ```python
   results = vs.semantic_search_with_reranking(q, k=20, rerank_top_k=5)
   ```

### For Faster Retrieval
1. Use smaller models (faster, less accurate)
   ```python
   vs = FAISSVectorStore("all-MiniLM-L6-v2")
   ```

2. Reduce chunk retrieval
   ```python
   response = agent.query(text, k=3)
   ```

3. Cache frequently accessed results

## Troubleshooting

### Issue: "No documents found"
**Solution:** Run `python main.py --mode init` to create and index sample documents

### Issue: Low retrieval quality
**Solution:** 
- Increase `chunk_size` in DocumentIngester
- Use `semantic_search_with_reranking()` instead of `search()`
- Reduce document chunk count to preserve context

### Issue: Slow performance
**Solution:**
- Use smaller embedding model (all-MiniLM-L6-v2)
- Reduce `k` parameter in search
- Cache FAISS index after building

### Issue: API errors
**Solution:**
- Verify API key is set
- Check rate limits
- Use demo mode which works without API

## System Requirements

- Python 3.8+
- 500MB disk space (for indexes and models)
- 2GB RAM (more for larger documents)
- Internet connection (for first-time downloads)

## Architecture Components

```
Document → Ingestion → Chunking → Embeddings → FAISS Index
                                                    ↓
                                          Semantic Retrieval
                                                    ↓
                                              Reranking
                                                    ↓
                                           Context → Claude
                                                    ↓
                                           Answer + Sources
```

## Next Steps

1. **Explore the UI**: Open `http://localhost:8501` and try different queries
2. **Use Your Own Documents**: Copy your documents to `sample_docs/`
3. **Customize Settings**: Adjust embedding models and chunking in code
4. **Integrate with Apps**: Use the RAGAgent class in your own applications
5. **Deploy to Production**: Use Docker or cloud platforms for deployment

## API Keys Setup

### For OpenAI (recommended for testing)
```bash
export OPENAI_API_KEY="sk-..."
```

### For Anthropic Claude
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### For Development (no API key needed)
The system will use fallback responses when API is unavailable

## Support & More

- **Full Documentation**: See `README.md`
- **Code Examples**: Check files in `/home/claude/`
- **Paper**: [RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- **FAISS Docs**: https://github.com/facebookresearch/faiss

---

**Happy querying!**
