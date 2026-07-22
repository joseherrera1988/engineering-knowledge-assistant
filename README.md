# Engineering Knowledge Assistant: RAG System

## What This Is

This project builds a retrieval-augmented generation (RAG) system that answers technical questions over engineering documentation by combining FAISS-based semantic search with large language model (LLM)-powered reasoning and source attribution.

It focuses on a core production challenge in LLM systems: answering technical questions accurately without hallucination, with citations the reader can verify.

## Why This Matters

Internal engineering knowledge is fragmented across READMEs, schema docs, architecture notes, and code:

1. Engineers spend significant time searching through documentation that ideally should be accessible through a single query.
2. Unrefined LLMs often generate fluent responses but invent APIs, fields, and behaviors that do not exist.
3. An unsubstantiated answer in a technical context can be more detrimental than no answer, as it contains load-bearing misinformation.

This project sets out to answer the question:

**Can a small, well-instrumented retrieval pipeline answer technical questions accurately, with citations, fast enough to be useful?**

## What This Repository Contains

1. A document ingestion pipeline that chunks Markdown by headers, code by function/class boundaries, and plain text by sentences with overlap, preserving source-file and line-range metadata ([`ingestion.py`](ingestion.py)).
2. A FAISS-backed vector store with `sentence-transformers` embeddings (default `all-MiniLM-L6-v2`), an offline-mode hash-based fallback embedder, and a retriever that ranks by vector similarity with an optional term-overlap reranking pass and source filtering ([`vector_store.py`](vector_store.py)). The keyword signal is a lightweight token-overlap score, not a BM25 or lexical index.
3. An LLM agent that performs lightweight tool selection (Q&A vs. summarization vs. SQL generation), supports streaming and multi-turn conversations, and falls back to a deterministic demo mode when no API key is configured ([`rag_agent.py`](rag_agent.py)).
4. A Streamlit UI with four tabs (Q&A, Summarize, SQL Query, Chat), source-attribution display with similarity scores, document upload, and persisted FAISS index reload ([`app.py`](app.py)).
5. A CLI entrypoint with four modes (`init`, `demo`, `ui`, `full`) covering indexing, scripted demo queries, the UI, and the end-to-end flow ([`main.py`](main.py)).
6. A small retrieval evaluation harness with five hand-written gold cases scored by `recall@k` ([`eval_harness.py`](eval_harness.py)).
7. Unit and integration tests across all eight core modules ([`tests/`](tests/)).
8. Long-form technical and user documentation ([`IMPLEMENTATION.md`](IMPLEMENTATION.md), [`QUICKSTART.md`](QUICKSTART.md)).

## Why Retrieval-First

A primary challenge in developing large language model (LLM)-backed systems for technical content is ensuring that answers remain grounded in verifiable sources. This repository does just that: every response references an accessible source, with retrieval quality as the main determinant of system reliability.

## Results

The current evaluation surface remains intentionally limited. The script `eval_harness.py` specifies five manually constructed cases addressing API authentication, architecture, schema, FAISS implementation, and rate-limiting, and evaluates binary recall@k against predefined source substrings.

Run the script locally to generate results for this section based on the specific configuration in use:

```bash
python eval_harness.py
```

This set serves as a smoke-test gold set rather than a comprehensive benchmark. Expanding the evaluation, including reporting per-query and per-stratum metrics, latency, and conducting paired model comparisons, is the primary item on the planned improvements list. The README will present actual result tables after these probes are implemented. 

## Running the System

```bash
# Clone and install dependencies
git clone https://github.com/joseherrera1988/engineering-knowledge-assistant.git
cd engineering-knowledge-assistant
pip install -e .                      # uses pyproject.toml
# or, for a fast path with uv:
# uv sync

# Set your API key (either provider works; OpenAI is the default)
export OPENAI_API_KEY=sk-...          # or
export ANTHROPIC_API_KEY=sk-ant-...

# Index sample documents only
python main.py --mode init

# Run scripted demo queries against the index
python main.py --mode demo

# Launch the Streamlit UI (auto-loads persisted index if present)
python main.py --mode ui
# or directly:
streamlit run app.py

# Full path: index, run demo queries, then launch the UI
python main.py --mode full

# Run the test suite
python -m pytest tests/

# Run the retrieval gold-set harness
python eval_harness.py
```

The first run downloads the `sentence-transformers` model (~80 MB) and generates a small set of sample documents under `sample_docs/`. The persisted FAISS index lives under `rag_index/`; uploaded files land under `uploads/`. Without an API key the agent falls back to deterministic demo responses, which is enough to exercise retrieval and the UI end-to-end.

See [`QUICKSTART.md`](QUICKSTART.md) for a fuller user-facing walkthrough and [`IMPLEMENTATION.md`](IMPLEMENTATION.md) for the technical deep-dive on chunking, retrieval, and agent design.

## Repository Structure

```
engineering-knowledge-assistant/
├── ingestion.py            # document ingestion + smart chunking
├── vector_store.py         # FAISS vector store, embeddings, retriever
├── rag_agent.py            # LLM agent with tool selection (Q&A / summarize / SQL)
├── app.py                  # Streamlit UI (Q&A / Summarize / SQL / Chat tabs)
├── main.py                 # CLI entrypoint (--mode init / demo / ui / full)
├── eval_harness.py         # 5-case retrieval gold set with recall@k
├── uploads.py              # user file upload + incremental ingestion
├── paths.py                # project path constants
├── tests/                  # unit + integration tests for all 8 modules
├── sample_docs/            # generated sample docs (created on first run)
├── rag_index/              # persisted FAISS index (created on first run)
├── uploads/                # user-uploaded documents (created on first run)
├── IMPLEMENTATION.md       # technical deep-dive
├── QUICKSTART.md           # user-facing quickstart
├── pyproject.toml          # deps + mypy/pytest/ruff config (Python 3.11+)
└── README.md
```

## A Note on Sample Documents

The four files generated under `sample_docs/` (an API doc, an architecture note, a database schema, and a small vector-search code sample) are produced by `create_sample_documents()` in [`ingestion.py`](ingestion.py). They exist solely to give a new user something to index and query on a fresh checkout; that is, they are illustrative of, rather than modeled after any real system, and contain no proprietary content. Point the ingester at your own directory to evaluate the system on real documentation.

## License

All materials contained herein are released under the MIT license.

## Author

José Eduardo Herrera. Feedback welcome via GitHub issues.
