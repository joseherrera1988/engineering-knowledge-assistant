# Engineering Knowledge Assistant: A Retrieval-First RAG System

## What This Is

This project builds a Retrieval-Augmented Generation (RAG) system that answers technical questions over engineering documentation by combining FAISS-based semantic search with LLM-powered reasoning and source attribution.

It focuses on a core production challenge in LLM systems: **answering technical questions accurately without hallucination, with citations the reader can verify**.

## Why This Matters

Internal engineering knowledge is fragmented across READMEs, schema docs, architecture notes, and code:

- Engineers spend siginificant time searching through documentation that ideally should be accessible through a single query.
- Unrefined LLMs often generate fluent responses but invent APIs, fields, and behaviors that do not exist.
- An unsubstantiated answer in a technical context can be more detrimental than no answer, as it contains load-bearing misinformation.

This project sets out to answer the question:

> **Can a small, well-instrumented retrieval pipeline answer technical questions accurately, with citations, fast enough to be useful?**

## What This Repository Contains

- A document ingestion pipeline that chunks Markdown by headers, code by function/class boundaries, and plain text by sentences with overlap, preserving source-file and line-range metadata ([`ingestion.py`](ingestion.py)).
- A FAISS-backed vector store with `sentence-transformers` embeddings (default `all-MiniLM-L6-v2`), an offline-mode hash-based fallback embedder, and a hybrid retriever that combines vector similarity with optional keyword reranking and source filtering ([`vector_store.py`](vector_store.py)).
- An LLM agent that performs lightweight tool selection (Q&A vs. summarization vs. SQL generation), supports streaming and multi-turn conversations, and falls back to a deterministic demo mode when no API key is configured ([`rag_agent.py`](rag_agent.py)).
- A Streamlit UI with four tabs (Q&A, Summarize, SQL Query, Chat), source-attribution display with similarity scores, document upload, and persisted FAISS index reload ([`app.py`](app.py)).
- A CLI entrypoint with four modes (`init`, `demo`, `ui`, `full`) covering indexing, scripted demo queries, the UI, and the end-to-end flow ([`main.py`](main.py)).
- A small retrieval evaluation harness with five hand-written gold cases scored by `recall@k` ([`eval_harness.py`](eval_harness.py)).
- Unit and integration tests across all eight core modules ([`tests/`](tests/)).
- Long-form technical and user documentation ([`IMPLEMENTATION.md`](IMPLEMENTATION.md), [`QUICKSTART.md`](QUICKSTART.md)).

## Why Retrieval-First

The hardest part of building LLM-backed systems for technical content is keeping answers grounded. This repository is built on the principle that **every answer must point back to a source the reader can open**, and that retrieval quality is the upstream lever.

Three concrete commitments fall out of that:

- **Chunk by structure, not by character count alone.** Splitting Markdown on headers and code on function/class boundaries preserves the unit a reader would actually want to see cited.
- **Citations are first-class outputs, not a UI afterthought.** Every `AgentResponse` carries source files, excerpts, and similarity scores; the UI surfaces them next to the answer.
- **Retrieval is independently testable.** `eval_harness.py` scores retrieval directly (recall@k against expected source substrings) so retrieval regressions are visible without going through the LLM.

## Current Status

**Completed:**

- End-to-end pipeline working: ingestion → FAISS index → retrieval (with optional reranking and source filters) → LLM agent → Streamlit UI, with persisted index reload across sessions.
- Tool-selecting agent: Q&A, summarization, and SQL-from-schema, each with source attribution; streaming and multi-turn variants implemented.
- Offline-friendly fallback path: deterministic hash-seeded embeddings and a canned demo response generator so the system runs without network or API keys.
- Five-case retrieval gold set with `recall@k` scoring ([`eval_harness.py`](eval_harness.py)).
- Unit/integration tests across all eight core modules under [`tests/`](tests/), wired up with `pytest`, `mypy`, and `ruff` via [`pyproject.toml`](pyproject.toml).

**Planned:**

- Expand the retrieval gold set well beyond five cases, stratified by query type (factoid, summarization, schema-grounded SQL, multi-source synthesis), and report `recall@k`, `MRR`, and per-stratum accuracy.
- Retrieval ablations: chunk size, chunk overlap, embedding model (`all-MiniLM-L6-v2` vs. `all-mpnet-base-v2`), and reranking on/off, scored on the expanded gold set.
- Latency benchmarks for indexing throughput, retrieval p50/p95, and end-to-end response time.
- Model comparison (`gpt-4o-mini` vs. `claude-haiku-4-5`) on answer quality and citation faithfulness, using the same paired-comparison methodology used in the sibling drive-thru-voice-processing project.
- Index persistence and scaling work: incremental updates, larger corpora, and a measured memory/latency curve.

## Results

The current evaluation surface is intentionally small. `eval_harness.py` defines five hand-authored cases covering API authentication, architecture, schema, FAISS implementation, and rate-limiting, and scores binary `recall@k` against expected source substrings.

Run it locally to populate this section against your own configuration:

```bash
python eval_harness.py
```

This is a smoke-test gold set, not a benchmark. Expanding it (and reporting per-query and per-stratum metrics, latency, and a paired model comparison) is the first item on the **Planned** list above. The README will grow real result tables once those probes exist; until then, claiming numbers here would be theatre.

## Validation

Validation today is limited to two surfaces:

- The unit and integration tests under [`tests/`](tests/) (eight modules covered, run with `pytest`).
- The five-case retrieval harness in [`eval_harness.py`](eval_harness.py).

Held-out sets, run-to-run variance probes, concurrency / rate-limit characterization, and end-to-end answer-quality probes are deliberately *not* claimed here because they don't exist in this repo yet. They are listed under **Planned**, and will get their own subsections once the probes ship — same shape as the validation work in the sibling [drive-thru-voice-processing](https://github.com/joseherrera1988/drive-thru-voice-processing) project.

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
├── vector_store.py         # FAISS vector store, embeddings, hybrid retriever
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
