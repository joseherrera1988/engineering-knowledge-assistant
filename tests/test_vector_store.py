from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from ingestion import Document
from vector_store import (
    FAISSVectorStore,
    HybridRetriever,
    SimpleEmbeddingModel,
)


def _docs() -> list[Document]:
    return [
        Document(
            content="FAISS is a library for vector similarity search.", source="a.md", chunk_id=0
        ),
        Document(content="Bearer tokens authenticate API requests.", source="b.md", chunk_id=1),
        Document(
            content="The database schema includes a documents table.", source="c.md", chunk_id=2
        ),
    ]


def _store() -> FAISSVectorStore:
    s = FAISSVectorStore(model=SimpleEmbeddingModel())
    return s


def test_constructor_accepts_injected_model() -> None:
    """An embedding model can be injected, so tests need no SentenceTransformer download."""
    m = SimpleEmbeddingModel()
    s = FAISSVectorStore(model=m)
    assert s.model is m
    assert s.embedding_dim == 384


def test_simple_embedding_is_deterministic() -> None:
    m = SimpleEmbeddingModel()
    a = m.encode("hello")
    b = m.encode("hello")
    assert (a == b).all()


def test_simple_embedding_list_shape() -> None:
    m = SimpleEmbeddingModel()
    out = m.encode(["x", "y", "z"])
    assert out.shape == (3, 384)


def test_encode_does_not_mutate_global_rng() -> None:
    """encode() must be deterministic without touching global numpy RNG state."""
    m = SimpleEmbeddingModel()
    np.random.seed(12345)
    before = np.random.rand()
    np.random.seed(12345)
    m.encode(["some text", "other text"])
    after = np.random.rand()
    assert before == after


def test_add_and_search_returns_results() -> None:
    s = _store()
    s.add_documents(_docs())
    results = s.search("vector similarity FAISS", k=2)
    assert len(results) == 2
    assert results[0].similarity_score >= results[1].similarity_score


def test_search_empty_store_returns_empty() -> None:
    s = _store()
    assert s.search("anything") == []


def test_index_uses_inner_product_metric() -> None:
    """Embeddings are cosine-normalized and searched by inner product, not raw L2."""
    s = _store()
    s.add_documents(_docs())
    assert s.index.metric_type == faiss.METRIC_INNER_PRODUCT


def test_similarity_scores_are_cosine() -> None:
    s = _store()
    s.add_documents(_docs())
    # A query identical to a stored document maps to the same unit vector, so
    # cosine similarity is ~1.0, and all scores stay within [-1, 1].
    results = s.search("Bearer tokens authenticate API requests.", k=3)
    assert results[0].source == "b.md"
    assert results[0].similarity_score == pytest.approx(1.0, abs=1e-4)
    assert all(-1.0 - 1e-6 <= r.similarity_score <= 1.0 + 1e-6 for r in results)


def test_indexed_vectors_are_unit_norm() -> None:
    s = _store()
    s.add_documents(_docs())
    stored = s.index.reconstruct_n(0, s.index.ntotal)
    norms = np.linalg.norm(stored, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_get_document_stats() -> None:
    s = _store()
    s.add_documents(_docs())
    stats = s.get_document_stats()
    assert stats["total_documents"] == 3
    assert stats["total_indexed"] == 3
    assert stats["unique_sources"] == 3
    assert stats["embedding_dimension"] == 384
    assert stats["indexed"] is True


def test_stats_when_empty() -> None:
    s = _store()
    stats = s.get_document_stats()
    assert stats["total_documents"] == 0
    assert stats["indexed"] is False


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    s = _store()
    s.add_documents(_docs())
    s.save(str(tmp_path))
    s2 = _store()
    s2.load(str(tmp_path))
    assert len(s2.documents) == 3
    assert s2.index is not None
    assert s2.index.ntotal == 3


def test_save_writes_json_not_pickle(tmp_path: Path) -> None:
    s = _store()
    s.add_documents(_docs())
    s.save(str(tmp_path))

    assert (tmp_path / "documents.json").exists()
    assert not (tmp_path / "documents.pkl").exists()

    data = json.loads((tmp_path / "documents.json").read_text(encoding="utf-8"))
    assert len(data) == 3
    assert data[0]["content"] and data[0]["source"]


def test_reranking_returns_topk() -> None:
    s = _store()
    s.add_documents(_docs())
    out = s.semantic_search_with_reranking("documents table schema", k=3, rerank_top_k=2)
    assert len(out) == 2


def test_reranking_passthrough_when_few_results() -> None:
    s = _store()
    s.add_documents(_docs()[:1])
    out = s.semantic_search_with_reranking("anything", k=5, rerank_top_k=3)
    assert len(out) == 1


def test_hybrid_retriever_filters_by_source() -> None:
    s = _store()
    s.add_documents(_docs())
    h = HybridRetriever(s)
    out = h.retrieve_with_filters("anything", k=3, sources=["a.md"])
    assert all(r.source == "a.md" for r in out)


class _AngleModel:
    """Deterministic 2-D embedding model: each text maps to a unit vector at a
    fixed angle, so cosine ranking against the query (angle 0) is fully known."""

    def __init__(self, angles: dict[str, float]) -> None:
        self.angles = angles
        self.dimension = 2

    def encode(
        self, texts: str | list[str], convert_to_numpy: bool = True, show_progress_bar: bool = False
    ) -> np.ndarray:
        items = texts if isinstance(texts, list) else [texts]
        vecs = np.array(
            [[np.cos(self.angles[t]), np.sin(self.angles[t])] for t in items], dtype=np.float32
        )
        return vecs if isinstance(texts, list) else vecs[0]

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension


def test_filter_returns_k_even_when_source_ranks_below_overfetch_window() -> None:
    """A source filter must still return k results when the corpus has them,
    even if those docs rank far below the top few by similarity."""
    query = "Q"
    # 12 noise docs hug the query (small angles → high cosine); 3 target docs
    # sit at large angles → lowest cosine, well outside any small over-fetch.
    angles = {query: 0.0}
    docs = []
    for i in range(12):
        t = f"noise {i}"
        angles[t] = 0.05 * (i + 1)
        docs.append(Document(content=t, source="noise.md", chunk_id=i))
    for i in range(3):
        t = f"target {i}"
        angles[t] = 1.4 + 0.01 * i
        docs.append(Document(content=t, source="target.md", chunk_id=100 + i))

    s = FAISSVectorStore(model=_AngleModel(angles))
    s.add_documents(docs)

    h = HybridRetriever(s)
    out = h.retrieve_with_filters(query, k=3, sources=["target.md"])
    assert len(out) == 3
    assert all(r.source == "target.md" for r in out)


def test_add_documents_no_op_on_empty() -> None:
    s = _store()
    s.add_documents([])
    assert s.index is None
    assert s.documents == []
