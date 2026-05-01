from __future__ import annotations

from pathlib import Path

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
    s = FAISSVectorStore.__new__(FAISSVectorStore)
    s.model = SimpleEmbeddingModel()
    s.embedding_dim = 384
    s.index = None
    s.documents = []
    s.use_gpu = False
    return s


def test_simple_embedding_is_deterministic() -> None:
    m = SimpleEmbeddingModel()
    a = m.encode("hello")
    b = m.encode("hello")
    assert (a == b).all()


def test_simple_embedding_list_shape() -> None:
    m = SimpleEmbeddingModel()
    out = m.encode(["x", "y", "z"])
    assert out.shape == (3, 384)


def test_add_and_search_returns_results() -> None:
    s = _store()
    s.add_documents(_docs())
    results = s.search("vector similarity FAISS", k=2)
    assert len(results) == 2
    assert results[0].similarity_score >= results[1].similarity_score


def test_search_empty_store_returns_empty() -> None:
    s = _store()
    assert s.search("anything") == []


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


def test_add_documents_no_op_on_empty() -> None:
    s = _store()
    s.add_documents([])
    assert s.index is None
    assert s.documents == []
