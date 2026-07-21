from __future__ import annotations

from pathlib import Path

import pytest

from eval_harness import GOLD_SET, EvalCase, compute_recall_at_k, evaluate
from ingestion import Document, DocumentIngester, create_sample_documents
from vector_store import FAISSVectorStore, SimpleEmbeddingModel


def _store_with(docs: list[Document]) -> FAISSVectorStore:
    s = FAISSVectorStore(model=SimpleEmbeddingModel())
    s.add_documents(docs)
    return s


def test_recall_one_when_expected_source_retrieved() -> None:
    store = _store_with(
        [
            Document(content="bearer token authentication", source="api_docs.md", chunk_id=0),
            Document(content="postgres tables", source="database_schema.md", chunk_id=1),
        ]
    )
    case = EvalCase(query="bearer token", expected_sources=["api_docs"])
    assert compute_recall_at_k(store, case, k=2) == 1.0


def test_recall_zero_when_expected_source_missing() -> None:
    store = _store_with([Document(content="something unrelated", source="other.md", chunk_id=0)])
    case = EvalCase(query="bearer token", expected_sources=["api_docs"])
    assert compute_recall_at_k(store, case, k=2) == 0.0


def test_evaluate_aggregates_recall_across_cases() -> None:
    store = _store_with(
        [
            Document(content="bearer token authentication", source="api_docs.md", chunk_id=0),
            Document(content="postgres tables", source="database_schema.md", chunk_id=1),
        ]
    )
    cases = [
        EvalCase(query="bearer token", expected_sources=["api_docs"]),
        EvalCase(query="nonexistent", expected_sources=["nowhere"]),
    ]
    result = evaluate(store, cases, k=2)
    assert result["recall_at_k"] == pytest.approx(0.5)
    assert result["k"] == 2
    assert result["num_cases"] == 2


def test_gold_set_is_non_empty() -> None:
    assert len(GOLD_SET) >= 4
    assert all(isinstance(c, EvalCase) for c in GOLD_SET)


def test_eval_harness_main_prints_recall(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`python eval_harness.py` should build a store and print recall numbers."""
    import eval_harness

    store = _store_with(
        [
            Document(content="bearer token authentication", source="api_docs.md", chunk_id=0),
            Document(content="system architecture services", source="architecture.md", chunk_id=1),
            Document(content="documents table columns", source="database_schema.md", chunk_id=2),
            Document(content="faiss index build", source="vector_search.py", chunk_id=3),
        ]
    )
    monkeypatch.setattr(eval_harness, "build_sample_store", lambda: store)

    eval_harness.main()

    out = capsys.readouterr().out.lower()
    assert "recall@" in out


def test_sample_docs_meet_recall_floor(tmp_path: Path) -> None:
    """Integration: sample docs + gold set must clear the recall floor."""
    docs_dir = tmp_path / "sample_docs"
    create_sample_documents(str(docs_dir))
    documents = DocumentIngester(chunk_size=512, overlap=100).ingest_directory(str(docs_dir))

    store = FAISSVectorStore()
    store.add_documents(documents)

    result = evaluate(store, GOLD_SET, k=5)
    assert result["recall_at_k"] >= 0.8, result
