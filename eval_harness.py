"""Retrieval evaluation: gold-set recall@k for the FAISS vector store."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from vector_store import FAISSVectorStore


@dataclass(frozen=True)
class EvalCase:
    """A single retrieval evaluation case.

    A retrieved document is considered relevant if its `source` contains any of
    the `expected_sources` substrings (case-insensitive).
    """

    query: str
    expected_sources: tuple[str, ...] | list[str]


GOLD_SET: list[EvalCase] = [
    EvalCase(
        query="What authentication does the API use?",
        expected_sources=["api_docs"],
    ),
    EvalCase(
        query="How is the system architecture organized into services?",
        expected_sources=["architecture"],
    ),
    EvalCase(
        query="What columns does the documents table have?",
        expected_sources=["database_schema"],
    ),
    EvalCase(
        query="How does the FAISS vector search build its index?",
        expected_sources=["vector_search", "architecture"],
    ),
    EvalCase(
        query="What is the API rate limit?",
        expected_sources=["api_docs"],
    ),
]


def compute_recall_at_k(store: FAISSVectorStore, case: EvalCase, k: int = 5) -> float:
    """Return 1.0 if any retrieved doc's source matches an expected substring, else 0.0."""
    results = store.semantic_search_with_reranking(case.query, k=max(k, k + 1), rerank_top_k=k)
    expected = [s.lower() for s in case.expected_sources]
    for r in results:
        src = r.source.lower()
        if any(e in src for e in expected):
            return 1.0
    return 0.0


def evaluate(
    store: FAISSVectorStore, cases: Sequence[EvalCase], k: int = 5
) -> dict[str, float | int]:
    """Run the gold set and return aggregate retrieval metrics."""
    if not cases:
        return {"recall_at_k": 0.0, "k": k, "num_cases": 0}
    scores = [compute_recall_at_k(store, c, k=k) for c in cases]
    return {
        "recall_at_k": sum(scores) / len(scores),
        "k": k,
        "num_cases": len(cases),
    }
