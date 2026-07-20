"""
Vector Store and Retrieval Module
FAISS-based semantic search with source tracking
"""

import hashlib
import pickle
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


@dataclass
class RetrievalResult:
    """Result from semantic search"""

    content: str
    source: str
    chunk_id: int
    similarity_score: float
    distance: float


class SimpleEmbeddingModel:
    """Fallback embedding model using hash-based embeddings"""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def encode(
        self,
        texts: str | list[str],
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray[Any, Any]:
        """Create deterministic embeddings from text."""
        embeddings: list[np.ndarray[Any, Any]] = []
        text_list = texts if isinstance(texts, list) else [texts]

        for text in text_list:
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            rng = np.random.default_rng(hash_val % (2**31))
            embedding = rng.standard_normal(self.dimension).astype(np.float32)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            embeddings.append(embedding)

        if isinstance(texts, str):
            return embeddings[0]
        return np.array(embeddings)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension


class FAISSVectorStore:
    """FAISS-based vector store for semantic search"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_gpu: bool = False) -> None:
        self.model: Any
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name)
                get_dim = getattr(
                    self.model,
                    "get_embedding_dimension",
                    self.model.get_sentence_embedding_dimension,
                )
                self.embedding_dim = get_dim()
            except Exception:
                print("⚠️ Using fallback embedding model")
                self.model = SimpleEmbeddingModel()
                self.embedding_dim = 384
        else:
            self.model = SimpleEmbeddingModel()
            self.embedding_dim = 384

        self.index: Any | None = None
        self.documents: list[Any] = []
        self.use_gpu = use_gpu

    def add_documents(self, documents: list[Any]) -> None:
        """Add documents to the vector store"""
        if not documents:
            print("No documents to add")
            return

        print(f"Encoding {len(documents)} documents...")

        embeddings = self.model.encode(
            [doc.content for doc in documents], convert_to_numpy=True, show_progress_bar=True
        )

        if self.index is None:
            # Inner product over L2-normalized vectors == cosine similarity,
            # which is what MiniLM-family embeddings are trained for.
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.documents.extend(documents)
        print(f"✓ Index now contains {self.index.ntotal} documents")

    def search(self, query: str, k: int = 5, threshold: float = -1.0) -> list[RetrievalResult]:
        """Search for similar documents.

        Scores are cosine similarity in [-1, 1]; the default threshold of -1.0
        returns all k neighbors. Pass a higher threshold (e.g. 0.3) to drop
        weakly related results.
        """
        if self.index is None or len(self.documents) == 0:
            return []

        query_embedding = self.model.encode(query, convert_to_numpy=True)
        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                # Inner product of unit vectors is cosine similarity in [-1, 1].
                similarity = float(scores[0][i])
                distance = 1.0 - similarity

                if similarity >= threshold:
                    doc = self.documents[idx]
                    results.append(
                        RetrievalResult(
                            content=doc.content,
                            source=doc.source,
                            chunk_id=doc.chunk_id,
                            similarity_score=similarity,
                            distance=distance,
                        )
                    )

        return results

    def semantic_search_with_reranking(
        self, query: str, k: int = 10, rerank_top_k: int = 5
    ) -> list[RetrievalResult]:
        """Perform semantic search with reranking"""
        results = self.search(query, k=k)

        if len(results) <= rerank_top_k:
            return results

        query_terms = set(query.lower().split())
        scored_results = []

        for result in results:
            content_terms = set(result.content.lower().split())
            term_overlap = len(query_terms & content_terms) / len(query_terms)
            combined_score = result.similarity_score * 0.7 + term_overlap * 0.3
            scored_results.append((result, combined_score))

        scored_results.sort(key=lambda x: x[1], reverse=True)
        return [result for result, _ in scored_results[:rerank_top_k]]

    def save(self, path: str) -> None:
        """Save index and metadata to disk"""
        import os

        os.makedirs(path, exist_ok=True)

        faiss.write_index(self.index, f"{path}/index.faiss")

        with open(f"{path}/documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        with open(f"{path}/model_info.txt", "w") as f:
            f.write(f"embedding_dim: {self.embedding_dim}\n")
            f.write(f"total_docs: {len(self.documents)}\n")

        print(f"✓ Vector store saved to {path}")

    def load(self, path: str) -> None:
        """Load index and metadata from disk"""
        self.index = faiss.read_index(f"{path}/index.faiss")

        with open(f"{path}/documents.pkl", "rb") as f:
            self.documents = pickle.load(f)

        print(f"✓ Loaded {len(self.documents)} documents from {path}")

    def get_document_stats(self) -> dict[str, Any]:
        """Get statistics about the vector store"""
        if self.index is None:
            return {"total_documents": 0, "indexed": False}

        unique_sources = set(doc.source for doc in self.documents)

        return {
            "total_documents": len(self.documents),
            "total_indexed": self.index.ntotal,
            "unique_sources": len(unique_sources),
            "embedding_dimension": self.embedding_dim,
            "indexed": True,
            "sources": list(unique_sources),
        }


class HybridRetriever:
    """Combines semantic and keyword-based retrieval"""

    def __init__(self, vector_store: FAISSVectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """Retrieve using semantic search"""
        return self.vector_store.semantic_search_with_reranking(
            query=query, k=k * 2, rerank_top_k=k
        )

    def retrieve_with_filters(
        self, query: str, k: int = 5, sources: list[str] | None = None
    ) -> list[RetrievalResult]:
        """Retrieve with optional source filtering.

        Filtering happens over the whole ranked corpus rather than a small
        over-fetched slice, so a source filter returns up to k results whenever
        the corpus holds that many for the given sources -- it never silently
        returns fewer just because the matching docs ranked low.
        """
        if not sources:
            return self.retrieve(query, k=k)

        total = self.vector_store.index.ntotal if self.vector_store.index is not None else 0
        if total == 0:
            return []

        allowed = set(sources)
        ranked = self.vector_store.search(query, k=total)
        filtered = [r for r in ranked if r.source in allowed]
        return filtered[:k]
