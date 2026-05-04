from __future__ import annotations

from pathlib import Path

from ingestion import Document
from uploads import ingest_uploaded_files
from vector_store import FAISSVectorStore, SimpleEmbeddingModel


class _FakeUpload:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def _empty_store() -> FAISSVectorStore:
    s = FAISSVectorStore.__new__(FAISSVectorStore)
    s.model = SimpleEmbeddingModel()
    s.embedding_dim = 384
    s.index = None
    s.documents = []
    s.use_gpu = False
    return s


def test_ingest_writes_files_and_returns_chunk_count(tmp_path: Path) -> None:
    uploads_dir = tmp_path / "uploads"
    files = [
        _FakeUpload("notes.md", b"# Heading\nBody paragraph."),
        _FakeUpload("snippet.py", b"def foo():\n    return 1\n"),
    ]

    store, count = ingest_uploaded_files(files, uploads_dir, _empty_store())

    assert (uploads_dir / "notes.md").exists()
    assert (uploads_dir / "snippet.py").exists()
    assert count > 0
    assert store.index is not None
    assert store.index.ntotal == count


def test_ingest_appends_to_existing_store(tmp_path: Path) -> None:
    store = _empty_store()
    store.add_documents([Document(content="seed doc", source="seed.md", chunk_id=0)])
    seed_total = store.index.ntotal

    new_files = [_FakeUpload("more.md", b"# Another\nNew content here.")]
    store, added = ingest_uploaded_files(new_files, tmp_path / "uploads", store)

    assert added > 0
    assert store.index.ntotal == seed_total + added


def test_ingest_creates_store_when_none_provided(tmp_path: Path) -> None:
    files = [_FakeUpload("a.md", b"# Title\nText.")]
    store, count = ingest_uploaded_files(files, tmp_path / "uploads", existing_store=None)
    assert isinstance(store, FAISSVectorStore)
    assert count > 0


def test_ingest_with_no_files_returns_zero(tmp_path: Path) -> None:
    store = _empty_store()
    out_store, count = ingest_uploaded_files([], tmp_path / "uploads", store)
    assert out_store is store
    assert count == 0
