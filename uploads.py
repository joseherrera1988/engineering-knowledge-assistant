"""User-uploaded document ingestion."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from ingestion import Document, DocumentIngester
from vector_store import FAISSVectorStore


class UploadedFileLike(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


def ingest_uploaded_files(
    files: Iterable[UploadedFileLike],
    uploads_dir: Path,
    existing_store: FAISSVectorStore | None = None,
) -> tuple[FAISSVectorStore, int]:
    """Persist uploaded files, ingest them, and append to a vector store.

    Returns the (possibly newly created) store and the number of chunks added.
    """
    files = list(files)
    if not files:
        return existing_store if existing_store is not None else FAISSVectorStore(), 0

    uploads_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for f in files:
        target = uploads_dir / f.name
        target.write_bytes(f.getvalue())
        written.append(target)

    ingester = DocumentIngester(chunk_size=512, overlap=100)
    docs: list[Document] = []
    for p in written:
        docs.extend(ingester.ingest_file(str(p)))

    store = existing_store if existing_store is not None else FAISSVectorStore()
    if docs:
        store.add_documents(docs)
    return store, len(docs)
