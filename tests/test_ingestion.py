from __future__ import annotations

from pathlib import Path

from ingestion import Document, DocumentIngester, create_sample_documents


def test_document_dataclass_defaults() -> None:
    doc = Document(content="hi", source="x.md", chunk_id=0)
    assert doc.start_line == 0
    assert doc.end_line == 0


def test_ingest_text_file_produces_chunks(tmp_path: Path) -> None:
    f = tmp_path / "notes.txt"
    f.write_text("Sentence one. Sentence two. " * 50, encoding="utf-8")
    ingester = DocumentIngester(chunk_size=128, overlap=20)
    chunks = ingester.ingest_file(str(f))
    assert len(chunks) >= 2
    assert all(isinstance(c, Document) for c in chunks)
    assert all(c.source == str(f) for c in chunks)


def test_ingest_markdown_splits_on_headers(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text(
        "# Title\nintro paragraph\n\n## Section A\nbody A\n\n## Section B\nbody B\n",
        encoding="utf-8",
    )
    chunks = DocumentIngester(chunk_size=2048).ingest_file(str(f))
    assert len(chunks) >= 2
    combined = "\n".join(c.content for c in chunks)
    assert "Section A" in combined
    assert "Section B" in combined


def test_ingest_python_splits_on_defs(tmp_path: Path) -> None:
    f = tmp_path / "mod.py"
    f.write_text(
        "def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n",
        encoding="utf-8",
    )
    chunks = DocumentIngester(chunk_size=2048).ingest_file(str(f))
    assert len(chunks) >= 1
    assert any("alpha" in c.content for c in chunks)
    assert any("beta" in c.content for c in chunks)


def test_ingest_directory_chunks_unique_by_source_and_id(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\nbody\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("just text content here.", encoding="utf-8")
    (tmp_path / "skip.bin").write_text("ignored", encoding="utf-8")
    chunks = DocumentIngester().ingest_directory(str(tmp_path))
    pairs = [(c.source, c.chunk_id) for c in chunks]
    assert len(pairs) == len(set(pairs))
    assert all(not c.source.endswith(".bin") for c in chunks)


def test_ingest_directory_preserves_per_file_chunk_indices(tmp_path: Path) -> None:
    """chunk_id is a per-file index, so every file's chunks start at 0."""
    (tmp_path / "a.md").write_text("# A\nalpha body\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta text content here.", encoding="utf-8")
    chunks = DocumentIngester().ingest_directory(str(tmp_path))

    by_source: dict[str, list[int]] = {}
    for c in chunks:
        by_source.setdefault(c.source, []).append(c.chunk_id)

    assert len(by_source) == 2
    for ids in by_source.values():
        assert min(ids) == 0


def test_ingest_file_missing_returns_empty(tmp_path: Path) -> None:
    chunks = DocumentIngester().ingest_file(str(tmp_path / "nope.txt"))
    assert chunks == []


def test_create_sample_documents_default_dir(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import paths

    monkeypatch.setattr(paths, "SAMPLE_DOCS_DIR", tmp_path / "sample_docs")
    create_sample_documents()
    out = tmp_path / "sample_docs"
    assert (out / "api_docs.md").exists()
    assert (out / "architecture.md").exists()
    assert (out / "database_schema.md").exists()
    assert (out / "vector_search.py").exists()


def test_create_sample_documents_custom_dir(tmp_path: Path) -> None:
    target = tmp_path / "custom"
    create_sample_documents(str(target))
    assert (target / "api_docs.md").exists()
