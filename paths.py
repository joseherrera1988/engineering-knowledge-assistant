"""Project-relative default paths for sample docs and the FAISS index."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent
SAMPLE_DOCS_DIR: Path = PROJECT_ROOT / "sample_docs"
INDEX_DIR: Path = PROJECT_ROOT / "rag_index"
APP_PY: Path = PROJECT_ROOT / "app.py"
