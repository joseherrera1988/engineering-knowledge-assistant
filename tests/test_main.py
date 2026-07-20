from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import main
from rag_agent import AgentResponse


def _patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    docs = tmp_path / "sample_docs"
    index = tmp_path / "rag_index"
    monkeypatch.setattr(main, "SAMPLE_DOCS_DIR", docs)
    monkeypatch.setattr(main, "INDEX_DIR", index)
    return docs, index


def test_initialize_system_creates_docs_and_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    docs, index = _patch_paths(monkeypatch, tmp_path)

    vector_store, documents = main.initialize_system()

    assert docs.exists() and any(docs.iterdir())
    assert (index / "index.faiss").exists()
    assert (index / "documents.pkl").exists()
    assert len(documents) > 0
    assert vector_store.index.ntotal == len(documents)


def test_run_demo_queries_invokes_agent_for_each_query(capsys: pytest.CaptureFixture[str]) -> None:
    agent = MagicMock()
    agent.query.return_value = AgentResponse(
        answer="short answer",
        sources=[{"file": "a.md"}],
        tool_used="question_answering",
        confidence=0.85,
    )
    queries = ["q1", "q2", "q3"]

    main.run_demo_queries(agent, queries)

    assert agent.query.call_count == 3
    for q, call in zip(queries, agent.query.call_args_list, strict=True):
        assert call.args[0] == q
        assert call.kwargs["k"] == 5
    out = capsys.readouterr().out
    assert "Demo Queries" in out
    assert "short answer" in out


def test_run_demo_queries_truncates_long_answers(capsys: pytest.CaptureFixture[str]) -> None:
    agent = MagicMock()
    long = "x" * 600
    agent.query.return_value = AgentResponse(
        answer=long, sources=[], tool_used="question_answering", confidence=0.5
    )
    main.run_demo_queries(agent, ["q"])
    out = capsys.readouterr().out
    assert "..." in out
    assert long not in out


def test_main_init_mode_only_initializes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.argv", ["main.py", "--mode", "init"])
    sentinel = MagicMock()
    monkeypatch.setattr(main.subprocess, "run", sentinel)

    main.main()

    sentinel.assert_not_called()


def test_main_demo_mode_loads_existing_index_and_runs_queries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_paths(monkeypatch, tmp_path)

    # First, build an index that demo mode can load.
    main.initialize_system()

    monkeypatch.setattr("sys.argv", ["main.py", "--mode", "demo"])
    fake_agent = MagicMock()
    fake_agent.query.return_value = AgentResponse(
        answer="ok", sources=[], tool_used="question_answering", confidence=0.5
    )
    monkeypatch.setattr(main, "RAGAgent", lambda store, **_: fake_agent)
    sentinel = MagicMock()
    monkeypatch.setattr(main.subprocess, "run", sentinel)

    main.main()

    assert fake_agent.query.called
    sentinel.assert_not_called()


def test_main_ui_mode_launches_streamlit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.argv", ["main.py", "--mode", "ui"])
    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str]) -> None:
        captured["cmd"] = cmd

    monkeypatch.setattr(main.subprocess, "run", fake_run)

    main.main()

    assert captured["cmd"][0] == "streamlit"
    assert captured["cmd"][1] == "run"
    assert captured["cmd"][2].endswith("app.py")


def test_main_full_mode_runs_all_phases(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.argv", ["main.py", "--mode", "full"])

    fake_agent = MagicMock()
    fake_agent.query.return_value = AgentResponse(
        answer="ok", sources=[], tool_used="question_answering", confidence=0.5
    )
    monkeypatch.setattr(main, "RAGAgent", lambda store, **_: fake_agent)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(main.subprocess, "run", lambda cmd: captured.setdefault("cmd", cmd))

    main.main()

    assert (tmp_path / "rag_index" / "index.faiss").exists()
    assert fake_agent.query.called
    assert captured["cmd"][0] == "streamlit"
