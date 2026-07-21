from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest

from ingestion import Document
from rag_agent import StreamingResponse
from vector_store import FAISSVectorStore, SimpleEmbeddingModel

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _build_saved_index(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    vs = FAISSVectorStore(model=SimpleEmbeddingModel())
    vs.add_documents([Document(content="hello world", source="a.md", chunk_id=0)])
    vs.save(str(target_dir))


def _new_app(timeout: int = 60) -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=timeout)


def _seed_loaded_state(at: AppTest, agent: MagicMock | None = None) -> MagicMock:
    agent = agent or MagicMock()
    at.session_state["documents_loaded"] = True
    at.session_state["agent"] = agent
    at.session_state["vector_store"] = MagicMock()
    at.session_state["conversation_history"] = []
    at.session_state["index_stats"] = {
        "total_documents": 1,
        "total_indexed": 1,
        "unique_sources": 1,
        "embedding_dimension": 384,
        "indexed": True,
        "sources": ["a.md"],
    }
    return agent


def test_app_auto_loads_saved_index_on_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    index_dir = tmp_path / "rag_index"
    _build_saved_index(index_dir)

    import paths

    monkeypatch.setattr(paths, "INDEX_DIR", index_dir)

    at = _new_app()
    at.run()

    assert not at.exception
    assert at.session_state["documents_loaded"] is True
    assert at.session_state["agent"] is not None
    assert at.session_state["index_stats"]["total_indexed"] == 1


def test_sidebar_renders_file_uploader() -> None:
    at = _new_app()
    at.run()
    assert not at.exception
    keys = [getattr(el, "key", None) for el in at.sidebar.file_uploader]
    assert "file_uploader" in keys


def test_initial_state_shows_warning_and_init_button() -> None:
    at = _new_app()
    at.run()
    assert not at.exception
    assert at.session_state["documents_loaded"] is False
    warning_text = " ".join(w.value for w in at.warning)
    assert "No documents indexed" in warning_text
    button_labels = [b.label for b in at.button]
    assert any("Initialize System" in label for label in button_labels)


def _stream_response(chunks: list[str], **kwargs: object) -> StreamingResponse:
    defaults = {"sources": [], "tool_used": "question_answering", "confidence": 0.9}
    defaults.update(kwargs)
    return StreamingResponse(_chunks=iter(chunks), **defaults)  # type: ignore[arg-type]


def test_qa_tab_streams_answer_via_agent() -> None:
    at = _new_app()
    agent = MagicMock()
    agent.query_stream.return_value = _stream_response(["the ", "answer"])
    _seed_loaded_state(at, agent)
    at.run()
    assert not at.exception

    qa_input = next(t for t in at.text_input if t.key == "qa_question")
    qa_input.set_value("what is the answer?")
    search_btn = next(b for b in at.button if b.label == "Search & Answer")
    search_btn.click()
    at.run()

    agent.query_stream.assert_called_once()
    assert agent.query_stream.call_args.args[0] == "what is the answer?"


class _RaisingChunks:
    def __iter__(self) -> _RaisingChunks:
        return self

    def __next__(self) -> str:
        raise RuntimeError("api exploded")


def test_qa_tab_shows_error_instead_of_traceback_when_llm_fails() -> None:
    at = _new_app()
    agent = MagicMock()
    agent.query_stream.return_value = StreamingResponse(
        _chunks=_RaisingChunks(),
        sources=[],
        tool_used="question_answering",
        confidence=0.0,
    )
    _seed_loaded_state(at, agent)
    at.run()
    next(t for t in at.text_input if t.key == "qa_question").set_value("boom?")
    next(b for b in at.button if b.label == "Search & Answer").click()
    at.run()

    # A raised LLM error must be surfaced as an st.error message, not an app crash.
    assert not at.exception
    assert any("failed" in e.value.lower() for e in at.error)


def test_qa_tab_warns_on_empty_question() -> None:
    at = _new_app()
    agent = _seed_loaded_state(at)
    at.run()
    next(b for b in at.button if b.label == "Search & Answer").click()
    at.run()
    assert any("enter a question" in w.value.lower() for w in at.warning)
    agent.query_stream.assert_not_called()


def test_summarize_tab_prefixes_query() -> None:
    at = _new_app()
    agent = MagicMock()
    agent.query_stream.return_value = _stream_response(
        ["sum"], tool_used="summarization", confidence=0.7
    )
    _seed_loaded_state(at, agent)
    at.run()
    next(t for t in at.text_input if t.key == "summary_query").set_value("the api")
    next(b for b in at.button if b.label == "Summarize").click()
    at.run()
    agent.query_stream.assert_called_once()
    assert agent.query_stream.call_args.args[0].startswith("Summarize the following:")


def test_sql_tab_prefixes_query() -> None:
    at = _new_app()
    agent = MagicMock()
    agent.query_stream.return_value = _stream_response(
        ["SELECT 1"], tool_used="sql_generation", confidence=0.8
    )
    _seed_loaded_state(at, agent)
    at.run()
    next(t for t in at.text_input if t.key == "sql_request").set_value("recent docs")
    next(b for b in at.button if b.label == "Generate SQL").click()
    at.run()
    agent.query_stream.assert_called_once()
    assert agent.query_stream.call_args.args[0].startswith("Generate SQL for:")


def test_qa_with_sources_renders_format_sources() -> None:
    at = _new_app()
    agent = MagicMock()
    agent.query_stream.return_value = _stream_response(
        ["here ", "you ", "go"],
        sources=[
            {
                "file": "/tmp/api_docs.md",
                "chunk_id": 3,
                "excerpt": "Bearer token auth.",
                "similarity": "82.50%",
            }
        ],
        tool_used="question_answering",
        confidence=0.825,
    )
    _seed_loaded_state(at, agent)
    at.run()
    next(t for t in at.text_input if t.key == "qa_question").set_value("auth?")
    next(b for b in at.button if b.label == "Search & Answer").click()
    at.run()
    assert not at.exception
    metric_labels = [m.label for m in at.metric]
    assert "Confidence" in metric_labels
    assert "Tool Used" in metric_labels


def test_summarize_empty_input_warns() -> None:
    at = _new_app()
    agent = _seed_loaded_state(at)
    at.run()
    next(b for b in at.button if b.label == "Summarize").click()
    at.run()
    assert any("what to summarize" in w.value.lower() for w in at.warning)
    agent.query.assert_not_called()


def test_sql_empty_input_warns() -> None:
    at = _new_app()
    agent = _seed_loaded_state(at)
    at.run()
    next(b for b in at.button if b.label == "Generate SQL").click()
    at.run()
    assert any("describe your query" in w.value.lower() for w in at.warning)
    agent.query.assert_not_called()


def test_chat_send_invokes_multi_turn_and_renders_history() -> None:
    at = _new_app()
    agent = MagicMock()
    agent.multi_turn_conversation_stream.return_value = _stream_response(
        ["hi ", "back"], tool_used="question_answering", confidence=0.5
    )
    _seed_loaded_state(at, agent)
    at.session_state["conversation_history"] = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    at.run()
    next(t for t in at.text_input if t.key == "chat_message").set_value("again")
    next(b for b in at.button if b.label == "Send").click()
    at.run()
    agent.multi_turn_conversation_stream.assert_called_once_with("again")


def test_chat_send_empty_warns() -> None:
    at = _new_app()
    agent = _seed_loaded_state(at)
    at.run()
    next(b for b in at.button if b.label == "Send").click()
    at.run()
    assert any("enter a message" in w.value.lower() for w in at.warning)
    agent.multi_turn_conversation_stream.assert_not_called()


def test_clear_history_button_resets_history() -> None:
    at = _new_app()
    agent = _seed_loaded_state(at)
    at.session_state["conversation_history"] = [{"role": "user", "content": "hi"}]
    at.run()
    next(b for b in at.button if "Clear History" in b.label).click()
    at.run()
    assert at.session_state["conversation_history"] == []
    agent.clear_history.assert_called_once()
