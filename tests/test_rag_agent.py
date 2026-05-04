from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ingestion import Document
from rag_agent import AgentResponse, RAGAgent
from vector_store import FAISSVectorStore, SimpleEmbeddingModel


def _store_with_docs() -> FAISSVectorStore:
    s = FAISSVectorStore.__new__(FAISSVectorStore)
    s.model = SimpleEmbeddingModel()
    s.embedding_dim = 384
    s.index = None
    s.documents = []
    s.use_gpu = False
    s.add_documents(
        [
            Document(content="FAISS provides vector similarity search.", source="a.md", chunk_id=0),
            Document(content="API uses Bearer token authentication.", source="b.md", chunk_id=1),
            Document(
                content="The documents table has a uuid primary key.", source="c.md", chunk_id=2
            ),
        ]
    )
    return s


class _FakeCompletions:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))]
        )


class _FakeClient:
    def __init__(self, reply: str = "STUBBED") -> None:
        self.completions = _FakeCompletions(reply)
        self.chat = SimpleNamespace(completions=self.completions)


def test_query_returns_answer_from_openai_client() -> None:
    fake = _FakeClient(reply="The answer is 42.")
    agent = RAGAgent(_store_with_docs(), client=fake, model="gpt-4o-mini")
    response = agent.query("What does FAISS do?")
    assert isinstance(response, AgentResponse)
    assert response.answer == "The answer is 42."
    assert response.tool_used == "question_answering"
    assert len(response.sources) > 0
    assert fake.completions.calls[0]["model"] == "gpt-4o-mini"


def test_query_routes_summarize() -> None:
    fake = _FakeClient(reply="summary text")
    agent = RAGAgent(_store_with_docs(), client=fake)
    response = agent.query("Summarize the docs")
    assert response.tool_used == "summarization"


def test_query_routes_sql() -> None:
    fake = _FakeClient(reply="SELECT 1;")
    agent = RAGAgent(_store_with_docs(), client=fake)
    response = agent.query("Generate a SQL query for documents")
    assert response.tool_used == "sql_generation"


def test_query_no_results() -> None:
    empty = FAISSVectorStore.__new__(FAISSVectorStore)
    empty.model = SimpleEmbeddingModel()
    empty.embedding_dim = 384
    empty.index = None
    empty.documents = []
    empty.use_gpu = False
    fake = _FakeClient()
    agent = RAGAgent(empty, client=fake)
    response = agent.query("anything")
    assert response.tool_used == "retrieval_failed"
    assert response.confidence == 0.0
    assert response.sources == []
    assert fake.completions.calls == []


def test_select_tool_routing() -> None:
    agent = RAGAgent(_store_with_docs(), client=_FakeClient())
    assert agent._select_tool("Summarize this") == "summarization"
    assert agent._select_tool("Write SQL for users") == "sql_generation"
    assert agent._select_tool("What is X?") == "question_answering"


def test_call_llm_falls_back_when_client_raises() -> None:
    class Boom:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._boom))

        def _boom(self, **_: Any) -> Any:
            raise RuntimeError("api down")

    agent = RAGAgent(_store_with_docs(), client=Boom())
    out = agent._call_llm("Summarize something")
    assert "Summary" in out


def test_multi_turn_conversation_tracks_history() -> None:
    fake = _FakeClient(reply="hi back")
    agent = RAGAgent(_store_with_docs(), client=fake)
    agent.multi_turn_conversation("hello")
    agent.multi_turn_conversation("again")
    assert len(agent.conversation_history) == 4
    agent.clear_history()
    assert agent.conversation_history == []


class _FakeStreamCompletions:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(
                [
                    SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=c))])
                    for c in self.chunks
                ]
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="".join(self.chunks)))]
        )


class _FakeStreamClient:
    def __init__(self, chunks: list[str]) -> None:
        self.completions = _FakeStreamCompletions(chunks)
        self.chat = SimpleNamespace(completions=self.completions)


def test_query_stream_yields_chunks_and_aggregates_answer() -> None:
    fake = _FakeStreamClient(["Hello, ", "world", "!"])
    agent = RAGAgent(_store_with_docs(), client=fake)
    stream = agent.query_stream("What does FAISS do?")

    chunks = list(stream)

    assert chunks == ["Hello, ", "world", "!"]
    assert stream.answer == "Hello, world!"
    assert stream.tool_used == "question_answering"
    assert len(stream.sources) > 0
    assert fake.completions.calls[0]["stream"] is True


def test_query_stream_no_results_returns_empty_stream() -> None:
    empty = FAISSVectorStore.__new__(FAISSVectorStore)
    empty.model = SimpleEmbeddingModel()
    empty.embedding_dim = 384
    empty.index = None
    empty.documents = []
    empty.use_gpu = False
    fake = _FakeStreamClient(["unused"])

    agent = RAGAgent(empty, client=fake)
    stream = agent.query_stream("anything")
    chunks = list(stream)

    assert chunks == []
    assert stream.tool_used == "retrieval_failed"
    assert stream.confidence == 0.0
    assert stream.sources == []
    assert "No relevant documents" in stream.answer
    assert fake.completions.calls == []


def test_query_stream_falls_back_when_client_raises() -> None:
    class Boom:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._boom))

        def _boom(self, **_: Any) -> Any:
            raise RuntimeError("api down")

    agent = RAGAgent(_store_with_docs(), client=Boom())
    stream = agent.query_stream("Summarize the docs")
    chunks = list(stream)
    assert chunks
    assert stream.answer == "".join(chunks)
    assert "Summary" in stream.answer


def test_get_stats_shape() -> None:
    agent = RAGAgent(_store_with_docs(), client=_FakeClient())
    stats = agent.get_stats()
    assert "vector_store_stats" in stats
    assert stats["conversation_turns"] == 0
    assert stats["total_queries"] == 0
