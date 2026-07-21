"""RAG Agent with LLM-powered tools using the OpenAI SDK."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from vector_store import FAISSVectorStore, RetrievalResult

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 1024

DEMO_NOTICE = (
    "⚠️ DEMO MODE — no LLM API key is configured, so this is a canned response "
    "and is not grounded in the retrieved sources.\n\n"
)


def demo_mode_from_env() -> bool:
    """True when no OpenAI API key is configured, so callers should run in demo mode."""
    return not os.environ.get("OPENAI_API_KEY")


@dataclass
class AgentResponse:
    """Response from RAG agent."""

    answer: str
    sources: list[dict[str, Any]]
    tool_used: str
    confidence: float


@dataclass
class StreamingResponse:
    """Iterable streaming RAG response. The full answer is available via `.answer`
    after iteration completes."""

    sources: list[dict[str, Any]]
    tool_used: str
    confidence: float
    _chunks: Iterator[str]
    _parts: list[str] = field(default_factory=list)

    def __iter__(self) -> Iterator[str]:
        for chunk in self._chunks:
            self._parts.append(chunk)
            yield chunk

    @property
    def answer(self) -> str:
        return "".join(self._parts)


def _default_client() -> OpenAI:
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-test"))


class RAGAgent:
    """Agent that reasons over retrieved documents using OpenAI."""

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        client: Any | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        demo_mode: bool = False,
    ) -> None:
        self.vector_store = vector_store
        self.client: Any = client if client is not None else _default_client()
        self.model = model
        self.max_tokens = max_tokens
        self.demo_mode = demo_mode
        self.conversation_history: list[dict[str, str]] = []

    def query(self, user_query: str, k: int = 5) -> AgentResponse:
        retrieved_docs = self.vector_store.semantic_search_with_reranking(
            user_query, k=min(k, 10), rerank_top_k=k
        )

        if not retrieved_docs:
            return AgentResponse(
                answer="No relevant documents found. Please try a different search query.",
                sources=[],
                tool_used="retrieval_failed",
                confidence=0.0,
            )

        context = self._format_context(retrieved_docs)
        tool = self._select_tool(user_query)

        if tool == "summarization":
            answer = self._summarize(context, user_query)
            tool_used = "summarization"
        elif tool == "sql_generation":
            answer = self._generate_sql(context, user_query)
            tool_used = "sql_generation"
        else:
            answer = self._answer_question(context, user_query)
            tool_used = "question_answering"

        sources = [
            {
                "file": doc.source,
                "chunk_id": doc.chunk_id,
                "excerpt": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                "similarity": f"{doc.similarity_score:.2%}",
            }
            for doc in retrieved_docs
        ]

        confidence = retrieved_docs[0].similarity_score
        return AgentResponse(
            answer=answer, sources=sources, tool_used=tool_used, confidence=confidence
        )

    def _select_tool(self, query: str) -> str:
        q = query.lower()
        if any(word in q for word in ["summarize", "summary", "overview"]):
            return "summarization"
        # SQL generation is a deliberate action; require an explicit SQL signal.
        # The bare words "query"/"database" also occur in ordinary questions
        # ("how do I query the docs?"), so they no longer route here.
        if "sql" in q or "select statement" in q:
            return "sql_generation"
        return "question_answering"

    def _format_context(self, documents: list[RetrievalResult]) -> str:
        return "\n\n".join(
            f"[Source {i}: {doc.source}]\n{doc.content}" for i, doc in enumerate(documents, 1)
        )

    def _answer_prompt(self, context: str, question: str) -> str:
        return (
            "You are a helpful assistant answering questions about technical documentation.\n\n"
            f"Context from documents:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Provide a clear, accurate answer based on the context above. "
            "If the context doesn't contain the answer, say so honestly."
        )

    def _summarize_prompt(self, context: str, query: str) -> str:
        topic = query.replace("summarize", "").replace("summary", "").strip()
        return (
            "You are a technical documentation expert. "
            "Provide a concise summary of the following documents.\n\n"
            f"Documents:\n{context}\n\n"
            f"Focus: {topic if topic else 'all key points'}\n\n"
            "Provide a clear, well-organized summary. Use headers and bullet points where appropriate."
        )

    def _sql_prompt(self, context: str, query: str) -> str:
        return (
            "You are an SQL expert. Based on the database schema and documentation below, "
            "generate a SQL query to fulfill the request.\n\n"
            f"Documentation and Schema:\n{context}\n\n"
            f"Request: {query}\n\n"
            "Provide:\n1. The SQL query with proper formatting\n"
            "2. A brief explanation of what the query does\n3. Any assumptions made\n\n"
            "Format the SQL code in a markdown code block with ```sql markers."
        )

    def _answer_question(self, context: str, question: str) -> str:
        return self._call_llm(self._answer_prompt(context, question))

    def _summarize(self, context: str, query: str) -> str:
        return self._call_llm(self._summarize_prompt(context, query))

    def _generate_sql(self, context: str, query: str) -> str:
        return self._call_llm(self._sql_prompt(context, query))

    def query_stream(self, user_query: str, k: int = 5) -> StreamingResponse:
        retrieved_docs = self.vector_store.semantic_search_with_reranking(
            user_query, k=min(k, 10), rerank_top_k=k
        )

        if not retrieved_docs:
            stream = StreamingResponse(
                sources=[], tool_used="retrieval_failed", confidence=0.0, _chunks=iter(())
            )
            stream._parts.append(
                "No relevant documents found. Please try a different search query."
            )
            return stream

        context = self._format_context(retrieved_docs)
        tool = self._select_tool(user_query)

        if tool == "summarization":
            prompt = self._summarize_prompt(context, user_query)
            tool_used = "summarization"
        elif tool == "sql_generation":
            prompt = self._sql_prompt(context, user_query)
            tool_used = "sql_generation"
        else:
            prompt = self._answer_prompt(context, user_query)
            tool_used = "question_answering"

        sources = [
            {
                "file": doc.source,
                "chunk_id": doc.chunk_id,
                "excerpt": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                "similarity": f"{doc.similarity_score:.2%}",
            }
            for doc in retrieved_docs
        ]
        return StreamingResponse(
            sources=sources,
            tool_used=tool_used,
            confidence=retrieved_docs[0].similarity_score,
            _chunks=self._call_llm_stream(prompt),
        )

    def _call_llm_stream(self, prompt: str) -> Iterator[str]:
        if self.demo_mode:
            yield self._generate_demo_response(prompt)
            return
        stream = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if isinstance(delta, str) and delta:
                yield delta

    def _call_llm(self, prompt: str) -> str:
        if self.demo_mode:
            return self._generate_demo_response(prompt)
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        return content if isinstance(content, str) else ""

    def _generate_demo_response(self, prompt: str) -> str:
        p = prompt.lower()
        if "summarize" in p or "summary" in p:
            body = (
                "## Summary\n\n"
                "- Document management with versioning\n"
                "- Bearer-token authenticated REST API\n"
                "- Microservices: PostgreSQL + Redis + FAISS/Elasticsearch\n"
                "- Rate limited to 100 req/min\n"
            )
        elif "sql" in p:
            body = (
                "```sql\n"
                "SELECT id, title, created_at FROM documents\n"
                "WHERE is_deleted = FALSE\n"
                "ORDER BY created_at DESC LIMIT 10;\n"
                "```\n"
                "Returns the 10 most recent non-deleted documents."
            )
        else:
            body = (
                "Based on the documentation, the system uses a microservices architecture "
                "with Document, Search, and Analytics services. Authentication is via Bearer "
                "tokens with a 100 req/min rate limit."
            )
        return DEMO_NOTICE + body

    def multi_turn_conversation_stream(self, user_input: str) -> StreamingResponse:
        self.conversation_history.append({"role": "user", "content": user_input})
        base = self.query_stream(user_input)
        history = self.conversation_history

        def wrapped() -> Iterator[str]:
            try:
                yield from base
            finally:
                history.append({"role": "assistant", "content": base.answer})

        return StreamingResponse(
            sources=base.sources,
            tool_used=base.tool_used,
            confidence=base.confidence,
            _chunks=wrapped(),
        )

    def multi_turn_conversation(self, user_input: str) -> AgentResponse:
        self.conversation_history.append({"role": "user", "content": user_input})
        response = self.query(user_input)
        self.conversation_history.append({"role": "assistant", "content": response.answer})
        return response

    def clear_history(self) -> None:
        self.conversation_history = []

    def get_stats(self) -> dict[str, Any]:
        return {
            "vector_store_stats": self.vector_store.get_document_stats(),
            "conversation_turns": len(self.conversation_history),
            "total_queries": sum(1 for h in self.conversation_history if h["role"] == "user"),
        }
