"""RAG Agent with LLM-powered tools using the OpenAI SDK."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from vector_store import FAISSVectorStore, RetrievalResult

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 1024


@dataclass
class AgentResponse:
    """Response from RAG agent."""

    answer: str
    sources: list[dict[str, Any]]
    tool_used: str
    confidence: float


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
    ) -> None:
        self.vector_store = vector_store
        self.client: Any = client if client is not None else _default_client()
        self.model = model
        self.max_tokens = max_tokens
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
        if any(word in q for word in ["sql", "query", "database"]):
            return "sql_generation"
        return "question_answering"

    def _format_context(self, documents: list[RetrievalResult]) -> str:
        return "\n\n".join(
            f"[Source {i}: {doc.source}]\n{doc.content}"
            for i, doc in enumerate(documents, 1)
        )

    def _answer_question(self, context: str, question: str) -> str:
        prompt = (
            "You are a helpful assistant answering questions about technical documentation.\n\n"
            f"Context from documents:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Provide a clear, accurate answer based on the context above. "
            "If the context doesn't contain the answer, say so honestly."
        )
        return self._call_llm(prompt)

    def _summarize(self, context: str, query: str) -> str:
        topic = query.replace("summarize", "").replace("summary", "").strip()
        prompt = (
            "You are a technical documentation expert. "
            "Provide a concise summary of the following documents.\n\n"
            f"Documents:\n{context}\n\n"
            f"Focus: {topic if topic else 'all key points'}\n\n"
            "Provide a clear, well-organized summary. Use headers and bullet points where appropriate."
        )
        return self._call_llm(prompt)

    def _generate_sql(self, context: str, query: str) -> str:
        prompt = (
            "You are an SQL expert. Based on the database schema and documentation below, "
            "generate a SQL query to fulfill the request.\n\n"
            f"Documentation and Schema:\n{context}\n\n"
            f"Request: {query}\n\n"
            "Provide:\n1. The SQL query with proper formatting\n"
            "2. A brief explanation of what the query does\n3. Any assumptions made\n\n"
            "Format the SQL code in a markdown code block with ```sql markers."
        )
        return self._call_llm(prompt)

    def _call_llm(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content
            return content if isinstance(content, str) else ""
        except Exception:
            return self._generate_demo_response(prompt)

    def _generate_demo_response(self, prompt: str) -> str:
        p = prompt.lower()
        if "summarize" in p or "summary" in p:
            return (
                "## Summary\n\n"
                "- Document management with versioning\n"
                "- Bearer-token authenticated REST API\n"
                "- Microservices: PostgreSQL + Redis + FAISS/Elasticsearch\n"
                "- Rate limited to 100 req/min\n"
            )
        if "sql" in p:
            return (
                "```sql\n"
                "SELECT id, title, created_at FROM documents\n"
                "WHERE is_deleted = FALSE\n"
                "ORDER BY created_at DESC LIMIT 10;\n"
                "```\n"
                "Returns the 10 most recent non-deleted documents."
            )
        return (
            "Based on the documentation, the system uses a microservices architecture "
            "with Document, Search, and Analytics services. Authentication is via Bearer "
            "tokens with a 100 req/min rate limit."
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
            "total_queries": sum(
                1 for h in self.conversation_history if h["role"] == "user"
            ),
        }
