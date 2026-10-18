"""Retrieval chain: turn a natural-language question into schema snippets.

This is the thin chain that sits between the user's question and the
upcoming SQL-generation prompt template (next roadmap step): it queries the
Chroma vector store built by ``data_copilot.vector_store``, applies a
relevance cutoff so unrelated tables don't get dragged into the prompt just
to fill ``max_snippets``, and renders the surviving table docs into a single
context block ready to be injected into a prompt.

Deliberately framework-free for now: no LangChain ``BaseRetriever``, no LLM
call. That keeps the chain deterministic and runnable offline/in CI, matching
``vector_store``'s dependency-light embedding. The LangGraph generation loop
(a later roadmap step) can wrap ``retrieve_context``/``retrieve_snippets``
behind a ``BaseRetriever`` or a LangGraph node without changing this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import chromadb
from chromadb.api.types import EmbeddingFunction

from data_copilot.config import Settings, get_settings
from data_copilot.vector_store import query_schema_docs

# Cosine distance cutoff for the deterministic hash embedding: hits above
# this distance are treated as noise rather than genuinely relevant context.
# It's generous on purpose (the hash embedding is a coarse bag-of-words
# signal, not a trained semantic model) — it exists to drop clearly
# unrelated tables, not to be a precise relevance score.
DEFAULT_MAX_DISTANCE = 1.6
DEFAULT_MAX_SNIPPETS = 3


@dataclass(frozen=True)
class RetrievedSnippet:
    """One table's schema doc, retrieved for a given question."""

    table: str
    content: str
    distance: float


def retrieve_snippets(
    question: str,
    settings: Settings | None = None,
    client: chromadb.ClientAPI | None = None,
    embedding_function: EmbeddingFunction | None = None,
    max_snippets: int = DEFAULT_MAX_SNIPPETS,
    max_distance: float | None = DEFAULT_MAX_DISTANCE,
) -> list[RetrievedSnippet]:
    """Return the schema snippets most relevant to ``question``, most relevant first.

    ``max_distance`` (cosine distance, lower is more relevant) filters out
    weak matches after retrieval; pass ``None`` to disable filtering and keep
    every one of the ``max_snippets`` nearest tables regardless of distance.
    """
    settings = settings or get_settings()
    raw_results = query_schema_docs(
        question,
        settings=settings,
        client=client,
        embedding_function=embedding_function,
        n_results=max_snippets,
    )

    snippets = [
        RetrievedSnippet(table=result["table"], content=result["document"], distance=result["distance"])
        for result in raw_results
    ]

    if max_distance is not None:
        snippets = [snippet for snippet in snippets if snippet.distance <= max_distance]

    return snippets


def format_context(snippets: list[RetrievedSnippet]) -> str:
    """Render retrieved snippets into a single Markdown block for prompt injection."""
    if not snippets:
        return "No relevant schema context was found for this question."
    return "\n\n".join(snippet.content for snippet in snippets)


def retrieve_context(
    question: str,
    settings: Settings | None = None,
    client: chromadb.ClientAPI | None = None,
    embedding_function: EmbeddingFunction | None = None,
    max_snippets: int = DEFAULT_MAX_SNIPPETS,
    max_distance: float | None = DEFAULT_MAX_DISTANCE,
) -> str:
    """Question -> ready-to-inject schema context string.

    Convenience wrapper combining ``retrieve_snippets`` and ``format_context``
    — the full "question -> relevant schema snippets" chain in one call. The
    SQL-generation prompt template (next roadmap step) calls this directly.
    """
    snippets = retrieve_snippets(
        question,
        settings=settings,
        client=client,
        embedding_function=embedding_function,
        max_snippets=max_snippets,
        max_distance=max_distance,
    )
    return format_context(snippets)


if __name__ == "__main__":
    import sys

    asked = " ".join(sys.argv[1:]) or "What columns does the orders table have?"
    print(f"[data-copilot] question: {asked}")
    print(retrieve_context(asked))
