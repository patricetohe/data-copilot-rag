"""Embed schema documentation into a Chroma vector store.

Chunks each table's Markdown documentation (see ``data_copilot.schema_docs``)
into one document per table and upserts it into a persistent Chroma
collection, keyed by table name. The next roadmap step (the retrieval chain)
will query this collection to pull the schema context relevant to a given
natural-language question before handing it to the SQL-generation prompt.

Embeddings: this module ships with ``DeterministicHashEmbedding``, a small,
fully offline, dependency-free embedding function (feature hashing over
lowercased tokens, L2-normalized). It has no external model to download and
no API key to configure, so the pipeline stays deterministic and runnable in
CI. It is good enough to validate retrieval end-to-end (semantically related
text hashes into overlapping buckets), but a production deployment should
swap it for a hosted/local embedding model (OpenAI, Cohere,
sentence-transformers, ...) by passing a different ``embedding_function`` to
the functions below.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Sequence

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from data_copilot.config import Settings, get_settings
from data_copilot.schema_docs import TableDoc, generate_schema_docs

COLLECTION_NAME = "schema_docs"

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class DeterministicHashEmbedding(EmbeddingFunction):
    """Offline bag-of-words embedding via feature hashing.

    Deterministic and dependency-free: no model download, no API key. Tokens
    are lowercased, hashed into a fixed-size vector (with a hashed sign bit
    to reduce collision bias), and the result is L2-normalized so cosine
    similarity behaves sensibly.
    """

    def __init__(self, dimensions: int = 4096) -> None:
        self.dimensions = dimensions

    @staticmethod
    def name() -> str:
        return "deterministic_hash"

    def get_config(self) -> dict:
        return {"dimensions": self.dimensions}

    @staticmethod
    def build_from_config(config: dict) -> "DeterministicHashEmbedding":
        return DeterministicHashEmbedding(dimensions=config.get("dimensions", 4096))

    def _tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in _TOKEN_RE.findall(text)]

    def _vectorize(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token, count in Counter(self._tokenize(text)).items():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign * count
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def __call__(self, input: Documents) -> Embeddings:
        return [self._vectorize(text) for text in input]


def table_doc_id(table_doc: TableDoc) -> str:
    """Stable Chroma document id for a given table."""
    return f"table::{table_doc.name}"


def get_chroma_client(settings: Settings | None = None) -> chromadb.ClientAPI:
    """Open (creating if needed) the persistent Chroma client for this project."""
    settings = settings or get_settings()
    settings.ensure_dirs()
    return chromadb.PersistentClient(path=str(settings.vector_store_dir))


def get_schema_collection(
    client: chromadb.ClientAPI,
    embedding_function: EmbeddingFunction | None = None,
) -> Collection:
    """Get or create the ``schema_docs`` collection.

    The embedding function is always passed explicitly (never relied on from
    collection metadata), so this works the same whether the collection is
    brand new or was created by an earlier process.
    """
    embedding_function = embedding_function or DeterministicHashEmbedding()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )


def embed_schema_docs(
    table_docs: Sequence[TableDoc] | None = None,
    settings: Settings | None = None,
    client: chromadb.ClientAPI | None = None,
    embedding_function: EmbeddingFunction | None = None,
) -> Collection:
    """Chunk the schema docs (one chunk per table) and upsert them into Chroma.

    Re-running this is idempotent: each table is upserted by its stable id,
    so the collection count stays equal to the number of tables no matter
    how many times this is called (e.g. after the warehouse data changes).

    Raises:
        FileNotFoundError: if the warehouse hasn't been loaded yet (bubbled
            up from ``generate_schema_docs`` when ``table_docs`` isn't
            supplied directly).
    """
    settings = settings or get_settings()
    if table_docs is None:
        table_docs = generate_schema_docs(settings=settings)

    client = client or get_chroma_client(settings=settings)
    collection = get_schema_collection(client, embedding_function=embedding_function)

    if not table_docs:
        return collection

    collection.upsert(
        ids=[table_doc_id(table_doc) for table_doc in table_docs],
        documents=[table_doc.to_markdown() for table_doc in table_docs],
        metadatas=[
            {"table": table_doc.name, "row_count": table_doc.row_count or 0}
            for table_doc in table_docs
        ],
    )
    return collection


def query_schema_docs(
    question: str,
    settings: Settings | None = None,
    client: chromadb.ClientAPI | None = None,
    embedding_function: EmbeddingFunction | None = None,
    n_results: int = 3,
) -> list[dict]:
    """Return the ``n_results`` table docs most relevant to a natural-language question.

    Each result is ``{"table": str, "document": str, "distance": float}``,
    ordered from most to least relevant. This is a thin convenience wrapper
    around the Chroma collection used to sanity-check the embeddings; the
    full retrieval chain (prompt assembly, LangChain integration) is a
    separate roadmap step.
    """
    settings = settings or get_settings()
    client = client or get_chroma_client(settings=settings)
    collection = get_schema_collection(client, embedding_function=embedding_function)

    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[question],
        n_results=min(n_results, collection.count()),
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        {"table": metadata["table"], "document": document, "distance": distance}
        for document, metadata, distance in zip(documents, metadatas, distances)
    ]


if __name__ == "__main__":
    collection = embed_schema_docs()
    print(f"[data-copilot] embedded {collection.count()} table(s) into the '{COLLECTION_NAME}' Chroma collection")
