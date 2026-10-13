"""Unit tests for embedding schema docs into the Chroma vector store."""
from __future__ import annotations

import pytest

from data_copilot.config import Settings
from data_copilot.data_loader import load_sample_warehouse
from data_copilot.schema_docs import TableDoc, generate_schema_docs
import numpy as np

from data_copilot.vector_store import (
    DeterministicHashEmbedding,
    embed_schema_docs,
    get_chroma_client,
    query_schema_docs,
    table_doc_id,
)


@pytest.fixture
def loaded_settings(tmp_path) -> Settings:
    settings = Settings(
        data_dir=tmp_path / "data",
        vector_store_dir=tmp_path / "data" / "chroma",
        eval_dir=tmp_path / "eval",
        duckdb_path=tmp_path / "data" / "warehouse.duckdb",
    )
    load_sample_warehouse(settings=settings, seed=7)
    return settings


def test_embed_schema_docs_raises_without_warehouse(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        vector_store_dir=tmp_path / "data" / "chroma",
        eval_dir=tmp_path / "eval",
        duckdb_path=tmp_path / "data" / "warehouse.duckdb",
    )

    with pytest.raises(FileNotFoundError):
        embed_schema_docs(settings=settings)


def test_embed_schema_docs_creates_one_chunk_per_table(loaded_settings):
    collection = embed_schema_docs(settings=loaded_settings)

    assert collection.count() == 4
    stored_ids = set(collection.get()["ids"])
    assert stored_ids == {
        "table::customers",
        "table::products",
        "table::orders",
        "table::order_items",
    }


def test_embed_schema_docs_is_idempotent(loaded_settings):
    table_docs = generate_schema_docs(settings=loaded_settings)
    client = get_chroma_client(settings=loaded_settings)

    embed_schema_docs(table_docs=table_docs, settings=loaded_settings, client=client)
    embed_schema_docs(table_docs=table_docs, settings=loaded_settings, client=client)

    collection = embed_schema_docs(table_docs=table_docs, settings=loaded_settings, client=client)
    assert collection.count() == 4


def test_query_schema_docs_returns_empty_before_embedding(loaded_settings):
    assert query_schema_docs("anything", settings=loaded_settings) == []


def test_query_schema_docs_finds_relevant_table(loaded_settings):
    embed_schema_docs(settings=loaded_settings)

    results = query_schema_docs(
        "Which column stores the customer's email address?",
        settings=loaded_settings,
        n_results=1,
    )

    assert len(results) == 1
    assert results[0]["table"] == "customers"
    assert "email" in results[0]["document"].lower()


def test_query_schema_docs_respects_n_results(loaded_settings):
    embed_schema_docs(settings=loaded_settings)

    results = query_schema_docs("order total amount", settings=loaded_settings, n_results=2)

    assert len(results) == 2


def test_table_doc_id_is_stable_and_namespaced():
    assert table_doc_id(TableDoc(name="orders")) == "table::orders"


def test_deterministic_hash_embedding_is_reproducible():
    embedding_function = DeterministicHashEmbedding(dimensions=64)

    first = embedding_function(["orders table with total amount"])
    second = embedding_function(["orders table with total amount"])

    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert len(first[0]) == 64


def test_cli_embed_schema_docs_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_COPILOT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DATA_COPILOT_VECTOR_DIR", str(tmp_path / "data" / "chroma"))
    monkeypatch.setenv("DATA_COPILOT_DUCKDB_PATH", str(tmp_path / "data" / "warehouse.duckdb"))

    from data_copilot.cli import main
    from data_copilot.data_loader import load_sample_warehouse
    from data_copilot.config import get_settings

    load_sample_warehouse(settings=get_settings())

    exit_code = main(["--embed-schema-docs"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "embedded 4 table(s)" in captured.out


def test_cli_embed_schema_docs_without_warehouse_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_COPILOT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DATA_COPILOT_VECTOR_DIR", str(tmp_path / "data" / "chroma"))
    monkeypatch.setenv("DATA_COPILOT_DUCKDB_PATH", str(tmp_path / "data" / "warehouse_missing.duckdb"))

    from data_copilot.cli import main

    exit_code = main(["--embed-schema-docs"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "No DuckDB database" in captured.err
