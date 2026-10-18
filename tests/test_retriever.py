"""Unit tests for the retrieval chain (question -> relevant schema snippets)."""
from __future__ import annotations

import pytest

from data_copilot.config import Settings
from data_copilot.data_loader import load_sample_warehouse
from data_copilot.retriever import (
    RetrievedSnippet,
    format_context,
    retrieve_context,
    retrieve_snippets,
)
from data_copilot.vector_store import embed_schema_docs


@pytest.fixture
def embedded_settings(tmp_path) -> Settings:
    settings = Settings(
        data_dir=tmp_path / "data",
        vector_store_dir=tmp_path / "data" / "chroma",
        eval_dir=tmp_path / "eval",
        duckdb_path=tmp_path / "data" / "warehouse.duckdb",
    )
    load_sample_warehouse(settings=settings, seed=11)
    embed_schema_docs(settings=settings)
    return settings


def test_retrieve_snippets_finds_relevant_table(embedded_settings):
    snippets = retrieve_snippets(
        "Which column stores the customer's email address?",
        settings=embedded_settings,
        max_snippets=1,
    )

    assert len(snippets) == 1
    assert isinstance(snippets[0], RetrievedSnippet)
    assert snippets[0].table == "customers"
    assert "email" in snippets[0].content.lower()


def test_retrieve_snippets_respects_max_snippets(embedded_settings):
    snippets = retrieve_snippets("order amount and status", settings=embedded_settings, max_snippets=2)

    assert len(snippets) == 2
    # Ordered most relevant first.
    assert snippets[0].distance <= snippets[1].distance


def test_retrieve_snippets_filters_by_max_distance(embedded_settings):
    # An absurdly tight cutoff should drop every candidate.
    snippets = retrieve_snippets(
        "order amount and status",
        settings=embedded_settings,
        max_snippets=3,
        max_distance=-1.0,
    )

    assert snippets == []


def test_retrieve_snippets_without_distance_filter_keeps_all_candidates(embedded_settings):
    snippets = retrieve_snippets(
        "order amount and status",
        settings=embedded_settings,
        max_snippets=4,
        max_distance=None,
    )

    assert len(snippets) == 4


def test_retrieve_snippets_empty_before_embedding(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        vector_store_dir=tmp_path / "data" / "chroma",
        eval_dir=tmp_path / "eval",
        duckdb_path=tmp_path / "data" / "warehouse.duckdb",
    )
    load_sample_warehouse(settings=settings, seed=3)

    assert retrieve_snippets("anything at all", settings=settings) == []


def test_format_context_joins_snippet_content():
    snippets = [
        RetrievedSnippet(table="orders", content="## Table: orders\n...", distance=0.1),
        RetrievedSnippet(table="customers", content="## Table: customers\n...", distance=0.3),
    ]

    context = format_context(snippets)

    assert "## Table: orders" in context
    assert "## Table: customers" in context
    assert context.index("orders") < context.index("customers")


def test_format_context_handles_no_matches():
    assert format_context([]) == "No relevant schema context was found for this question."


def test_retrieve_context_returns_rendered_schema_for_table(embedded_settings):
    context = retrieve_context(
        "What is the primary key of the products table?",
        settings=embedded_settings,
        max_snippets=1,
    )

    assert "## Table: products" in context
    assert "Primary key" in context


def test_cli_retrieve_context_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_COPILOT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DATA_COPILOT_VECTOR_DIR", str(tmp_path / "data" / "chroma"))
    monkeypatch.setenv("DATA_COPILOT_DUCKDB_PATH", str(tmp_path / "data" / "warehouse.duckdb"))

    from data_copilot.cli import main
    from data_copilot.config import get_settings
    from data_copilot.data_loader import load_sample_warehouse
    from data_copilot.vector_store import embed_schema_docs

    settings = get_settings()
    load_sample_warehouse(settings=settings)
    embed_schema_docs(settings=settings)

    exit_code = main(["--retrieve-context", "customer email address"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "schema context for" in captured.out
    assert "## Table: customers" in captured.out
