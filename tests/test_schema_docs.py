"""Unit tests for the schema documentation generator."""
from __future__ import annotations

import pytest

from data_copilot.config import Settings
from data_copilot.data_loader import load_sample_warehouse
from data_copilot.schema_docs import (
    ColumnDoc,
    generate_schema_docs,
    render_schema_markdown,
    write_schema_docs,
)


@pytest.fixture
def loaded_settings(tmp_path) -> Settings:
    settings = Settings(
        data_dir=tmp_path / "data",
        vector_store_dir=tmp_path / "data" / "chroma",
        eval_dir=tmp_path / "eval",
        duckdb_path=tmp_path / "data" / "warehouse.duckdb",
    )
    load_sample_warehouse(settings=settings, seed=11)
    return settings


def test_generate_schema_docs_raises_without_warehouse(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        vector_store_dir=tmp_path / "data" / "chroma",
        eval_dir=tmp_path / "eval",
        duckdb_path=tmp_path / "data" / "warehouse.duckdb",
    )

    with pytest.raises(FileNotFoundError):
        generate_schema_docs(settings=settings)


def test_generate_schema_docs_covers_every_table(loaded_settings):
    table_docs = generate_schema_docs(settings=loaded_settings)

    table_names = {table_doc.name for table_doc in table_docs}
    assert table_names == {"customers", "products", "orders", "order_items"}


def test_primary_keys_are_detected(loaded_settings):
    table_docs = {table_doc.name: table_doc for table_doc in generate_schema_docs(settings=loaded_settings)}

    assert table_docs["orders"].primary_key_columns() == ["order_id"]
    assert table_docs["order_items"].primary_key_columns() == ["order_item_id"]


def test_foreign_keys_are_detected(loaded_settings):
    table_docs = {table_doc.name: table_doc for table_doc in generate_schema_docs(settings=loaded_settings)}

    orders_fks = dict(table_docs["orders"].foreign_keys())
    assert orders_fks["customer_id"] == "customers.customer_id"

    order_items_fks = dict(table_docs["order_items"].foreign_keys())
    assert order_items_fks["order_id"] == "orders.order_id"
    assert order_items_fks["product_id"] == "products.product_id"


def test_row_counts_match_loaded_data(loaded_settings):
    table_docs = {table_doc.name: table_doc for table_doc in generate_schema_docs(settings=loaded_settings)}

    assert table_docs["customers"].row_count == 40
    assert table_docs["orders"].row_count == 150


def test_column_doc_describe_includes_tags():
    pk_column = ColumnDoc(name="order_id", data_type="INTEGER", nullable=False, is_primary_key=True)
    fk_column = ColumnDoc(
        name="customer_id", data_type="INTEGER", nullable=False, references="customers.customer_id"
    )
    plain_column = ColumnDoc(name="status", data_type="VARCHAR", nullable=True)

    assert pk_column.describe() == "order_id: INTEGER (primary key, not null)"
    assert fk_column.describe() == "customer_id: INTEGER (references customers.customer_id, not null)"
    assert plain_column.describe() == "status: VARCHAR"


def test_render_schema_markdown_contains_every_table(loaded_settings):
    table_docs = generate_schema_docs(settings=loaded_settings)
    markdown = render_schema_markdown(table_docs)

    assert "# Warehouse Schema Documentation" in markdown
    for table_doc in table_docs:
        assert f"## Table: {table_doc.name}" in markdown


def test_write_schema_docs_creates_markdown_file(loaded_settings):
    output_path = write_schema_docs(settings=loaded_settings)

    assert output_path == loaded_settings.data_dir / "schema_docs.md"
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "Primary key: order_id" in content
    assert "customer_id -> customers.customer_id" in content


def test_cli_generate_schema_docs_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_COPILOT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DATA_COPILOT_VECTOR_DIR", str(tmp_path / "data" / "chroma"))
    monkeypatch.setenv("DATA_COPILOT_DUCKDB_PATH", str(tmp_path / "data" / "warehouse.duckdb"))

    from data_copilot.cli import main
    from data_copilot.data_loader import load_sample_warehouse
    from data_copilot.config import get_settings

    load_sample_warehouse(settings=get_settings())

    exit_code = main(["--generate-schema-docs"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "schema docs written to" in captured.out
    assert (tmp_path / "data" / "schema_docs.md").exists()


def test_cli_generate_schema_docs_without_warehouse_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_COPILOT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DATA_COPILOT_VECTOR_DIR", str(tmp_path / "data" / "chroma"))
    monkeypatch.setenv("DATA_COPILOT_DUCKDB_PATH", str(tmp_path / "data" / "warehouse_missing.duckdb"))

    from data_copilot.cli import main

    exit_code = main(["--generate-schema-docs"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "No DuckDB database" in captured.err
