"""Unit tests for the sample e-commerce dataset loader."""
from __future__ import annotations

import duckdb
import pytest

from data_copilot.config import Settings
from data_copilot.data_loader import (
    SeedCounts,
    generate_dataset,
    load_sample_warehouse,
    table_row_counts,
)


@pytest.fixture
def tmp_settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        vector_store_dir=tmp_path / "data" / "chroma",
        eval_dir=tmp_path / "eval",
        duckdb_path=tmp_path / "data" / "warehouse.duckdb",
    )


def test_generate_dataset_is_deterministic():
    first = generate_dataset(seed=7)
    second = generate_dataset(seed=7)

    assert first == second
    assert set(first.keys()) == {"customers", "products", "orders", "order_items"}


def test_generate_dataset_respects_counts():
    counts = SeedCounts(customers=5, orders=10, max_items_per_order=2)
    dataset = generate_dataset(seed=1, counts=counts)

    assert len(dataset["customers"]) == 5
    assert len(dataset["orders"]) == 10
    assert len(dataset["order_items"]) >= 10  # at least one item per order


def test_load_sample_warehouse_creates_db_file(tmp_settings):
    db_path = load_sample_warehouse(settings=tmp_settings)

    assert db_path == tmp_settings.duckdb_path
    assert db_path.exists()


def test_load_sample_warehouse_populates_expected_tables(tmp_settings):
    load_sample_warehouse(settings=tmp_settings, seed=3)

    counts = table_row_counts(settings=tmp_settings)

    assert counts["customers"] == SeedCounts().customers
    assert counts["products"] == len(generate_dataset()["products"])
    assert counts["orders"] == SeedCounts().orders
    assert counts["order_items"] > 0


def test_foreign_keys_reference_valid_rows(tmp_settings):
    load_sample_warehouse(settings=tmp_settings, seed=9)

    connection = duckdb.connect(str(tmp_settings.duckdb_path), read_only=True)
    try:
        orphan_orders = connection.execute(
            "SELECT COUNT(*) FROM orders o LEFT JOIN customers c "
            "ON o.customer_id = c.customer_id WHERE c.customer_id IS NULL"
        ).fetchone()[0]
        orphan_items = connection.execute(
            "SELECT COUNT(*) FROM order_items oi LEFT JOIN orders o "
            "ON oi.order_id = o.order_id WHERE o.order_id IS NULL"
        ).fetchone()[0]
    finally:
        connection.close()

    assert orphan_orders == 0
    assert orphan_items == 0


def test_load_sample_warehouse_is_idempotent(tmp_settings):
    load_sample_warehouse(settings=tmp_settings, seed=5)
    first_counts = table_row_counts(settings=tmp_settings)

    load_sample_warehouse(settings=tmp_settings, seed=5)
    second_counts = table_row_counts(settings=tmp_settings)

    assert first_counts == second_counts


def test_cli_load_sample_data_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_COPILOT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DATA_COPILOT_VECTOR_DIR", str(tmp_path / "data" / "chroma"))
    monkeypatch.setenv("DATA_COPILOT_DUCKDB_PATH", str(tmp_path / "data" / "warehouse.duckdb"))

    from data_copilot.cli import main

    exit_code = main(["--load-sample-data"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "sample warehouse loaded at" in captured.out
    assert (tmp_path / "data" / "warehouse.duckdb").exists()
