"""Unit tests for sandboxed SQL execution against the sample warehouse."""
from __future__ import annotations

import pytest

from data_copilot.config import Settings
from data_copilot.data_loader import load_sample_warehouse
from data_copilot.sandbox import SandboxError, run_query_in_sandbox


@pytest.fixture
def warehouse_settings(tmp_path) -> Settings:
    settings = Settings(
        data_dir=tmp_path / "data",
        vector_store_dir=tmp_path / "data" / "chroma",
        eval_dir=tmp_path / "eval",
        duckdb_path=tmp_path / "data" / "warehouse.duckdb",
        max_result_rows=5,
    )
    load_sample_warehouse(settings=settings, seed=1)
    return settings


def test_select_query_returns_rows(warehouse_settings):
    result = run_query_in_sandbox("SELECT customer_id, full_name FROM customers ORDER BY customer_id", settings=warehouse_settings)

    assert result.ok
    assert result.error is None
    assert result.columns == ["customer_id", "full_name"]
    assert result.row_count == 5  # capped by max_result_rows
    assert result.truncated is True


def test_row_cap_not_truncated_when_fewer_rows(warehouse_settings):
    result = run_query_in_sandbox("SELECT * FROM products WHERE product_id = 1", settings=warehouse_settings)

    assert result.ok
    assert result.row_count == 1
    assert result.truncated is False


def test_explicit_max_rows_overrides_settings(warehouse_settings):
    result = run_query_in_sandbox(
        "SELECT * FROM customers ORDER BY customer_id", settings=warehouse_settings, max_rows=2
    )

    assert result.ok
    assert result.row_count == 2
    assert result.truncated is True


def test_invalid_sql_returns_structured_error_instead_of_raising(warehouse_settings):
    result = run_query_in_sandbox("SELECT * FROM table_that_does_not_exist", settings=warehouse_settings)

    assert result.ok is False
    assert result.error is not None
    assert result.rows == []


def test_write_statement_is_rejected_by_read_only_connection(warehouse_settings):
    result = run_query_in_sandbox("DELETE FROM customers WHERE customer_id = 1", settings=warehouse_settings)

    assert result.ok is False
    assert result.error is not None


def test_multiple_statements_are_rejected(warehouse_settings):
    with pytest.raises(SandboxError):
        run_query_in_sandbox("SELECT 1; SELECT 2", settings=warehouse_settings)


def test_single_statement_with_trailing_semicolon_is_allowed(warehouse_settings):
    result = run_query_in_sandbox("SELECT 1;", settings=warehouse_settings)

    assert result.ok
    assert result.rows == [(1,)]


def test_empty_sql_is_rejected(warehouse_settings):
    with pytest.raises(SandboxError):
        run_query_in_sandbox("   ", settings=warehouse_settings)


def test_cli_run_sql_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_COPILOT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DATA_COPILOT_VECTOR_DIR", str(tmp_path / "data" / "chroma"))
    monkeypatch.setenv("DATA_COPILOT_DUCKDB_PATH", str(tmp_path / "data" / "warehouse.duckdb"))

    from data_copilot.cli import main

    load_sample_warehouse(settings=Settings(
        data_dir=tmp_path / "data",
        vector_store_dir=tmp_path / "data" / "chroma",
        eval_dir=tmp_path / "eval",
        duckdb_path=tmp_path / "data" / "warehouse.duckdb",
    ))

    exit_code = main(["--run-sql", "SELECT COUNT(*) AS n FROM products"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "n" in captured.out
