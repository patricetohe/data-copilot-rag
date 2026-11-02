"""Unit tests for the SQL validator."""
from __future__ import annotations

import pytest

from data_copilot.config import Settings
from data_copilot.sql_validator import SQLValidationError, ValidatedQuery, validate_sql

SETTINGS = Settings(max_result_rows=10)


def test_valid_select_gets_limit_added():
    result = validate_sql("SELECT * FROM customers", settings=SETTINGS)

    assert isinstance(result, ValidatedQuery)
    assert "LIMIT 10" in result.sql
    assert result.row_limit == 10


def test_existing_limit_within_cap_is_kept():
    result = validate_sql("SELECT * FROM customers LIMIT 5", settings=SETTINGS)

    assert "LIMIT 5" in result.sql
    assert result.row_limit == 5


def test_existing_limit_above_cap_is_clamped():
    result = validate_sql("SELECT * FROM customers LIMIT 10000", settings=SETTINGS)

    assert "LIMIT 10" in result.sql
    assert "LIMIT 10000" not in result.sql
    assert result.row_limit == 10


def test_cte_select_is_allowed():
    result = validate_sql(
        "WITH recent AS (SELECT * FROM orders) SELECT * FROM recent", settings=SETTINGS
    )

    assert "LIMIT 10" in result.sql


def test_union_select_is_allowed_and_limited():
    result = validate_sql(
        "SELECT customer_id FROM customers UNION SELECT customer_id FROM orders",
        settings=SETTINGS,
    )

    assert "LIMIT 10" in result.sql


def test_aggregate_query_still_gets_a_limit():
    result = validate_sql("SELECT COUNT(*) FROM customers", settings=SETTINGS)

    assert result.row_limit == 10


@pytest.mark.parametrize(
    "mutating_sql",
    [
        "INSERT INTO customers VALUES (1, 'a')",
        "UPDATE customers SET full_name = 'x'",
        "DELETE FROM customers",
        "DROP TABLE customers",
        "CREATE TABLE evil (id INT)",
        "ALTER TABLE customers ADD COLUMN evil INT",
        "ATTACH 'other.duckdb' AS other",
        "PRAGMA table_info('customers')",
    ],
)
def test_mutating_statements_are_rejected(mutating_sql):
    with pytest.raises(SQLValidationError):
        validate_sql(mutating_sql, settings=SETTINGS)


def test_multiple_statements_are_rejected():
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT 1; SELECT 2", settings=SETTINGS)


def test_empty_sql_is_rejected():
    with pytest.raises(SQLValidationError):
        validate_sql("   ", settings=SETTINGS)


def test_unparseable_sql_is_rejected():
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT FROM WHERE *** garbage ===", settings=SETTINGS)


def test_insert_hidden_in_a_cte_is_still_rejected():
    # A read-only root (Select) wrapping a mutating statement somewhere
    # inside it should still be caught by the whole-tree scan, not just the
    # root-type check.
    with pytest.raises(SQLValidationError):
        validate_sql(
            "SELECT * FROM (INSERT INTO customers VALUES (1, 'a') RETURNING *) AS t",
            settings=SETTINGS,
        )


def test_default_dialect_is_duckdb():
    result = validate_sql("SELECT * FROM customers", settings=SETTINGS)

    assert result.dialect == "duckdb"
