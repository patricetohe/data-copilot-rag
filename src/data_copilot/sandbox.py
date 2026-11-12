"""Sandbox execution of generated SQL against the sample DuckDB warehouse.

This is the execution half of the "generate SQL -> execute -> self-correct"
loop described in ROADMAP.md: given a SQL string, run it safely against the
warehouse and return a structured result (or a structured error) instead of
raising, so a future agent loop can inspect ``SandboxResult.error`` and decide
whether to retry with a corrected query.

Safety is defense-in-depth and intentionally independent of any separate SQL
validator that may run before this module: even if a query reaches the
sandbox unvalidated, execution still happens against a **read-only** DuckDB
connection, rejects more than one statement, and truncates returned rows to
``Settings.max_result_rows`` (or an explicit ``max_rows`` override).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import duckdb

from data_copilot.config import Settings, get_settings


class SandboxError(Exception):
    """Raised for sandbox-level rejections (as opposed to DuckDB query errors).

    Kept distinct from a failed query so callers can tell "this was never
    sent to the database" apart from "the database rejected it" when
    building a self-correction prompt.
    """


@dataclass
class SandboxResult:
    """Outcome of running one query through :func:`run_query_in_sandbox`.

    Exactly one of (``columns``/``rows`` populated) or ``error`` is set on
    success vs. failure; ``ok`` is the cheap way to branch on which.
    """

    sql: str
    ok: bool
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    elapsed_ms: float = 0.0
    error: str | None = None


def _strip_trailing_semicolon(sql: str) -> str:
    stripped = sql.strip()
    if stripped.endswith(";"):
        stripped = stripped[:-1].strip()
    return stripped


def _reject_multiple_statements(sql: str) -> None:
    """Reject more than one statement, ignoring a single trailing semicolon.

    This is a simple guard, not a full SQL parser: it splits on ``;`` and
    checks that at most one non-empty segment remains. Good enough to stop
    the common "SELECT ...; DROP TABLE ..." injection shape before it ever
    reaches DuckDB.
    """
    segments = [segment.strip() for segment in sql.split(";")]
    non_empty = [segment for segment in segments if segment]
    if len(non_empty) > 1:
        raise SandboxError("Sandbox only accepts a single SQL statement.")


def run_query_in_sandbox(
    sql: str,
    settings: Settings | None = None,
    max_rows: int | None = None,
) -> SandboxResult:
    """Execute ``sql`` read-only against the warehouse and cap the row count.

    Never raises for a query-execution failure (a malformed or disallowed
    query comes back as ``SandboxResult(ok=False, error=...)``) so a caller
    can feed that message straight into a self-correction prompt. Sandbox-
    level rejections (multiple statements, an empty query) are raised as
    :class:`SandboxError` immediately, since those mean the request itself
    was invalid rather than that DuckDB rejected it.
    """
    settings = settings or get_settings()
    row_cap = max_rows if max_rows is not None else settings.max_result_rows

    cleaned_sql = _strip_trailing_semicolon(sql)
    if not cleaned_sql:
        raise SandboxError("Sandbox received an empty SQL statement.")
    _reject_multiple_statements(sql)

    started = time.perf_counter()
    try:
        connection = duckdb.connect(str(settings.duckdb_path), read_only=True)
    except duckdb.Error as error:
        elapsed_ms = (time.perf_counter() - started) * 1000
        return SandboxResult(sql=cleaned_sql, ok=False, elapsed_ms=elapsed_ms, error=str(error))

    try:
        try:
            cursor = connection.execute(cleaned_sql)
        except duckdb.Error as error:
            elapsed_ms = (time.perf_counter() - started) * 1000
            return SandboxResult(sql=cleaned_sql, ok=False, elapsed_ms=elapsed_ms, error=str(error))

        columns = [description[0] for description in cursor.description] if cursor.description else []

        # Fetch one extra row so we can tell "exactly row_cap rows" apart
        # from "there were more rows and we truncated them".
        fetched = cursor.fetchmany(row_cap + 1) if row_cap >= 0 else cursor.fetchall()
        truncated = row_cap >= 0 and len(fetched) > row_cap
        rows = fetched[:row_cap] if truncated else fetched

        elapsed_ms = (time.perf_counter() - started) * 1000
        return SandboxResult(
            sql=cleaned_sql,
            ok=True,
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            elapsed_ms=elapsed_ms,
        )
    finally:
        connection.close()
