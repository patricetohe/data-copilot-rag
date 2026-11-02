"""SQL validator: read-only, single statement, row limit enforced.

Sits between the text-to-SQL prompt template (``data_copilot.prompt``) and
the sandbox executor (a later roadmap step): the LLM's raw SQL text goes
through ``validate_sql`` before anything is ever run against DuckDB.
``validate_sql`` rejects candidate SQL outright when it is not exactly one
read-only statement, and otherwise returns a rewritten, safe-to-execute
``ValidatedQuery`` with an enforced ``LIMIT``.

Deliberately framework-free, like ``retriever``/``prompt``: this is a pure
parse-and-check step built on ``sqlglot`` (already a dependency, slated for
the multi-dialect roadmap item), with no LLM call and no database
connection, so it is cheap to run on every candidate and to unit test in
isolation.
"""
from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from data_copilot.config import Settings, get_settings

# DuckDB is the only warehouse Data Copilot targets today (see
# data_copilot.data_loader); the multi-dialect roadmap item will make this
# configurable instead of hard-coded.
DEFAULT_DIALECT = "duckdb"

# Root expression types that represent a read-only query. ``Select`` also
# covers a `WITH ... SELECT` CTE (sqlglot attaches the CTEs to the Select's
# own ``with`` arg rather than producing a separate node type), and
# ``Union``/``Intersect``/``Except`` cover a set operation between two
# otherwise-read-only SELECTs.
_READ_ONLY_ROOT_TYPES: tuple[type[exp.Expression], ...] = (
    exp.Select,
    exp.Union,
    exp.Intersect,
    exp.Except,
)

# Node types that mutate data, schema, or session/connection state. Checked
# against every node in the parsed tree (not just the root) so that a
# mutating statement smuggled in through an unexpected sub-node is still
# caught, not just a mutating statement at the top level.
_MUTATING_NODE_TYPES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Merge,
    exp.TruncateTable,
    exp.Attach,
    exp.Copy,
    exp.Grant,
    exp.Command,
    exp.Pragma,
    exp.Use,
    exp.Set,
)


class SQLValidationError(ValueError):
    """Raised when candidate SQL fails the read-only/single-statement check.

    Carries the same message a caller would want to show back to the
    self-correcting generation loop (a later roadmap step), so it can be
    fed straight back to the LLM as the reason the previous attempt failed.
    """


@dataclass(frozen=True)
class ValidatedQuery:
    """A candidate query that passed validation, ready for the sandbox executor.

    Attributes:
        sql: The statement, rewritten with an enforced ``LIMIT`` and
            rendered back out in ``dialect``. This is what the sandbox
            executor (a later roadmap step) should actually run -- never
            the original, unvalidated candidate text.
        row_limit: The ``LIMIT`` enforced on ``sql``.
        dialect: The SQL dialect ``sql`` is rendered in.
    """

    sql: str
    row_limit: int
    dialect: str = DEFAULT_DIALECT


def _parse_single_statement(candidate_sql: str, dialect: str) -> exp.Expression:
    """Parse ``candidate_sql`` and return its one statement, or raise."""
    try:
        statements = [
            statement
            for statement in sqlglot.parse(candidate_sql, read=dialect)
            if statement is not None
        ]
    except sqlglot.errors.ParseError as error:
        raise SQLValidationError(f"Could not parse SQL: {error}") from error

    if len(statements) == 0:
        raise SQLValidationError("No SQL statement found.")
    if len(statements) > 1:
        raise SQLValidationError(
            f"Expected exactly one SQL statement, found {len(statements)}."
        )
    return statements[0]


def _ensure_read_only(statement: exp.Expression) -> None:
    """Raise ``SQLValidationError`` unless ``statement`` is read-only."""
    if not isinstance(statement, _READ_ONLY_ROOT_TYPES):
        raise SQLValidationError(
            "Only read-only SELECT statements are allowed, got "
            f"{type(statement).__name__}."
        )

    for node in statement.walk():
        # Older/newer sqlglot releases have at times yielded (node, parent,
        # key) tuples from walk() instead of bare nodes; handle both so this
        # check keeps working across the pinned-vs-latest sqlglot versions.
        current = node[0] if isinstance(node, tuple) else node
        if isinstance(current, _MUTATING_NODE_TYPES):
            raise SQLValidationError(
                "Only read-only SELECT statements are allowed, found a "
                f"{type(current).__name__} inside the query."
            )


def _enforce_row_limit(
    statement: exp.Expression, max_result_rows: int, dialect: str
) -> int:
    """Clamp or add ``LIMIT`` on ``statement`` in place; return the enforced value."""
    existing_limit = statement.args.get("limit")
    if existing_limit is None:
        statement.set("limit", exp.Limit(expression=exp.Literal.number(max_result_rows)))
        return max_result_rows

    try:
        requested = int(existing_limit.expression.this)
    except (AttributeError, TypeError, ValueError):
        # A non-literal LIMIT (e.g. a bound parameter) can't be compared
        # against max_result_rows, so replace it outright with the safe cap
        # rather than risk letting an unbounded result set through.
        statement.set("limit", exp.Limit(expression=exp.Literal.number(max_result_rows)))
        return max_result_rows

    enforced = min(requested, max_result_rows)
    statement.set("limit", exp.Limit(expression=exp.Literal.number(enforced)))
    return enforced


def validate_sql(
    candidate_sql: str,
    settings: Settings | None = None,
    dialect: str = DEFAULT_DIALECT,
) -> ValidatedQuery:
    """Validate LLM-generated SQL and return a safe-to-execute ``ValidatedQuery``.

    Raises ``SQLValidationError`` when ``candidate_sql`` is not exactly one
    read-only statement. Otherwise returns the statement rewritten with its
    ``LIMIT`` clamped to (or set to, if missing) ``settings.max_result_rows``.
    """
    settings = settings or get_settings()

    statement = _parse_single_statement(candidate_sql, dialect)
    _ensure_read_only(statement)
    row_limit = _enforce_row_limit(statement, settings.max_result_rows, dialect)

    return ValidatedQuery(
        sql=statement.sql(dialect=dialect),
        row_limit=row_limit,
        dialect=dialect,
    )
