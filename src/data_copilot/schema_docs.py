"""Schema documentation generator.

Introspects the DuckDB sample warehouse (table names, columns, types,
nullability, primary keys, foreign keys, row counts) and turns that into
structured, human-readable documentation.

This is the raw material the next roadmap step will chunk and embed into a
Chroma vector store, so the retriever can hand the SQL-generation chain the
right schema context for a given natural-language question. It also doubles
as a quick, auto-refreshed reference doc for the warehouse on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from data_copilot.config import Settings, get_settings


@dataclass(frozen=True)
class ColumnDoc:
    """Documentation for a single column."""

    name: str
    data_type: str
    nullable: bool
    is_primary_key: bool = False
    references: str | None = None  # "table.column" when this column is a foreign key

    def describe(self) -> str:
        """One-line, human-readable description of this column."""
        tags: list[str] = []
        if self.is_primary_key:
            tags.append("primary key")
        if self.references:
            tags.append(f"references {self.references}")
        if not self.nullable:
            tags.append("not null")
        tag_str = f" ({', '.join(tags)})" if tags else ""
        return f"{self.name}: {self.data_type}{tag_str}"


@dataclass(frozen=True)
class TableDoc:
    """Documentation for a single table."""

    name: str
    columns: list[ColumnDoc] = field(default_factory=list)
    row_count: int | None = None

    def primary_key_columns(self) -> list[str]:
        """Names of the columns that make up this table's primary key."""
        return [column.name for column in self.columns if column.is_primary_key]

    def foreign_keys(self) -> list[tuple[str, str]]:
        """List of ``(column_name, "table.column")`` pairs for this table's FKs."""
        return [(column.name, column.references) for column in self.columns if column.references]

    def to_markdown(self) -> str:
        """Render this table's documentation as a self-contained Markdown chunk."""
        lines = [f"## Table: {self.name}"]
        if self.row_count is not None:
            lines.append(f"Row count: {self.row_count}")

        primary_key = self.primary_key_columns()
        if primary_key:
            lines.append(f"Primary key: {', '.join(primary_key)}")

        foreign_keys = self.foreign_keys()
        if foreign_keys:
            fk_desc = ", ".join(f"{column} -> {reference}" for column, reference in foreign_keys)
            lines.append(f"Foreign keys: {fk_desc}")

        lines.append("Columns:")
        lines.extend(f"- {column.describe()}" for column in self.columns)
        return "\n".join(lines)


def _fetch_table_names(connection: duckdb.DuckDBPyConnection) -> list[str]:
    rows = connection.execute(
        "SELECT table_name FROM duckdb_tables() WHERE schema_name = 'main' ORDER BY table_name"
    ).fetchall()
    return [row[0] for row in rows]


def _fetch_foreign_keys(connection: duckdb.DuckDBPyConnection, table: str) -> dict[str, str]:
    """Map local column name -> ``"ref_table.ref_column"`` for every FK on ``table``."""
    rows = connection.execute(
        "SELECT constraint_column_names, referenced_table, referenced_column_names "
        "FROM duckdb_constraints() WHERE table_name = ? AND constraint_type = 'FOREIGN KEY'",
        [table],
    ).fetchall()

    foreign_keys: dict[str, str] = {}
    for column_names, referenced_table, referenced_column_names in rows:
        for local_column, referenced_column in zip(column_names, referenced_column_names):
            foreign_keys[local_column] = f"{referenced_table}.{referenced_column}"
    return foreign_keys


def _build_table_doc(connection: duckdb.DuckDBPyConnection, table: str) -> TableDoc:
    foreign_keys = _fetch_foreign_keys(connection, table)
    column_rows = connection.execute(f'PRAGMA table_info("{table}")').fetchall()

    columns = [
        ColumnDoc(
            name=name,
            data_type=data_type,
            nullable=not bool(notnull),
            is_primary_key=bool(is_pk),
            references=foreign_keys.get(name),
        )
        for _cid, name, data_type, notnull, _default, is_pk in column_rows
    ]

    row_count = connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]

    return TableDoc(name=table, columns=columns, row_count=row_count)


def generate_schema_docs(settings: Settings | None = None) -> list[TableDoc]:
    """Introspect the sample warehouse and return one ``TableDoc`` per table.

    Raises:
        FileNotFoundError: if the warehouse hasn't been loaded yet (see
            ``data_copilot.data_loader.load_sample_warehouse``).
    """
    settings = settings or get_settings()
    if not settings.duckdb_path.exists():
        raise FileNotFoundError(
            f"No DuckDB database at {settings.duckdb_path}. "
            "Run `data-copilot --load-sample-data` first."
        )

    connection = duckdb.connect(str(settings.duckdb_path), read_only=True)
    try:
        return [_build_table_doc(connection, table) for table in _fetch_table_names(connection)]
    finally:
        connection.close()


def render_schema_markdown(table_docs: list[TableDoc]) -> str:
    """Render every table doc as one Markdown document, ready to be chunked for embedding."""
    header = "# Warehouse Schema Documentation\n\nAuto-generated by `data_copilot.schema_docs` — do not edit by hand."
    sections = [table_doc.to_markdown() for table_doc in table_docs]
    return "\n\n".join([header, *sections]) + "\n"


def write_schema_docs(settings: Settings | None = None, output_path: Path | None = None) -> Path:
    """Generate the schema docs and write them to Markdown.

    Defaults to ``<data_dir>/schema_docs.md``. Returns the path written to.
    """
    settings = settings or get_settings()
    table_docs = generate_schema_docs(settings=settings)
    settings.ensure_dirs()
    destination = output_path or (settings.data_dir / "schema_docs.md")
    destination.write_text(render_schema_markdown(table_docs), encoding="utf-8")
    return destination


if __name__ == "__main__":
    output = write_schema_docs()
    print(f"[data-copilot] schema docs written to: {output}")
