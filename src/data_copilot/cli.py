"""Minimal CLI entry point for Data Copilot.

Full `data-copilot ask "..."` behavior lands with the retrieval/generation
chain (see ROADMAP.md). For now this wires the console script, prints the
resolved settings, and can load the sample e-commerce warehouse into DuckDB
via `data-copilot load-sample-data`.
"""
from __future__ import annotations

import argparse
import sys

from data_copilot.config import get_settings
from data_copilot.data_loader import load_sample_warehouse, table_row_counts


def _run_load_sample_data() -> int:
    settings = get_settings()
    db_path = load_sample_warehouse(settings=settings)
    print(f"[data-copilot] sample warehouse loaded at: {db_path}")
    for table, count in table_row_counts(settings=settings).items():
        print(f"  {table:<12} {count} rows")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="data-copilot", description="Natural-language-to-SQL agent (WIP).")
    parser.add_argument("question", nargs="?", help="Question to ask the copilot (not yet implemented).")
    parser.add_argument(
        "--load-sample-data",
        action="store_true",
        help="Generate and load the sample e-commerce dataset into DuckDB, then exit.",
    )
    args = parser.parse_args(argv)

    if args.load_sample_data:
        return _run_load_sample_data()

    settings = get_settings()
    if args.question:
        print(f"[data-copilot] agent pipeline not implemented yet. Received question: {args.question!r}")
        print(f"[data-copilot] would use DuckDB at: {settings.duckdb_path}")
        return 0

    print("data-copilot: project skeleton is set up.")
    print(f"  project_root      = {settings.project_root}")
    print(f"  data_dir          = {settings.data_dir}")
    print(f"  vector_store_dir  = {settings.vector_store_dir}")
    print(f"  duckdb_path       = {settings.duckdb_path}")
    print(f"  max_result_rows   = {settings.max_result_rows}")
    print(f"  llm_model         = {settings.llm_model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
