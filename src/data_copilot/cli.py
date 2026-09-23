"""Minimal CLI entry point placeholder.

Full `data-copilot ask "..."` behavior lands with the retrieval/generation
chain (see ROADMAP.md); for now this wires the console script and prints
the resolved settings so the skeleton is verifiable end-to-end.
"""
from __future__ import annotations

import argparse
import sys

from data_copilot.config import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="data-copilot", description="Natural-language-to-SQL agent (WIP).")
    parser.add_argument("question", nargs="?", help="Question to ask the copilot (not yet implemented).")
    args = parser.parse_args(argv)

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
