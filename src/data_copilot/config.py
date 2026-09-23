"""Central configuration for Data Copilot.

Reads overrides from environment variables so the same code runs the same
way locally, in CI, and inside the project's Docker Compose stack. Every
value has a sane default rooted at the repository layout described in the
README (data/, eval/, notebooks/, tests/).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    try:
        return int(value) if value else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the Data Copilot agent.

    Attributes:
        project_root: Repository root, used to resolve every other path.
        data_dir: Where the sample warehouse / raw data files live.
        vector_store_dir: Persistence directory for the Chroma vector store.
        eval_dir: Question/expected-SQL pairs and scoring artifacts.
        duckdb_path: DuckDB database file backing the sample warehouse.
        max_result_rows: Row cap enforced by the sandbox executor.
        llm_model: Default chat model name used by the generation chain.
    """

    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    data_dir: Path = field(default_factory=lambda: _env_path("DATA_COPILOT_DATA_DIR", PROJECT_ROOT / "data"))
    vector_store_dir: Path = field(
        default_factory=lambda: _env_path("DATA_COPILOT_VECTOR_DIR", PROJECT_ROOT / "data" / "chroma")
    )
    eval_dir: Path = field(default_factory=lambda: _env_path("DATA_COPILOT_EVAL_DIR", PROJECT_ROOT / "eval"))
    duckdb_path: Path = field(
        default_factory=lambda: _env_path("DATA_COPILOT_DUCKDB_PATH", PROJECT_ROOT / "data" / "warehouse.duckdb")
    )
    max_result_rows: int = field(default_factory=lambda: _env_int("DATA_COPILOT_MAX_ROWS", 500))
    llm_model: str = field(default_factory=lambda: os.environ.get("DATA_COPILOT_LLM_MODEL", "gpt-4o-mini"))

    def ensure_dirs(self) -> None:
        """Create the writable directories this settings object points at."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Return a fresh Settings instance built from the current environment."""
    return Settings()
