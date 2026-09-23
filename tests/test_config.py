"""Unit tests for the project skeleton: package metadata and settings."""
from pathlib import Path

import data_copilot
from data_copilot.config import Settings, get_settings


def test_package_version_is_defined():
    assert data_copilot.__version__ == "0.1.0"


def test_settings_default_paths_are_rooted_at_project(tmp_path, monkeypatch):
    for var in (
        "DATA_COPILOT_DATA_DIR",
        "DATA_COPILOT_VECTOR_DIR",
        "DATA_COPILOT_EVAL_DIR",
        "DATA_COPILOT_DUCKDB_PATH",
        "DATA_COPILOT_MAX_ROWS",
        "DATA_COPILOT_LLM_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)

    settings = get_settings()

    assert isinstance(settings, Settings)
    assert settings.data_dir == settings.project_root / "data"
    assert settings.vector_store_dir == settings.project_root / "data" / "chroma"
    assert settings.eval_dir == settings.project_root / "eval"
    assert settings.duckdb_path == settings.project_root / "data" / "warehouse.duckdb"
    assert settings.max_result_rows == 500
    assert settings.llm_model == "gpt-4o-mini"


def test_settings_respect_environment_overrides(tmp_path, monkeypatch):
    custom_data_dir = tmp_path / "custom_data"
    monkeypatch.setenv("DATA_COPILOT_DATA_DIR", str(custom_data_dir))
    monkeypatch.setenv("DATA_COPILOT_MAX_ROWS", "42")
    monkeypatch.setenv("DATA_COPILOT_LLM_MODEL", "gpt-4o")

    settings = get_settings()

    assert settings.data_dir == custom_data_dir
    assert settings.max_result_rows == 42
    assert settings.llm_model == "gpt-4o"


def test_ensure_dirs_creates_directories(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_COPILOT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DATA_COPILOT_VECTOR_DIR", str(tmp_path / "data" / "chroma"))

    settings = get_settings()
    settings.ensure_dirs()

    assert settings.data_dir.is_dir()
    assert settings.vector_store_dir.is_dir()


def test_cli_main_prints_settings_without_question(capsys):
    from data_copilot.cli import main

    exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "project skeleton is set up" in captured.out
