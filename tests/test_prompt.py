"""Unit tests for the text-to-SQL prompt template."""
from __future__ import annotations

from data_copilot.config import Settings
from data_copilot.prompt import (
    DEFAULT_FEW_SHOT_EXAMPLES,
    FewShotExample,
    build_messages,
    build_prompt,
    format_examples,
)


def test_format_examples_includes_every_example():
    rendered = format_examples(DEFAULT_FEW_SHOT_EXAMPLES)

    for example in DEFAULT_FEW_SHOT_EXAMPLES:
        assert example.question in rendered
        assert example.sql in rendered


def test_format_examples_empty_tuple_returns_empty_string():
    assert format_examples(()) == ""


def test_build_messages_returns_system_and_user_roles():
    messages = build_messages("How many customers signed up last month?", "## Table: customers\n...")

    assert [message["role"] for message in messages] == ["system", "user"]


def test_build_messages_system_prompt_states_read_only_rule():
    messages = build_messages("anything", "schema context")
    system_content = messages[0]["content"]

    assert "SELECT" in system_content
    assert "read-only" in system_content.lower()


def test_build_messages_embeds_max_result_rows_from_settings(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        vector_store_dir=tmp_path / "data" / "chroma",
        eval_dir=tmp_path / "eval",
        duckdb_path=tmp_path / "data" / "warehouse.duckdb",
        max_result_rows=42,
    )
    messages = build_messages("anything", "schema context", settings=settings)

    assert "LIMIT 42" in messages[0]["content"]


def test_build_messages_user_content_includes_schema_context_and_question():
    messages = build_messages(
        "What is the total order amount for order 7?",
        "## Table: orders\n...",
    )
    user_content = messages[1]["content"]

    assert "## Table: orders" in user_content
    assert "What is the total order amount for order 7?" in user_content


def test_build_messages_user_content_includes_examples_by_default():
    messages = build_messages("anything", "schema context")
    user_content = messages[1]["content"]

    assert DEFAULT_FEW_SHOT_EXAMPLES[0].question in user_content


def test_build_messages_without_examples_omits_examples_section():
    messages = build_messages("anything", "schema context", examples=())
    user_content = messages[1]["content"]

    assert "Examples:" not in user_content
    assert DEFAULT_FEW_SHOT_EXAMPLES[0].question not in user_content


def test_build_messages_custom_examples_override_default():
    custom = (FewShotExample(question="How many products are there?", sql="SELECT COUNT(*) FROM products;"),)
    messages = build_messages("anything", "schema context", examples=custom)
    user_content = messages[1]["content"]

    assert "How many products are there?" in user_content
    assert DEFAULT_FEW_SHOT_EXAMPLES[0].question not in user_content


def test_build_prompt_flattens_messages_into_single_string():
    prompt = build_prompt("anything", "schema context")

    assert "[SYSTEM]" in prompt
    assert "[USER]" in prompt
    assert "schema context" in prompt
