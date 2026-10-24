"""Text-to-SQL prompt template with few-shot examples.

Turns a natural-language question plus retrieved schema context (see
``data_copilot.retriever``) into a ready-to-send prompt for the SQL
generation step. Deliberately framework-free, like ``retriever``: the
LangGraph generate -> execute -> self-correct loop (a later roadmap step)
calls ``build_messages``/``build_prompt`` to get its initial prompt, then
owns the actual LLM call and the self-correction turns on top of it.

The few-shot examples are fixed and hand-written against the sample
e-commerce warehouse (``customers``, ``products``, ``orders``,
``order_items`` -- see ``data_copilot.data_loader``) so they stay accurate
without needing a live LLM or database at prompt-build time.
"""
from __future__ import annotations

from dataclasses import dataclass

from data_copilot.config import Settings, get_settings

SYSTEM_INSTRUCTIONS = """You are Data Copilot, a SQL analyst assistant for a DuckDB warehouse.

Rules:
- Write DuckDB-dialect SQL only.
- Produce exactly one statement, and it must be a read-only SELECT (or a
  SELECT wrapped in a CTE). Never write INSERT, UPDATE, DELETE, DROP,
  ALTER, or any other statement that mutates data or schema.
- Only reference tables and columns that appear in the "Schema context"
  section below. If the question cannot be answered from that context,
  say so instead of guessing at table or column names.
- Always include an explicit `LIMIT {max_result_rows}` (or fewer rows) on
  the final SELECT, unless the query already aggregates down to a single
  row.
- Return only the SQL, inside a single ```sql code block -- no commentary
  before or after it.""".strip()


@dataclass(frozen=True)
class FewShotExample:
    """One worked natural-language-question -> SQL pair."""

    question: str
    sql: str


# Hand-written against the sample warehouse schema (customers, products,
# orders, order_items). Chosen to cover the generation patterns the agent
# will need most often: a plain filter, a join, an aggregation, and a
# multi-table join with grouping.
DEFAULT_FEW_SHOT_EXAMPLES: tuple[FewShotExample, ...] = (
    FewShotExample(
        question="List the email addresses of customers in Canada.",
        sql="SELECT full_name, email FROM customers WHERE country = 'Canada' LIMIT 500;",
    ),
    FewShotExample(
        question="What is the total order amount for order 42?",
        sql="SELECT order_id, order_total FROM orders WHERE order_id = 42;",
    ),
    FewShotExample(
        question="How many orders has each customer placed?",
        sql=(
            "SELECT c.customer_id, c.full_name, COUNT(o.order_id) AS order_count\n"
            "FROM customers AS c\n"
            "LEFT JOIN orders AS o ON o.customer_id = c.customer_id\n"
            "GROUP BY c.customer_id, c.full_name\n"
            "ORDER BY order_count DESC\n"
            "LIMIT 500;"
        ),
    ),
    FewShotExample(
        question="What are the top 5 products by total revenue?",
        sql=(
            "SELECT p.product_name, SUM(oi.quantity * oi.unit_price) AS revenue\n"
            "FROM order_items AS oi\n"
            "JOIN products AS p ON p.product_id = oi.product_id\n"
            "GROUP BY p.product_name\n"
            "ORDER BY revenue DESC\n"
            "LIMIT 5;"
        ),
    ),
)


def format_examples(examples: tuple[FewShotExample, ...] = DEFAULT_FEW_SHOT_EXAMPLES) -> str:
    """Render few-shot examples into a Markdown block for prompt injection."""
    if not examples:
        return ""
    blocks = [
        f"Q: {example.question}\nSQL:\n```sql\n{example.sql}\n```" for example in examples
    ]
    return "\n\n".join(blocks)


def build_messages(
    question: str,
    schema_context: str,
    examples: tuple[FewShotExample, ...] = DEFAULT_FEW_SHOT_EXAMPLES,
    settings: Settings | None = None,
) -> list[dict[str, str]]:
    """Build a chat-style message list ready for a chat-completion LLM call.

    This is the primary entry point: ``retriever.retrieve_context`` supplies
    ``schema_context`` and this renders the full prompt around it. Returns
    ``[{"role": "system", ...}, {"role": "user", ...}]``, the shape LangChain
    chat models and the OpenAI-style chat-completion APIs both expect.
    """
    settings = settings or get_settings()
    system_content = SYSTEM_INSTRUCTIONS.format(max_result_rows=settings.max_result_rows)

    examples_block = format_examples(examples)
    examples_section = f"Examples:\n\n{examples_block}\n\n" if examples_block else ""

    user_content = (
        f"Schema context:\n{schema_context}\n\n"
        f"{examples_section}"
        f"Question: {question}"
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def build_prompt(
    question: str,
    schema_context: str,
    examples: tuple[FewShotExample, ...] = DEFAULT_FEW_SHOT_EXAMPLES,
    settings: Settings | None = None,
) -> str:
    """Flatten ``build_messages`` into a single string prompt.

    Useful for completion-style LLM calls, logging, or printing from the CLI
    (``--build-prompt``) without needing a chat-message-aware caller.
    """
    messages = build_messages(question, schema_context, examples=examples, settings=settings)
    return "\n\n".join(f"[{message['role'].upper()}]\n{message['content']}" for message in messages)
