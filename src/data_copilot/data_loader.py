"""Sample e-commerce dataset generation and loading into DuckDB.

Generates a small, deterministic synthetic e-commerce warehouse (customers,
products, orders, order_items) and loads it into the DuckDB database pointed
at by ``Settings.duckdb_path``. This gives the retrieval/generation chain
(see ROADMAP.md) a concrete sample warehouse to document and query against.

The dataset is intentionally tiny and seeded for reproducibility — it exists
to exercise the schema-doc generator, the retriever, and the text-to-SQL
chain end-to-end, not to be a realistic data volume.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb

from data_copilot.config import Settings, get_settings

DEFAULT_SEED = 42

_FIRST_NAMES = [
    "Alice", "Bruno", "Chidi", "Diane", "Emeka", "Fatima", "Gabriel", "Hana",
    "Ivan", "Julia", "Kwame", "Lina", "Mateo", "Nadia", "Omar", "Priya",
]
_LAST_NAMES = [
    "Nguyen", "Smith", "Okafor", "Garcia", "Dubois", "Khan", "Rossi", "Tremblay",
    "Silva", "Kim", "Haddad", "Novak", "Mensah", "Park", "Costa", "Ivanov",
]
_COUNTRIES = ["CA", "US", "FR", "NG", "BR", "IN"]

_PRODUCT_CATALOG = [
    ("Wireless Mouse", "Electronics", 24.99),
    ("Mechanical Keyboard", "Electronics", 89.99),
    ("USB-C Hub", "Electronics", 34.50),
    ("Noise-Cancelling Headphones", "Electronics", 149.00),
    ("Standing Desk Mat", "Office", 45.00),
    ("Ergonomic Chair", "Office", 310.00),
    ("Notebook Set", "Office", 12.75),
    ("Desk Lamp", "Office", 29.99),
    ("Running Shoes", "Sporting Goods", 79.95),
    ("Yoga Mat", "Sporting Goods", 22.00),
    ("Water Bottle", "Sporting Goods", 15.50),
    ("Coffee Grinder", "Home", 54.00),
    ("French Press", "Home", 32.25),
    ("Throw Blanket", "Home", 38.40),
    ("Scented Candle", "Home", 18.00),
    ("Board Game", "Toys", 27.00),
    ("Building Blocks Set", "Toys", 41.99),
    ("Puzzle 1000pc", "Toys", 16.99),
    ("Backpack", "Accessories", 59.00),
    ("Sunglasses", "Accessories", 48.50),
]

_ORDER_STATUSES = ["completed", "completed", "completed", "shipped", "pending", "cancelled"]


@dataclass(frozen=True)
class SeedCounts:
    """How many synthetic rows to generate for each table."""

    customers: int = 40
    products: int = len(_PRODUCT_CATALOG)
    orders: int = 150
    max_items_per_order: int = 4


def _generate_customers(rng: random.Random, n: int) -> list[tuple[Any, ...]]:
    rows = []
    start = date(2023, 1, 1)
    for customer_id in range(1, n + 1):
        first = rng.choice(_FIRST_NAMES)
        last = rng.choice(_LAST_NAMES)
        signup_offset = rng.randint(0, 900)
        rows.append(
            (
                customer_id,
                f"{first} {last}",
                f"{first.lower()}.{last.lower()}{customer_id}@example.com",
                rng.choice(_COUNTRIES),
                start + timedelta(days=signup_offset),
            )
        )
    return rows


def _generate_products() -> list[tuple[Any, ...]]:
    return [
        (product_id, name, category, price)
        for product_id, (name, category, price) in enumerate(_PRODUCT_CATALOG, start=1)
    ]


def _generate_orders_and_items(
    rng: random.Random, counts: SeedCounts
) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    orders: list[tuple[Any, ...]] = []
    items: list[tuple[Any, ...]] = []
    start = date(2024, 1, 1)
    item_id = 1

    for order_id in range(1, counts.orders + 1):
        customer_id = rng.randint(1, counts.customers)
        order_date = start + timedelta(days=rng.randint(0, 600))
        status = rng.choice(_ORDER_STATUSES)

        n_items = rng.randint(1, counts.max_items_per_order)
        chosen_products = rng.sample(range(1, counts.products + 1), k=min(n_items, counts.products))
        order_total = 0.0
        for product_id in chosen_products:
            unit_price = _PRODUCT_CATALOG[product_id - 1][2]
            quantity = rng.randint(1, 3)
            items.append((item_id, order_id, product_id, quantity, unit_price))
            order_total += unit_price * quantity
            item_id += 1

        orders.append((order_id, customer_id, order_date, status, round(order_total, 2)))

    return orders, items


def generate_dataset(seed: int = DEFAULT_SEED, counts: SeedCounts | None = None) -> dict[str, list[tuple[Any, ...]]]:
    """Generate the synthetic e-commerce rows for every table.

    Deterministic given the same ``seed`` and ``counts``, so repeated loads
    produce an identical warehouse.
    """
    rng = random.Random(seed)
    counts = counts or SeedCounts()

    customers = _generate_customers(rng, counts.customers)
    products = _generate_products()
    orders, order_items = _generate_orders_and_items(rng, counts)

    return {
        "customers": customers,
        "products": products,
        "orders": orders,
        "order_items": order_items,
    }


_SCHEMA_DDL = {
    "customers": """
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            full_name VARCHAR NOT NULL,
            email VARCHAR NOT NULL,
            country VARCHAR NOT NULL,
            signup_date DATE NOT NULL
        )
    """,
    "products": """
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            product_name VARCHAR NOT NULL,
            category VARCHAR NOT NULL,
            unit_price DOUBLE NOT NULL
        )
    """,
    "orders": """
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
            order_date DATE NOT NULL,
            status VARCHAR NOT NULL,
            order_total DOUBLE NOT NULL
        )
    """,
    "order_items": """
        CREATE TABLE order_items (
            order_item_id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL REFERENCES orders(order_id),
            product_id INTEGER NOT NULL REFERENCES products(product_id),
            quantity INTEGER NOT NULL,
            unit_price DOUBLE NOT NULL
        )
    """,
}

# Drop order matters too: children must be dropped before the parents they
# reference (reverse of _TABLE_ORDER, defined below).
_DROP_ORDER = ["order_items", "orders", "products", "customers"]

_INSERT_SQL = {
    "customers": "INSERT INTO customers VALUES (?, ?, ?, ?, ?)",
    "products": "INSERT INTO products VALUES (?, ?, ?, ?)",
    "orders": "INSERT INTO orders VALUES (?, ?, ?, ?, ?)",
    "order_items": "INSERT INTO order_items VALUES (?, ?, ?, ?, ?)",
}

# Table load order matters: children reference parents via foreign keys.
_TABLE_ORDER = ["customers", "products", "orders", "order_items"]


def load_sample_warehouse(
    settings: Settings | None = None,
    seed: int = DEFAULT_SEED,
    counts: SeedCounts | None = None,
) -> Path:
    """Create (or refresh) the sample e-commerce warehouse in DuckDB.

    Returns the path to the DuckDB database file. Safe to call repeatedly:
    each table is created with ``CREATE OR REPLACE`` and fully repopulated.
    """
    settings = settings or get_settings()
    settings.ensure_dirs()

    dataset = generate_dataset(seed=seed, counts=counts)

    connection = duckdb.connect(str(settings.duckdb_path))
    try:
        for table in _DROP_ORDER:
            connection.execute(f"DROP TABLE IF EXISTS {table}")
        for table in _TABLE_ORDER:
            connection.execute(_SCHEMA_DDL[table])
            rows = dataset[table]
            if rows:
                connection.executemany(_INSERT_SQL[table], rows)
    finally:
        connection.close()

    return settings.duckdb_path


def table_row_counts(settings: Settings | None = None) -> dict[str, int]:
    """Return the row count of each sample-warehouse table (for smoke checks)."""
    settings = settings or get_settings()
    connection = duckdb.connect(str(settings.duckdb_path), read_only=True)
    try:
        return {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in _TABLE_ORDER
        }
    finally:
        connection.close()


if __name__ == "__main__":
    path = load_sample_warehouse()
    print(f"[data-copilot] sample warehouse loaded at: {path}")
    for table, count in table_row_counts().items():
        print(f"  {table:<12} {count} rows")
