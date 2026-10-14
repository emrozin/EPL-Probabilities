"""Shared database helpers, used by every sport's importers.

One SQLite database (data/sports.db) holds every sport; schema.sql defines it.
Paths are relative to the project folder, so run scripts from there (python -m epl.import_epl).
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/sports.db")
RAW_DIR = Path("data/raw")       # downloaded source files, cached
SCHEMA_PATH = Path("schema.sql")


def as_int(value):
    return None if value is None or pd.isna(value) else int(value)


def get_team_id(conn: sqlite3.Connection, league_id: int, name: str, cache: dict) -> int:
    """Return the team's id, inserting it first if needed. `cache` avoids repeat lookups."""
    name = name.strip()
    if name not in cache:
        conn.execute(
            "INSERT OR IGNORE INTO teams (league_id, name) VALUES (?, ?)",
            (league_id, name),
        )
        cache[name] = conn.execute(
            "SELECT id FROM teams WHERE league_id = ? AND name = ?",
            (league_id, name),
        ).fetchone()[0]
    return cache[name]


def insert_price(conn, game_id, bookmaker, market, outcome, price, line, is_closing) -> int:
    """Insert one price. Returns 1 if a new row was added, 0 otherwise."""
    if price is None or line is None and market not in ("1x2", "moneyline"):   # only these have no line
        return 0
    try:
        price = float(price)
    except (TypeError, ValueError):
        return 0  # the occasional junk value in old files
    if price <= 1.0:
        return 0  # not a valid decimal price
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO odds (game_id, bookmaker, market, outcome, price, line, is_closing)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (game_id, bookmaker, market, outcome, price, line, is_closing),
    )
    return cursor.rowcount
