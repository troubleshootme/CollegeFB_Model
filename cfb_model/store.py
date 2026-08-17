"""SQLite warehouse helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from cfb_model import config


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    config.ensure_dirs()
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_schema(conn: sqlite3.Connection | None = None) -> sqlite3.Connection:
    own = conn is None
    conn = conn or connect()
    schema = config.SCHEMA_PATH.read_text()
    conn.executescript(schema)
    conn.commit()
    if own:
        return conn
    return conn


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def read_table(conn: sqlite3.Connection, name: str) -> pd.DataFrame:
    if not table_exists(conn, name):
        return pd.DataFrame()
    return pd.read_sql(f'SELECT * FROM "{name}"', conn)


def replace_rows(
    conn: sqlite3.Connection,
    table: str,
    frame: pd.DataFrame,
    *,
    year_col: str | None = None,
    year: int | None = None,
) -> int:
    if frame is None or frame.empty:
        return 0
    prepared = frame.copy()
    prepared = prepared.where(pd.notnull(prepared), None)
    if year_col and year is not None:
        conn.execute(f'DELETE FROM "{table}" WHERE "{year_col}" = ?', (year,))
    cols = [c for c in prepared.columns]
    placeholders = ",".join(["?"] * len(cols))
    quoted = ",".join(f'"{c}"' for c in cols)
    sql = f'INSERT OR REPLACE INTO "{table}" ({quoted}) VALUES ({placeholders})'
    conn.executemany(sql, prepared.itertuples(index=False, name=None))
    conn.commit()
    return len(prepared)


def replace_all(conn: sqlite3.Connection, table: str, frame: pd.DataFrame) -> int:
    if frame is None or frame.empty:
        return 0
    conn.execute(f'DELETE FROM "{table}"')
    return replace_rows(conn, table, frame)


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO ingest_meta (key, value) VALUES (?, ?)", (key, value)
    )
    conn.commit()
