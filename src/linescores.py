"""Per-quarter scores (CFBD ``/games`` line scores), stored beside ``games`` in ``game_line_scores``.

The main collector keeps only final points, so quarter data is fetched and stored
separately: it is additive, never rewrites ``games``, and refreshes cheaply
(one call per season and season type).
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pandas as pd

from src.cfbd_client import get_json
from src.config import DB_PATH, SEASON_TYPES

TABLE = "game_line_scores"
QUARTER_COLS = [f"{side}_q{q}" for side in ("home", "away") for q in (1, 2, 3, 4)]
COLUMNS = ["game_id", "season", "completed", *QUARTER_COLS, "home_ot", "away_ot", "went_to_ot"]


def parse_line_scores(value: Any) -> tuple[list[float], float]:
    """CFBD gives ``[q1, q2, q3, q4, ot1, ot2, ...]``. Returns (4 quarters, total overtime points)."""
    if not isinstance(value, (list, tuple)):
        return [float("nan")] * 4, float("nan")
    nums = []
    for item in value:
        try:
            nums.append(float(item))
        except (TypeError, ValueError):
            nums.append(float("nan"))
    quarters = (nums + [float("nan")] * 4)[:4]
    overtime = sum(x for x in nums[4:] if x == x)  # NaN-safe
    return quarters, overtime


def rows_from_games(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for game in payload or []:
        if not game.get("completed"):
            continue
        home_q, home_ot = parse_line_scores(game.get("homeLineScores"))
        away_q, away_ot = parse_line_scores(game.get("awayLineScores"))
        if any(x != x for x in home_q + away_q):
            continue  # incomplete line score: unusable for quarter modelling
        row: dict[str, Any] = {"game_id": game.get("id"), "season": game.get("season"), "completed": 1}
        for i, value in enumerate(home_q, start=1):
            row[f"home_q{i}"] = value
        for i, value in enumerate(away_q, start=1):
            row[f"away_q{i}"] = value
        row["home_ot"] = home_ot
        row["away_ot"] = away_ot
        row["went_to_ot"] = int(len(game.get("homeLineScores") or []) > 4 or len(game.get("awayLineScores") or []) > 4)
        rows.append(row)
    return rows


def fetch_line_scores(year: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for season_type in SEASON_TYPES:
        payload = get_json("/games", {"year": year, "seasonType": season_type, "classification": "fbs"})
        rows.extend(rows_from_games(payload if isinstance(payload, list) else []))
    return pd.DataFrame(rows, columns=COLUMNS)


def stored_years(con: sqlite3.Connection) -> set[int]:
    exists = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE,)).fetchone()
    if not exists:
        return set()
    return {int(y) for (y,) in con.execute(f'SELECT DISTINCT season FROM "{TABLE}"') if y is not None}


def sync_line_scores(years: list[int], db_path=DB_PATH, fetch=fetch_line_scores) -> int:
    """Replace the stored line scores of ``years``. Returns rows written."""
    written = 0
    with sqlite3.connect(db_path) as con:
        for year in years:
            frame = fetch(year)
            if frame.empty:
                print(f"  line scores {year}: none returned")
                continue
            exists = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE,)).fetchone()
            if exists:
                con.execute(f'DELETE FROM "{TABLE}" WHERE season = ?', (int(year),))
            frame.to_sql(TABLE, con, if_exists="append", index=False)
            written += len(frame)
            print(f"  line scores {year}: {len(frame):,} games")
        con.execute(f'CREATE INDEX IF NOT EXISTS ix_line_scores_game ON "{TABLE}"(game_id)')
        con.commit()
    return written


def load_line_scores(con: sqlite3.Connection) -> pd.DataFrame:
    if not stored_years(con):
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_sql_query(f'SELECT * FROM "{TABLE}"', con)
