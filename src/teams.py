"""FBS team directory with CFBD logos and colors."""

from __future__ import annotations

import sqlite3
from typing import Any

import pandas as pd

from src.collect import branding_fields, fetch_fbs_teams
from src.config import DB_PATH, END_YEAR

BRAND_COLS = (
    ("abbreviation", "TEXT"),
    ("color", "TEXT"),
    ("alt_color", "TEXT"),
    ("logo", "TEXT"),
)


def _ensure_branding_columns(con: sqlite3.Connection, table: str = "fbs_teams") -> None:
    existing = {row[1] for row in con.execute(f'PRAGMA table_info("{table}")')}
    if not existing:
        return
    for name, typ in BRAND_COLS:
        if name not in existing:
            con.execute(f'ALTER TABLE "{table}" ADD COLUMN {name} {typ}')


def _hex(value: object) -> str | None:
    return branding_fields({"color": value})["color"]


def refresh_team_branding(year: int | None = None) -> int:
    year = year or END_YEAR
    frame = fetch_fbs_teams(year)
    if frame.empty:
        return 0
    updated = 0
    with sqlite3.connect(DB_PATH) as con:
        _ensure_branding_columns(con, "fbs_teams")
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "fbsTeams" in tables:
            _ensure_branding_columns(con, "fbsTeams")
        for _, row in frame.iterrows():
            params = (
                row.get("abbreviation"),
                row.get("color"),
                row.get("alt_color"),
                row.get("logo"),
                int(row["year"]),
                int(row["team_id"]),
            )
            cur = con.execute(
                "UPDATE fbs_teams SET abbreviation=?, color=?, alt_color=?, logo=? WHERE year=? AND team_id=?",
                params,
            )
            updated += cur.rowcount
            if "fbsTeams" in tables:
                con.execute(
                    "UPDATE fbsTeams SET abbreviation=?, color=?, alt_color=?, logo=? WHERE id=?",
                    (row.get("abbreviation"), row.get("color"), row.get("alt_color"), row.get("logo"), int(row["team_id"])),
                )
        con.commit()
    return updated


def _row_to_team(row: pd.Series) -> dict[str, Any] | None:
    school = row.get("school") or row.get("name")
    if not school or pd.isna(school):
        return None
    team_id = row.get("team_id") if "team_id" in row.index else row.get("id")
    try:
        team_id_out = int(team_id) if team_id is not None and not pd.isna(team_id) else None
    except (TypeError, ValueError):
        team_id_out = None
    logo = row.get("logo")
    if logo is not None and pd.isna(logo):
        logo = None
    abbr = row.get("abbreviation")
    if abbr is not None and pd.isna(abbr):
        abbr = None
    conf = row.get("conference")
    if conf is not None and pd.isna(conf):
        conf = None
    return {
        "id": team_id_out,
        "name": str(school),
        "school": str(school),
        "abbreviation": None if abbr is None else str(abbr),
        "conference": None if conf is None else str(conf),
        "logo_url": None if logo is None else str(logo),
        "primary_color": _hex(row.get("color")),
        "secondary_color": _hex(row.get("alt_color")),
    }


def load_team_directory(refresh_if_empty: bool = False) -> dict[str, dict[str, Any]]:
    if not DB_PATH.exists():
        return {}
    with sqlite3.connect(DB_PATH) as con:
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "fbs_teams" not in tables and "fbsTeams" not in tables:
            return {}
        if "fbs_teams" in tables:
            _ensure_branding_columns(con, "fbs_teams")
            year_row = con.execute("SELECT MAX(year) FROM fbs_teams").fetchone()
            year = year_row[0] if year_row else None
            if year is None:
                frame = pd.read_sql("SELECT * FROM fbs_teams", con)
            else:
                frame = pd.read_sql("SELECT * FROM fbs_teams WHERE year = ?", con, params=(int(year),))
        else:
            _ensure_branding_columns(con, "fbsTeams")
            frame = pd.read_sql("SELECT * FROM fbsTeams", con)
    if refresh_if_empty and not frame.empty and ("logo" not in frame.columns or frame["logo"].isna().all()):
        try:
            refresh_team_branding()
            return load_team_directory(refresh_if_empty=False)
        except Exception:
            pass
    directory: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        team = _row_to_team(row)
        if team:
            directory[team["name"]] = team
    return directory
