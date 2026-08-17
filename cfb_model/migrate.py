"""Migrate the legacy year-sharded collegeFootball.db into the normalized warehouse."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from cfb_model import config, store
from cfb_model.ingest.flatten import flatten_legacy_box
from cfb_model.util import as_bool_int, as_float, as_int, iso_date


LEGACY_GAME_YEARS = range(2020, 2025)
STAT_TABLES = {
    2020: "advancedSeasonGameStat20",
    2021: "advancedSeasonGameStat21",
    2022: "advancedSeasonGameStat22",
    2023: "advancedSeasonGameStat23",
    2024: "advancedSeasonGameStat24",
}
BET_TABLES = {
    2020: "bets20",
    2021: "bets21",
    2022: "bets22",
    2023: "bets23",
    2024: "bets24",
}


def run_migrate(legacy_path: Path | None = None) -> dict[str, int]:
    path = Path(legacy_path or config.LEGACY_DB_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Legacy database not found: {path}")
    legacy = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn = store.init_schema()
    counts: dict[str, int] = {}
    try:
        counts["teams"] = _migrate_teams(legacy, conn)
        counts["venues"] = counts["teams"]
        counts["talent"] = _migrate_talent(legacy, conn)
        counts["coaches_seasons"] = _migrate_coaches(legacy, conn)
        games, lines = _migrate_games_and_lines(legacy)
        counts["games"] = store.replace_all(conn, "games", games) if not games.empty else 0
        counts["lines"] = store.replace_all(conn, "lines", lines) if not lines.empty else 0
        stats = _migrate_box_stats(legacy)
        counts["team_game_stats"] = store.replace_all(conn, "team_game_stats", stats) if not stats.empty else 0
        store.set_meta(conn, "migrated_from", str(path))
    finally:
        legacy.close()
        conn.close()
    return counts


def _migrate_teams(legacy, conn) -> int:
    if not store.table_exists(legacy, "fbsTeams"):
        return 0
    raw = pd.read_sql("SELECT * FROM fbsTeams", legacy)
    teams = pd.DataFrame(
        {
            "id": raw["id"].map(as_int),
            "school": raw["school"],
            "conference": raw["conference"],
            "classification": "fbs",
            "venue_id": raw["id"].map(as_int),
            "latitude": raw["latitude"].map(as_float),
            "longitude": raw["longitude"].map(as_float),
            "elevation": raw["elevation"].map(as_float),
            "capacity": raw["capacity"].map(as_float),
            "grass": raw["grass"].map(as_bool_int),
            "dome": raw["dome"].map(as_bool_int),
        }
    ).dropna(subset=["id", "school"])
    venues = pd.DataFrame(
        {
            "id": teams["venue_id"],
            "name": raw["school"] + " Stadium",
            "city": None,
            "state": None,
            "timezone": None,
            "latitude": teams["latitude"],
            "longitude": teams["longitude"],
            "elevation": teams["elevation"],
            "capacity": teams["capacity"],
            "grass": teams["grass"],
            "dome": teams["dome"],
        }
    )
    store.replace_all(conn, "teams", teams)
    store.replace_all(conn, "venues", venues)
    return len(teams)


def _migrate_talent(legacy, conn) -> int:
    if not store.table_exists(legacy, "talent20s"):
        return 0
    raw = pd.read_sql("SELECT * FROM talent20s", legacy)
    frame = pd.DataFrame(
        {
            "year": raw["year"].map(as_int),
            "team": raw["team"],
            "talent": raw["talent"].map(as_float),
        }
    ).dropna(subset=["year", "team"])
    return store.replace_all(conn, "talent", frame)


def _migrate_coaches(legacy, conn) -> int:
    if not store.table_exists(legacy, "coachHist"):
        return 0
    raw = pd.read_sql("SELECT * FROM coachHist", legacy)
    frame = pd.DataFrame(
        {
            "coach_id": None,
            "first_name": raw.get("first_name"),
            "last_name": raw.get("season_type"),  # legacy notebook stored last name here
            "school": raw.get("school"),
            "year": raw["year"].map(as_int) if "year" in raw else None,
            "games": raw["games"].map(as_int) if "games" in raw else None,
            "wins": None,
            "losses": raw["losses"].map(as_int) if "losses" in raw else None,
            "sp_overall": raw["sp_overall"].map(as_float) if "sp_overall" in raw else None,
            "sp_offense": raw["sp_offense"].map(as_float) if "sp_offense" in raw else None,
            "sp_defense": raw["sp_defense"].map(as_float) if "sp_defense" in raw else None,
            "srs": None,
        }
    ).dropna(subset=["school", "year"])
    if "sp_overall" in frame:
        sp = frame.dropna(subset=["sp_overall"])[["year", "school", "sp_overall", "sp_offense", "sp_defense"]].copy()
        sp = sp.rename(columns={"school": "team", "sp_overall": "rating", "sp_offense": "offense", "sp_defense": "defense"})
        sp["ranking"] = None
        sp["special_teams"] = None
        sp = sp.drop_duplicates(["year", "team"])
        store.replace_all(conn, "sp_ratings", sp)
    return store.replace_all(conn, "coaches_seasons", frame)


def _migrate_games_and_lines(legacy) -> tuple[pd.DataFrame, pd.DataFrame]:
    game_frames = []
    line_frames = []
    for year in LEGACY_GAME_YEARS:
        table = f"Games{year}"
        if not store.table_exists(legacy, table):
            continue
        raw = pd.read_sql(f'SELECT * FROM "{table}"', legacy)
        bets_table = BET_TABLES.get(year)
        bets = (
            pd.read_sql(f'SELECT * FROM "{bets_table}"', legacy)
            if bets_table and store.table_exists(legacy, bets_table)
            else pd.DataFrame()
        )
        id_map, lines = _reconstruct_lines(bets, year)
        games = _legacy_games(raw, year, id_map)
        game_frames.append(games)
        if not lines.empty:
            line_frames.append(lines)
    games = pd.concat(game_frames, ignore_index=True) if game_frames else pd.DataFrame()
    lines = pd.concat(line_frames, ignore_index=True) if line_frames else pd.DataFrame()
    if not games.empty:
        games = games.drop_duplicates(subset=["id"], keep="first")
    if not lines.empty:
        lines = lines.dropna(subset=["game_id", "provider"]).drop_duplicates(["game_id", "provider"])
    return games, lines


def _legacy_games(raw: pd.DataFrame, year: int, id_map: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "season": raw.get("season", year),
            "week": raw.get("week"),
            "season_type": raw.get("season_type"),
            "start_date": raw.get("start_date").map(iso_date) if "start_date" in raw else None,
            "completed": raw["completed"].map(as_bool_int) if "completed" in raw else 0,
            "neutral_site": 0,
            "conference_game": raw["conference_game"].map(as_bool_int) if "conference_game" in raw else 0,
            "venue_id": raw.get("venue_id"),
            "venue": raw.get("venue"),
            "home_id": raw.get("home_id"),
            "home_team": raw.get("home_team"),
            "home_conference": raw.get("home_conference"),
            "home_classification": raw.get("home_division"),
            "home_points": raw.get("home_points"),
            "home_pregame_elo": raw.get("home_pregame_elo"),
            "home_postgame_elo": raw.get("home_postgame_elo"),
            "home_postgame_wp": raw.get("home_post_win_prob"),
            "away_id": raw.get("away_id"),
            "away_team": raw.get("away_team"),
            "away_conference": raw.get("away_conference"),
            "away_classification": raw.get("away_division"),
            "away_points": raw.get("away_points"),
            "away_pregame_elo": raw.get("away_pregame_elo"),
            "away_postgame_elo": raw.get("away_postgame_elo"),
            "away_postgame_wp": raw.get("away_post_win_prob"),
            "excitement_index": raw.get("excitement_index"),
        }
    )
    frame["season"] = frame["season"].map(as_int).fillna(year).astype(int)
    frame["week"] = frame["week"].map(as_int)
    if not id_map.empty:
        frame = frame.merge(id_map, on=["season", "week", "home_team", "away_team"], how="left")
    else:
        frame["id"] = pd.NA
    missing = frame["id"].isna()
    if missing.any():
        keys = (
            frame.loc[missing, "season"].astype(str)
            + "|"
            + frame.loc[missing, "week"].astype(str)
            + "|"
            + frame.loc[missing, "home_team"].astype(str)
            + "|"
            + frame.loc[missing, "away_team"].astype(str)
        )
        frame.loc[missing, "id"] = keys.map(lambda k: int.from_bytes(k.encode(), "little") % 10**12)
    frame["id"] = frame["id"].map(as_int)
    # Prefer FBS vs FBS; keep FCS opponents of FBS homes for completeness.
    return frame.dropna(subset=["id", "home_team", "away_team"])


def _reconstruct_lines(bets: pd.DataFrame, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bets.empty:
        return pd.DataFrame(columns=["season", "week", "home_team", "away_team", "id"]), pd.DataFrame()
    current_id = None
    current_meta: dict | None = None
    id_rows: list[dict] = []
    line_rows: list[dict] = []
    for record in bets.to_dict(orient="records"):
        game_id = as_int(record.get("game_id"))
        if game_id is not None:
            current_id = game_id
            current_meta = {
                "id": game_id,
                "season": as_int(record.get("season") or year),
                "week": as_int(record.get("week")),
                "home_team": record.get("home_team"),
                "away_team": record.get("away_team"),
            }
            id_rows.append(current_meta)
            continue
        if current_id is None:
            continue
        providers = [c for c in bets.columns if str(c).startswith("provider")]
        for col in providers:
            idx = str(col).replace("provider", "")
            provider = record.get(col)
            spread = record.get(f"spread{idx}")
            if provider is None and spread is None:
                continue
            line_rows.append(
                {
                    "game_id": current_id,
                    "provider": provider or f"book{idx}",
                    "spread": as_float(spread),
                    "spread_open": None,
                    "over_under": as_float(record.get(f"over_under{idx}")),
                    "over_under_open": None,
                    "home_moneyline": None,
                    "away_moneyline": None,
                }
            )
    id_map = pd.DataFrame(id_rows).drop_duplicates(["season", "week", "home_team", "away_team"])
    lines = pd.DataFrame(line_rows)
    return id_map, lines


def _migrate_box_stats(legacy) -> pd.DataFrame:
    frames = []
    for year, table in STAT_TABLES.items():
        if not store.table_exists(legacy, table):
            continue
        raw = pd.read_sql(f'SELECT * FROM "{table}"', legacy)
        rows = [flatten_legacy_box(rec) for rec in raw.to_dict(orient="records")]
        frame = pd.DataFrame(rows).dropna(subset=["game_id", "team"])
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(["game_id", "team"])
