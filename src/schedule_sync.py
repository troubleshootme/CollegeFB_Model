"""Pull completed scores from cfbschedule. No CFBD calls."""

from __future__ import annotations

import json
import os
import sqlite3
import time
import unicodedata
import urllib.error
import urllib.request
from typing import Any, Callable

import pandas as pd

from src.config import DB_PATH, END_YEAR

DEFAULT_SCHEDULE_URL = "http://127.0.0.1:47392"
_LAST_SYNC = {"at": 0.0, "fetched": 0, "updated": 0, "season": None}


def _same_score(left: object, right: object) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    return int(left) == int(right)


def team_key(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    for char in ("'", "’"):
        text = text.replace(char, "")
    for char in (".", ",", "-"):
        text = text.replace(char, " ")
    return " ".join(text.lower().split())


def schedule_base_url() -> str:
    return os.getenv("CFBSCHEDULE_URL", DEFAULT_SCHEDULE_URL).rstrip("/")


def _match_tuple(season: object, week: object, home: object, away: object) -> tuple[int, int, str, str]:
    return (int(season), int(week), team_key(str(home)), team_key(str(away)))


def apply_schedule_results(games: pd.DataFrame, results: list[dict[str, Any]]) -> tuple[pd.DataFrame, int]:
    if games.empty or not results:
        return games.copy(), 0
    out = games.copy()
    if "completed" in out.columns:
        out["completed"] = out["completed"].fillna(0).astype(int)
    index = {
        _match_tuple(row["season"], row["week"], row["home_team"], row["away_team"]): i
        for i, row in out.iterrows()
    }
    patches: list[tuple[int, int, int]] = []
    for result in results:
        if str(result.get("status") or "").lower() != "completed":
            continue
        home_score = result.get("home_score")
        away_score = result.get("away_score")
        if home_score is None or away_score is None:
            continue
        key = _match_tuple(result.get("season"), result.get("week"), result.get("home_team"), result.get("away_team"))
        idx = index.get(key)
        if idx is None:
            continue
        current = out.loc[idx]
        already = int(current.get("completed") or 0) == 1 and pd.notna(current.get("home_points")) and pd.notna(current.get("away_points"))
        if already and int(current["home_points"]) == int(home_score) and int(current["away_points"]) == int(away_score):
            continue
        out.at[idx, "completed"] = 1
        out.at[idx, "home_points"] = int(home_score)
        out.at[idx, "away_points"] = int(away_score)
        patches.append((int(current["game_id"]), int(home_score), int(away_score)))
    out.attrs["patches"] = patches
    return out, len(patches)


def fetch_schedule_games(
    season: int,
    *,
    base_url: str | None = None,
    opener: Callable[[str], Any] | None = None,
    page_size: int = 500,
) -> list[dict[str, Any]]:
    root = (base_url or schedule_base_url()).rstrip("/")
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        url = f"{root}/api/games?season={int(season)}&limit={page_size}&offset={offset}"
        payload = _get_json(url, opener=opener)
        batch = payload.get("games") or []
        if not batch:
            break
        rows.extend(batch)
        total = int(payload.get("total") or 0)
        offset += len(batch)
        if offset >= total or len(batch) < page_size:
            break
    return rows


def fetch_schedule_json(
    path: str,
    *,
    base_url: str | None = None,
    opener: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    root = (base_url or schedule_base_url()).rstrip("/")
    url = f"{root}{path if path.startswith('/') else '/' + path}"
    return _get_json(url, opener=opener)


def fetch_schedule_ratings(
    season: int,
    *,
    base_url: str | None = None,
    opener: Callable[[str], Any] | None = None,
) -> list[dict[str, Any]]:
    try:
        payload = fetch_schedule_json(
            f"/api/ratings?year={int(season)}",
            base_url=base_url,
            opener=opener,
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return []
    return list(payload.get("ratings") or [])


def fetch_schedule_talent(
    season: int,
    *,
    base_url: str | None = None,
    opener: Callable[[str], Any] | None = None,
) -> list[dict[str, Any]]:
    try:
        payload = fetch_schedule_json(
            f"/api/rankings?year={int(season)}",
            base_url=base_url,
            opener=opener,
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return []
    talent = payload.get("talent") or {}
    return list(talent.get("ranks") or [])


def fetch_schedule_lines(
    season: int,
    *,
    base_url: str | None = None,
    opener: Callable[[str], Any] | None = None,
) -> list[dict[str, Any]]:
    try:
        games = fetch_schedule_games(season, base_url=base_url, opener=opener)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return []
    return lines_from_schedule_games(games)


def ratings_to_sp_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        sp = row.get("sp") if isinstance(row.get("sp"), dict) else {}
        out.append(
            {
                "year": row.get("year"),
                "team": row.get("team"),
                "conference": row.get("conference"),
                "sp_overall": sp.get("overall") if sp else row.get("sp_overall"),
                "sp_ranking": None,
                "sp_offense": sp.get("offense") if sp else row.get("sp_offense"),
                "sp_defense": sp.get("defense") if sp else row.get("sp_defense"),
                "sp_special": (sp.get("special_teams") if sp else None) or row.get("sp_special"),
                "fpi": row.get("fpi"),
                "elo": row.get("elo"),
                "srs": row.get("srs"),
            }
        )
    return out


def talent_to_rows(rows: list[dict[str, Any]], year: int) -> list[dict[str, Any]]:
    return [
        {
            "year": year,
            "team": row.get("school") or row.get("team"),
            "talent": row.get("talent_score") or row.get("talent"),
        }
        for row in rows
        if row.get("school") or row.get("team")
    ]


def lines_from_schedule_games(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for game in games:
        line = game.get("betting_line")
        if not line:
            continue
        home = game.get("home_team") or {}
        away = game.get("away_team") or {}
        out.append(
            {
                "game_id": game.get("id"),
                "season": game.get("season"),
                "week": game.get("week"),
                "home_team": home.get("name") if isinstance(home, dict) else home,
                "away_team": away.get("name") if isinstance(away, dict) else away,
                "provider": line.get("source"),
                "spread": line.get("spread"),
                "over_under": line.get("over_under"),
                "home_moneyline": line.get("home_moneyline"),
                "away_moneyline": line.get("away_moneyline"),
            }
        )
    return out


def _get_json(url: str, opener: Callable[[str], Any] | None = None) -> dict[str, Any]:
    if opener is not None:
        raw = opener(url)
        if isinstance(raw, (bytes, bytearray)):
            return json.loads(raw.decode("utf-8"))
        if isinstance(raw, str):
            return json.loads(raw)
        return raw
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=3) as response:
        return json.loads(response.read().decode("utf-8"))


def _flatten(game: dict[str, Any]) -> dict[str, Any]:
    home = game.get("home_team") or {}
    away = game.get("away_team") or {}
    return {
        "season": game.get("season"),
        "week": game.get("week"),
        "home_team": home.get("name") if isinstance(home, dict) else home,
        "away_team": away.get("name") if isinstance(away, dict) else away,
        "home_score": game.get("home_score"),
        "away_score": game.get("away_score"),
        "status": game.get("status"),
    }


def sync_schedule_results(
    season: int | None = None,
    *,
    db_path=None,
    force: bool = False,
    min_interval: float = 120.0,
    opener: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    year = int(season or END_YEAR)
    now = time.time()
    if not force and _LAST_SYNC["season"] == year and now - float(_LAST_SYNC["at"] or 0) < min_interval:
        return dict(_LAST_SYNC)
    path = db_path or DB_PATH
    if not path.exists():
        report = {"at": now, "fetched": 0, "updated": 0, "season": year, "error": "missing database"}
        _LAST_SYNC.update(report)
        return report
    try:
        remote = fetch_schedule_games(year, opener=opener)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        report = {"at": now, "fetched": 0, "updated": 0, "season": year, "error": str(exc)}
        _LAST_SYNC.update(report)
        return report
    results = [_flatten(game) for game in remote]
    with sqlite3.connect(path) as con:
        games = pd.read_sql_query("SELECT * FROM games", con)
        updated_frame, updated = apply_schedule_results(games, results)
        for game_id, home_points, away_points in updated_frame.attrs.get("patches") or []:
            con.execute(
                "UPDATE games SET completed=1, home_points=?, away_points=? WHERE game_id=?",
                (home_points, away_points, game_id),
            )
        if updated:
            con.commit()
            from src.simulate import invalidate_cache

            invalidate_cache()
    report = {"at": now, "fetched": len(results), "updated": updated, "season": year}
    _LAST_SYNC.update(report)
    return report


def main() -> None:
    report = sync_schedule_results(force=True)
    print(json.dumps(report, indent=2, default=str))
    if report.get("error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
