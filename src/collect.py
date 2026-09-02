from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Any

import pandas as pd

from src.cfbd_client import get_json
from src.config import DB_PATH, END_YEAR, SEASON_TYPES, START_YEAR, WEEKS


def _flatten_location(location: Any) -> dict[str, Any]:
    if not isinstance(location, dict):
        return {}
    return {
        "venue_id": location.get("venueId") or location.get("id"),
        "venue_name": location.get("name"),
        "city": location.get("city"),
        "state": location.get("state"),
        "zip": location.get("zip"),
        "timezone": location.get("timezone"),
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "elevation": location.get("elevation"),
        "capacity": location.get("capacity"),
        "grass": location.get("grass"),
        "dome": location.get("dome"),
        "year_constructed": location.get("yearConstructed") or location.get("constructionYear"),
    }


def fetch_games(year: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for season_type in SEASON_TYPES:
        payload = get_json(
            "/games",
            {
                "year": year,
                "seasonType": season_type,
                "classification": "fbs",
            },
        )
        for game in payload or []:
            rows.append(
                {
                    "game_id": game.get("id"),
                    "season": game.get("season"),
                    "week": game.get("week"),
                    "season_type": game.get("seasonType"),
                    "start_date": game.get("startDate"),
                    "start_time_tbd": game.get("startTimeTBD"),
                    "completed": bool(game.get("completed")),
                    "neutral_site": bool(game.get("neutralSite")),
                    "conference_game": bool(game.get("conferenceGame")),
                    "attendance": game.get("attendance"),
                    "venue_id": game.get("venueId"),
                    "venue": game.get("venue"),
                    "home_id": game.get("homeId"),
                    "home_team": game.get("homeTeam"),
                    "home_classification": game.get("homeClassification"),
                    "home_conference": game.get("homeConference"),
                    "home_points": game.get("homePoints"),
                    "home_pregame_elo": game.get("homePregameElo"),
                    "home_postgame_elo": game.get("homePostgameElo"),
                    "home_post_win_prob": game.get("homePostgameWinProbability"),
                    "away_id": game.get("awayId"),
                    "away_team": game.get("awayTeam"),
                    "away_classification": game.get("awayClassification"),
                    "away_conference": game.get("awayConference"),
                    "away_points": game.get("awayPoints"),
                    "away_pregame_elo": game.get("awayPregameElo"),
                    "away_postgame_elo": game.get("awayPostgameElo"),
                    "away_post_win_prob": game.get("awayPostgameWinProbability"),
                    "excitement_index": game.get("excitementIndex"),
                    "notes": game.get("notes"),
                }
            )
    return pd.DataFrame(rows)


def _parse_team_stats(game: dict[str, Any], season: int, week: int, season_type: str) -> list[dict[str, Any]]:
    rows = []
    for team in game.get("teams") or []:
        stats = {item.get("category"): item.get("stat") for item in team.get("stats") or []}
        rows.append(
            {
                "game_id": game.get("id"),
                "season": season,
                "week": week,
                "season_type": season_type,
                "team_id": team.get("teamId"),
                "team_name": team.get("team"),
                "conference": team.get("conference"),
                "home_away": team.get("homeAway"),
                "points": team.get("points"),
                **stats,
            }
        )
    return rows


def fetch_team_stats(year: int) -> pd.DataFrame:
    jobs: list[tuple[str, int]] = [(season_type, week) for season_type in SEASON_TYPES for week in WEEKS]
    rows: list[dict[str, Any]] = []

    def _one(season_type: str, week: int) -> list[dict[str, Any]]:
        payload = get_json(
            "/games/teams",
            {
                "year": year,
                "week": week,
                "seasonType": season_type,
                "classification": "fbs",
            },
        )
        out: list[dict[str, Any]] = []
        for game in payload or []:
            out.extend(_parse_team_stats(game, year, week, season_type))
        return out

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(_one, season_type, week) for season_type, week in jobs]
        for future in as_completed(futures):
            rows.extend(future.result())
    return pd.DataFrame(rows)


def fetch_lines(year: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for season_type in SEASON_TYPES:
        payload = get_json("/lines", {"year": year, "seasonType": season_type})
        if not payload:
            for week in WEEKS:
                payload = get_json("/lines", {"year": year, "seasonType": season_type, "week": week})
                rows.extend(_parse_lines(payload or []))
            continue
        rows.extend(_parse_lines(payload))
    return pd.DataFrame(rows)


def _parse_lines(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for game in payload:
        for line in game.get("lines") or []:
            rows.append(
                {
                    "game_id": game.get("id"),
                    "season": game.get("season"),
                    "season_type": game.get("seasonType"),
                    "week": game.get("week"),
                    "start_date": game.get("startDate"),
                    "home_team": game.get("homeTeam"),
                    "away_team": game.get("awayTeam"),
                    "home_score": game.get("homeScore"),
                    "away_score": game.get("awayScore"),
                    "provider": line.get("provider"),
                    "spread": line.get("spread"),
                    "formatted_spread": line.get("formattedSpread"),
                    "spread_open": line.get("spreadOpen"),
                    "over_under": line.get("overUnder"),
                    "over_under_open": line.get("overUnderOpen"),
                    "home_moneyline": line.get("homeMoneyline"),
                    "away_moneyline": line.get("awayMoneyline"),
                }
            )
    return rows


def _normalize_hex(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in text):
        return None
    return f"#{text.upper()}"


def _logo_url(team: dict[str, Any]) -> str | None:
    logos = team.get("logos")
    if isinstance(logos, list):
        for item in logos:
            if item:
                return str(item)
    logo = team.get("logo")
    if logo:
        return str(logo)
    return None


def branding_fields(team: dict[str, Any]) -> dict[str, Any]:
    return {
        "abbreviation": team.get("abbreviation"),
        "color": _normalize_hex(team.get("color")),
        "alt_color": _normalize_hex(team.get("altColor") or team.get("alt_color")),
        "logo": _logo_url(team),
    }


def fetch_fbs_teams(year: int) -> pd.DataFrame:
    rows = []
    for team in get_json("/teams/fbs", {"year": year}) or []:
        loc = _flatten_location(team.get("location"))
        rows.append(
            {
                "year": year,
                "team_id": team.get("id"),
                "school": team.get("school"),
                "conference": team.get("conference"),
                "division": team.get("division"),
                "classification": team.get("classification"),
                **branding_fields(team),
                **loc,
            }
        )
    return pd.DataFrame(rows)


def fetch_venues() -> pd.DataFrame:
    rows = []
    for venue in get_json("/venues") or []:
        rows.append(
            {
                "venue_id": venue.get("id"),
                "name": venue.get("name"),
                "capacity": venue.get("capacity"),
                "grass": venue.get("grass"),
                "dome": venue.get("dome"),
                "city": venue.get("city"),
                "state": venue.get("state"),
                "latitude": venue.get("latitude"),
                "longitude": venue.get("longitude"),
                "elevation": venue.get("elevation"),
                "year_constructed": venue.get("constructionYear"),
            }
        )
    return pd.DataFrame(rows)


def fetch_talent(year: int) -> pd.DataFrame:
    payload = get_json("/talent", {"year": year}) or []
    return pd.DataFrame(
        [{"year": row.get("year", year), "team": row.get("team"), "talent": row.get("talent")} for row in payload]
    )


def fetch_sp(year: int) -> pd.DataFrame:
    rows = []
    for row in get_json("/ratings/sp", {"year": year}) or []:
        offense = row.get("offense") or {}
        defense = row.get("defense") or {}
        special = row.get("specialTeams") or {}
        rows.append(
            {
                "year": row.get("year", year),
                "team": row.get("team"),
                "conference": row.get("conference"),
                "sp_overall": row.get("rating"),
                "sp_ranking": row.get("ranking"),
                "sp_offense": offense.get("rating") if isinstance(offense, dict) else None,
                "sp_defense": defense.get("rating") if isinstance(defense, dict) else None,
                "sp_special": special.get("rating") if isinstance(special, dict) else None,
            }
        )
    return pd.DataFrame(rows)


def fetch_recruiting(year: int) -> pd.DataFrame:
    payload = get_json("/recruiting/teams", {"year": year}) or []
    return pd.DataFrame(
        [
            {
                "year": row.get("year", year),
                "team": row.get("team"),
                "recruit_rank": row.get("rank"),
                "recruit_points": row.get("points"),
            }
            for row in payload
        ]
    )


def fetch_returning(year: int) -> pd.DataFrame:
    payload = get_json("/player/returning", {"year": year}) or []
    rows = []
    for row in payload:
        rows.append(
            {
                "year": row.get("season", year),
                "team": row.get("team"),
                "conference": row.get("conference"),
                "returning_total_ppa": row.get("totalPPA"),
                "returning_percent_ppa": row.get("percentPPA"),
                "returning_usage": row.get("usage"),
            }
        )
    return pd.DataFrame(rows)


def fetch_coaches(year: int) -> pd.DataFrame:
    rows = []
    for coach in get_json("/coaches", {"year": year}) or []:
        for season in coach.get("seasons") or []:
            if season.get("year") != year:
                continue
            rows.append(
                {
                    "year": year,
                    "first_name": coach.get("firstName"),
                    "last_name": coach.get("lastName"),
                    "hire_date": coach.get("hireDate"),
                    "school": season.get("school"),
                    "conference": season.get("conference"),
                    "games": season.get("games"),
                    "wins": season.get("wins"),
                    "losses": season.get("losses"),
                    "win_pct": season.get("winPercentage"),
                    "preseason_rank": season.get("preseasonRank"),
                    "postseason_rank": season.get("postseasonRank"),
                    "sp_overall": season.get("spOverall"),
                    "sp_offense": season.get("spOffense"),
                    "sp_defense": season.get("spDefense"),
                }
            )
    return pd.DataFrame(rows)


def fetch_ppa(year: int) -> pd.DataFrame:
    rows = []
    for row in get_json("/ppa/teams", {"year": year}) or []:
        offense = row.get("offense") or {}
        defense = row.get("defense") or {}
        rows.append(
            {
                "year": row.get("season", year),
                "team": row.get("team"),
                "conference": row.get("conference"),
                "off_ppa": (offense.get("overall") or {}).get("average") if isinstance(offense.get("overall"), dict) else offense.get("overall"),
                "def_ppa": (defense.get("overall") or {}).get("average") if isinstance(defense.get("overall"), dict) else defense.get("overall"),
            }
        )
    return pd.DataFrame(rows)


def fetch_advanced_season(year: int) -> pd.DataFrame:
    rows = []
    for row in get_json("/stats/season/advanced", {"year": year}) or []:
        offense = row.get("offense") or {}
        defense = row.get("defense") or {}
        rows.append(
            {
                "year": row.get("season", year),
                "team": row.get("team"),
                "conference": row.get("conference"),
                "off_success_rate": offense.get("successRate"),
                "off_explosiveness": offense.get("explosiveness"),
                "off_ppa": offense.get("ppa"),
                "off_stuff_rate": offense.get("stuffRate"),
                "def_success_rate": defense.get("successRate"),
                "def_explosiveness": defense.get("explosiveness"),
                "def_ppa": defense.get("ppa"),
                "def_stuff_rate": defense.get("stuffRate"),
            }
        )
    return pd.DataFrame(rows)


def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [part for part in parts if part is not None and not part.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _write_table(con: sqlite3.Connection, name: str, frame: pd.DataFrame) -> None:
    if frame is None or frame.empty:
        print(f"  skip {name}: empty")
        return
    frame.to_sql(name, con, if_exists="replace", index=False)
    print(f"  wrote {name}: {len(frame):,} rows")


def _table_has_rows(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    if not row:
        return False
    count = con.execute(f'SELECT 1 FROM "{name}" LIMIT 1').fetchone()
    return count is not None


def _read_table(con: sqlite3.Connection, name: str) -> pd.DataFrame:
    if not _table_has_rows(con, name):
        return pd.DataFrame()
    return pd.read_sql_query(f'SELECT * FROM "{name}"', con)


def stored_seasons(con: sqlite3.Connection) -> set[int]:
    if not _table_has_rows(con, "games"):
        return set()
    rows = con.execute("SELECT DISTINCT season FROM games WHERE season IS NOT NULL").fetchall()
    return {int(season) for (season,) in rows}


def live_seasons(today: date | None = None) -> set[int]:
    today = today or date.today()
    if today.month == 1:
        return {today.year - 1, today.year}
    return {today.year}


def seasons_to_fetch(
    start_year: int,
    end_year: int,
    existing: set[int],
    today: date | None = None,
) -> list[int]:
    live = live_seasons(today)
    return [year for year in range(start_year, end_year + 1) if year in live or year not in existing]


def merge_year_frames(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    year_col: str,
    years: list[int],
) -> pd.DataFrame:
    if existing is None or existing.empty:
        return incoming.copy() if incoming is not None and not incoming.empty else pd.DataFrame()
    if incoming is None or incoming.empty or year_col not in incoming.columns:
        return existing
    arrived = {int(year) for year in incoming[year_col].dropna().unique()}
    drop = [year for year in years if year in arrived]
    keep = existing[~existing[year_col].isin(drop)] if drop and year_col in existing.columns else existing
    return pd.concat([keep, incoming], ignore_index=True)


def _latest_teams(fbs_teams: pd.DataFrame) -> pd.DataFrame:
    if fbs_teams.empty or "year" not in fbs_teams.columns:
        return pd.DataFrame()
    return (
        fbs_teams[fbs_teams["year"] == fbs_teams["year"].max()]
        .drop(columns=["year"])
        .rename(columns={"team_id": "id"})
    )


def collect(start_year: int = START_YEAR, end_year: int = END_YEAR) -> None:
    with sqlite3.connect(DB_PATH) as con:
        existing = stored_seasons(con)
        have_venues = _table_has_rows(con, "venues")

    years = seasons_to_fetch(start_year, end_year, existing)
    skipped = [year for year in range(start_year, end_year + 1) if year not in years]
    if skipped:
        print(f"Skipping stored historical seasons (no CFBD calls): {skipped[0]}-{skipped[-1]} ({len(skipped)} seasons)")
    if not years:
        print("Nothing to collect; requested seasons are already stored")
        print(f"Database ready at {DB_PATH}")
        return

    label = f"{years[0]}-{years[-1]}" if len(years) > 1 else str(years[0])
    print(f"Collecting CFBD data for {label}")

    games_new = _concat([fetch_games(year) for year in years])
    print(f"games fetched {len(games_new):,}")

    team_stats_parts = []
    for year in years:
        print(f"team stats {year}...")
        team_stats_parts.append(fetch_team_stats(year))
    stats_new = _concat(team_stats_parts)
    print(f"team_game_stats fetched {len(stats_new):,}")

    lines_parts = []
    for year in years:
        print(f"lines {year}...")
        lines_parts.append(fetch_lines(year))
    lines_new = _concat(lines_parts)
    print(f"betting_lines fetched {len(lines_new):,}")

    print("teams / ratings for live or missing seasons...")
    teams_new = _concat([fetch_fbs_teams(year) for year in years])
    if have_venues:
        print("  skip venues: already stored")
        venues_new = pd.DataFrame()
    else:
        venues_new = fetch_venues()
    talent_new = _concat([fetch_talent(year) for year in years])
    sp_new = _concat([fetch_sp(year) for year in years])
    recruiting_new = _concat([fetch_recruiting(year) for year in years])
    returning_new = _concat([fetch_returning(year) for year in years])
    coaches_new = _concat([fetch_coaches(year) for year in years])
    ppa_new = _concat([fetch_ppa(year) for year in years])
    advanced_new = _concat([fetch_advanced_season(year) for year in years])

    with sqlite3.connect(DB_PATH) as con:
        games = merge_year_frames(_read_table(con, "games"), games_new, "season", years)
        team_stats = merge_year_frames(_read_table(con, "team_game_stats"), stats_new, "season", years)
        lines = merge_year_frames(_read_table(con, "betting_lines"), lines_new, "season", years)
        fbs_teams = merge_year_frames(_read_table(con, "fbs_teams"), teams_new, "year", years)
        venues = venues_new if not venues_new.empty else _read_table(con, "venues")
        talent = merge_year_frames(_read_table(con, "talent"), talent_new, "year", years)
        sp_ratings = merge_year_frames(_read_table(con, "sp_ratings"), sp_new, "year", years)
        recruiting = merge_year_frames(_read_table(con, "recruiting"), recruiting_new, "year", years)
        returning = merge_year_frames(_read_table(con, "returning_production"), returning_new, "year", years)
        coaches = merge_year_frames(_read_table(con, "coaches"), coaches_new, "year", years)
        ppa = merge_year_frames(_read_table(con, "ppa_teams"), ppa_new, "year", years)
        advanced = merge_year_frames(_read_table(con, "advanced_season_stats"), advanced_new, "year", years)
        latest_teams = _latest_teams(fbs_teams)

        _write_table(con, "games", games)
        _write_table(con, "team_game_stats", team_stats)
        _write_table(con, "betting_lines", lines)
        _write_table(con, "fbs_teams", fbs_teams)
        _write_table(con, "fbsTeams", latest_teams)
        _write_table(con, "venues", venues)
        _write_table(con, "talent", talent)
        _write_table(con, "talent20s", talent)
        _write_table(con, "sp_ratings", sp_ratings)
        _write_table(con, "recruiting", recruiting)
        _write_table(con, "returning_production", returning)
        _write_table(con, "coaches", coaches)
        _write_table(con, "coachHist", coaches)
        _write_table(con, "ppa_teams", ppa)
        _write_table(con, "advanced_season_stats", advanced)
        con.execute("CREATE INDEX IF NOT EXISTS ix_games_id ON games(game_id)")
        con.execute("CREATE INDEX IF NOT EXISTS ix_games_season ON games(season, week)")
        con.execute("CREATE INDEX IF NOT EXISTS ix_stats_game ON team_game_stats(game_id)")
        con.execute("CREATE INDEX IF NOT EXISTS ix_lines_game ON betting_lines(game_id)")
        con.commit()

    print(f"Database ready at {DB_PATH}")


if __name__ == "__main__":
    collect()
