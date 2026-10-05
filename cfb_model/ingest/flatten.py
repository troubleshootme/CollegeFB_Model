"""Flatten CFBD payloads into warehouse rows."""

from __future__ import annotations

from typing import Any, Iterable

from cfb_model.util import (
    as_bool_int,
    as_float,
    as_int,
    enum_value,
    getv,
    iso_date,
    parse_efficiency,
    parse_line_scores,
    parse_pair,
    parse_possession,
)


def _iter(payload: Any) -> Iterable[Any]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    return [payload]


def flatten_games(payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for game in _iter(payload):
        game_id = as_int(getv(game, "id"))
        if game_id is None:
            continue
        home_q1, home_q2, home_q3, home_q4 = parse_line_scores(
            getv(game, "home_line_scores") or getv(game, "homeLineScores")
        )
        away_q1, away_q2, away_q3, away_q4 = parse_line_scores(
            getv(game, "away_line_scores") or getv(game, "awayLineScores")
        )
        rows.append(
            {
                "id": game_id,
                "season": as_int(getv(game, "season")),
                "week": as_int(getv(game, "week")),
                "season_type": enum_value(getv(game, "season_type")),
                "start_date": iso_date(getv(game, "start_date")),
                "completed": as_bool_int(getv(game, "completed")),
                "neutral_site": as_bool_int(getv(game, "neutral_site")),
                "conference_game": as_bool_int(getv(game, "conference_game")),
                "venue_id": as_int(getv(game, "venue_id")),
                "venue": getv(game, "venue"),
                "home_id": as_int(getv(game, "home_id")),
                "home_team": getv(game, "home_team"),
                "home_conference": getv(game, "home_conference"),
                "home_classification": enum_value(getv(game, "home_classification")),
                "home_points": as_float(getv(game, "home_points")),
                "home_q1": home_q1,
                "home_q2": home_q2,
                "home_q3": home_q3,
                "home_q4": home_q4,
                "home_pregame_elo": as_float(getv(game, "home_pregame_elo")),
                "home_postgame_elo": as_float(getv(game, "home_postgame_elo")),
                "home_postgame_wp": as_float(
                    getv(game, "home_postgame_win_probability")
                    or getv(game, "home_postgame_wp")
                    or getv(game, "home_post_win_prob")
                ),
                "away_id": as_int(getv(game, "away_id")),
                "away_team": getv(game, "away_team"),
                "away_conference": getv(game, "away_conference"),
                "away_classification": enum_value(getv(game, "away_classification")),
                "away_points": as_float(getv(game, "away_points")),
                "away_q1": away_q1,
                "away_q2": away_q2,
                "away_q3": away_q3,
                "away_q4": away_q4,
                "away_pregame_elo": as_float(getv(game, "away_pregame_elo")),
                "away_postgame_elo": as_float(getv(game, "away_postgame_elo")),
                "away_postgame_wp": as_float(
                    getv(game, "away_postgame_win_probability")
                    or getv(game, "away_postgame_wp")
                    or getv(game, "away_post_win_prob")
                ),
                "excitement_index": as_float(getv(game, "excitement_index")),
            }
        )
    return rows


def flatten_lines(payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for game in _iter(payload):
        game_id = as_int(getv(game, "id") or getv(game, "game_id"))
        if game_id is None:
            continue
        for line in _iter(getv(game, "lines") or []):
            provider = getv(line, "provider") or "unknown"
            rows.append(
                {
                    "game_id": game_id,
                    "provider": provider,
                    "spread": as_float(getv(line, "spread")),
                    "spread_open": as_float(getv(line, "spread_open")),
                    "over_under": as_float(getv(line, "over_under")),
                    "over_under_open": as_float(getv(line, "over_under_open")),
                    "home_moneyline": as_float(getv(line, "home_moneyline")),
                    "away_moneyline": as_float(getv(line, "away_moneyline")),
                }
            )
    return rows


def flatten_ppa(payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _iter(payload):
        game_id = as_int(getv(item, "game_id"))
        team = getv(item, "team")
        if game_id is None or not team:
            continue
        offense = getv(item, "offense") or {}
        defense = getv(item, "defense") or {}
        rows.append(
            {
                "game_id": game_id,
                "season": as_int(getv(item, "season")),
                "week": as_int(getv(item, "week")),
                "season_type": enum_value(getv(item, "season_type")),
                "team": team,
                "conference": getv(item, "conference"),
                "opponent": getv(item, "opponent"),
                "off_overall": as_float(getv(offense, "overall")),
                "off_passing": as_float(getv(offense, "passing")),
                "off_rushing": as_float(getv(offense, "rushing")),
                "off_first_down": as_float(getv(offense, "first_down")),
                "off_second_down": as_float(getv(offense, "second_down")),
                "off_third_down": as_float(getv(offense, "third_down")),
                "def_overall": as_float(getv(defense, "overall")),
                "def_passing": as_float(getv(defense, "passing")),
                "def_rushing": as_float(getv(defense, "rushing")),
            }
        )
    return rows


def flatten_advanced(payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _iter(payload):
        game_id = as_int(getv(item, "game_id"))
        team = getv(item, "team")
        if game_id is None or not team:
            continue
        offense = getv(item, "offense") or {}
        defense = getv(item, "defense") or {}
        rows.append(
            {
                "game_id": game_id,
                "season": as_int(getv(item, "season")),
                "week": as_int(getv(item, "week")),
                "season_type": enum_value(getv(item, "season_type")),
                "team": team,
                "opponent": getv(item, "opponent"),
                "off_success_rate": as_float(getv(offense, "success_rate")),
                "off_explosiveness": as_float(getv(offense, "explosiveness")),
                "off_ppa": as_float(getv(offense, "ppa")),
                "off_stuff_rate": as_float(getv(offense, "stuff_rate")),
                "off_line_yards": as_float(getv(offense, "line_yards")),
                "off_plays": as_int(getv(offense, "plays")),
                "def_success_rate": as_float(getv(defense, "success_rate")),
                "def_explosiveness": as_float(getv(defense, "explosiveness")),
                "def_ppa": as_float(getv(defense, "ppa")),
                "def_stuff_rate": as_float(getv(defense, "stuff_rate")),
                "def_line_yards": as_float(getv(defense, "line_yards")),
                "def_plays": as_int(getv(defense, "plays")),
            }
        )
    return rows


def flatten_havoc(payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _iter(payload):
        game_id = as_int(getv(item, "game_id"))
        team = getv(item, "team")
        if game_id is None or not team:
            continue
        offense = getv(item, "offense") or {}
        defense = getv(item, "defense") or {}
        rows.append(
            {
                "game_id": game_id,
                "season": as_int(getv(item, "season")),
                "week": as_int(getv(item, "week")),
                "team": team,
                "opponent": getv(item, "opponent"),
                "off_havoc": as_float(getv(offense, "havoc_rate") or getv(offense, "total")),
                "def_havoc": as_float(getv(defense, "havoc_rate") or getv(defense, "total")),
            }
        )
    return rows


def flatten_team_game_stats(payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for game in _iter(payload):
        game_id = as_int(getv(game, "id") or getv(game, "game_id"))
        for team in _iter(getv(game, "teams") or []):
            stats = {getv(s, "category"): getv(s, "stat") for s in _iter(getv(team, "stats") or [])}
            third_conv, third_att, _ = parse_efficiency(stats.get("thirdDownEff") or stats.get("thirdDownConversions"))
            fourth_conv, fourth_att, _ = parse_efficiency(stats.get("fourthDownEff"))
            completions, pass_att = parse_pair(stats.get("completionAttempts"))
            team_name = getv(team, "team")
            if game_id is None or not team_name:
                continue
            rows.append(
                {
                    "game_id": game_id,
                    "team_id": as_int(getv(team, "team_id") or getv(team, "id")),
                    "team": team_name,
                    "home_away": getv(team, "home_away"),
                    "conference": getv(team, "conference"),
                    "points": as_float(getv(team, "points") or stats.get("points")),
                    "first_downs": as_float(stats.get("firstDowns")),
                    "third_down_att": third_att,
                    "third_down_conv": third_conv,
                    "fourth_down_att": fourth_att,
                    "fourth_down_conv": fourth_conv,
                    "possession_seconds": parse_possession(stats.get("possessionTime")),
                    "total_yards": as_float(stats.get("totalYards")),
                    "rushing_yards": as_float(stats.get("rushingYards")),
                    "rushing_attempts": as_float(stats.get("rushingAttempts")),
                    "net_passing_yards": as_float(stats.get("netPassingYards")),
                    "yards_per_pass": as_float(stats.get("yardsPerPass")),
                    "yards_per_rush": as_float(stats.get("yardsPerRushAttempt")),
                    "completions": completions,
                    "pass_attempts": pass_att,
                    "turnovers": as_float(stats.get("turnovers")),
                    "interceptions": as_float(stats.get("interceptions")),
                    "fumbles_lost": as_float(stats.get("fumblesLost")),
                    "sacks": as_float(stats.get("sacks")),
                    "tackles_for_loss": as_float(stats.get("tacklesForLoss")),
                }
            )
    return rows


def flatten_legacy_box(row: dict[str, Any], *, game_id: int | None = None) -> dict[str, Any]:
    third_conv, third_att, _ = parse_efficiency(row.get("thirdDownEff"))
    fourth_conv, fourth_att, _ = parse_efficiency(row.get("fourthDownEff"))
    completions, pass_att = parse_pair(row.get("completionAttempts"))
    return {
        "game_id": as_int(game_id or row.get("game_id")),
        "team_id": as_int(row.get("team_id")),
        "team": row.get("team_name") or row.get("team"),
        "home_away": row.get("home_away"),
        "conference": row.get("conference"),
        "points": as_float(row.get("points")),
        "first_downs": as_float(row.get("firstDowns")),
        "third_down_att": third_att,
        "third_down_conv": third_conv,
        "fourth_down_att": fourth_att,
        "fourth_down_conv": fourth_conv,
        "possession_seconds": parse_possession(row.get("possessionTime")),
        "total_yards": as_float(row.get("totalYards")),
        "rushing_yards": as_float(row.get("rushingYards")),
        "rushing_attempts": as_float(row.get("rushingAttempts")),
        "net_passing_yards": as_float(row.get("netPassingYards")),
        "yards_per_pass": as_float(row.get("yardsPerPass")),
        "yards_per_rush": as_float(row.get("yardsPerRushAttempt")),
        "completions": completions,
        "pass_attempts": pass_att,
        "turnovers": as_float(row.get("turnovers")),
        "interceptions": as_float(row.get("interceptions")),
        "fumbles_lost": as_float(row.get("fumblesLost")),
        "sacks": as_float(row.get("sacks")),
        "tackles_for_loss": as_float(row.get("tacklesForLoss")),
    }


def flatten_elo(payload: Any, *, year: int, week: int, season_type: str) -> list[dict[str, Any]]:
    rows = []
    for item in _iter(payload):
        team = getv(item, "team")
        if not team:
            continue
        rows.append(
            {
                "year": as_int(getv(item, "year") or year),
                "week": week,
                "season_type": season_type,
                "team": team,
                "conference": getv(item, "conference"),
                "elo": as_float(getv(item, "elo")),
            }
        )
    return rows


def flatten_core(payload: Any) -> list[dict[str, Any]]:
    rows = []
    for item in _iter(payload):
        team = getv(item, "team")
        if not team:
            continue
        rows.append(
            {
                "year": as_int(getv(item, "year")),
                "through_week": as_int(getv(item, "through_week")),
                "through_season_type": enum_value(getv(item, "through_season_type")),
                "team": team,
                "conference": getv(item, "conference"),
                "overall": as_float(getv(item, "overall")),
                "offense": as_float(getv(item, "offense")),
                "defense": as_float(getv(item, "defense")),
                "model_version": getv(item, "model_version"),
            }
        )
    return rows


def flatten_sp(payload: Any) -> list[dict[str, Any]]:
    rows = []
    for item in _iter(payload):
        team = getv(item, "team")
        if not team:
            continue
        offense = getv(item, "offense") or {}
        defense = getv(item, "defense") or {}
        special = getv(item, "special_teams") or {}
        rows.append(
            {
                "year": as_int(getv(item, "year")),
                "team": team,
                "rating": as_float(getv(item, "rating")),
                "ranking": as_int(getv(item, "ranking")),
                "offense": as_float(getv(offense, "rating")),
                "defense": as_float(getv(defense, "rating")),
                "special_teams": as_float(getv(special, "rating")),
            }
        )
    return rows


def flatten_talent(payload: Any) -> list[dict[str, Any]]:
    rows = []
    for item in _iter(payload):
        team = getv(item, "team")
        year = as_int(getv(item, "year"))
        if not team or year is None:
            continue
        rows.append({"year": year, "team": team, "talent": as_float(getv(item, "talent"))})
    return rows


def flatten_recruiting(payload: Any) -> list[dict[str, Any]]:
    rows = []
    for item in _iter(payload):
        team = getv(item, "team")
        year = as_int(getv(item, "year"))
        if not team or year is None:
            continue
        rows.append(
            {
                "year": year,
                "team": team,
                "rank": as_int(getv(item, "rank")),
                "points": as_float(getv(item, "points")),
            }
        )
    return rows


def flatten_returning(payload: Any) -> list[dict[str, Any]]:
    rows = []
    for item in _iter(payload):
        team = getv(item, "team")
        season = as_int(getv(item, "season") or getv(item, "year"))
        if not team or season is None:
            continue
        rows.append(
            {
                "season": season,
                "team": team,
                "conference": getv(item, "conference"),
                "percent_ppa": as_float(getv(item, "percent_ppa")),
                "percent_passing_ppa": as_float(getv(item, "percent_passing_ppa")),
                "percent_rushing_ppa": as_float(getv(item, "percent_rushing_ppa")),
                "usage": as_float(getv(item, "usage")),
            }
        )
    return rows


def flatten_venues(payload: Any) -> list[dict[str, Any]]:
    rows = []
    for item in _iter(payload):
        venue_id = as_int(getv(item, "id"))
        if venue_id is None:
            continue
        rows.append(
            {
                "id": venue_id,
                "name": getv(item, "name"),
                "city": getv(item, "city"),
                "state": getv(item, "state"),
                "timezone": getv(item, "timezone"),
                "latitude": as_float(getv(item, "latitude")),
                "longitude": as_float(getv(item, "longitude")),
                "elevation": as_float(getv(item, "elevation")),
                "capacity": as_float(getv(item, "capacity")),
                "grass": as_bool_int(getv(item, "grass")),
                "dome": as_bool_int(getv(item, "dome")),
            }
        )
    return rows


def flatten_teams(payload: Any) -> list[dict[str, Any]]:
    rows = []
    for item in _iter(payload):
        team_id = as_int(getv(item, "id"))
        school = getv(item, "school")
        if team_id is None or not school:
            continue
        loc = getv(item, "location") or {}
        rows.append(
            {
                "id": team_id,
                "school": school,
                "conference": getv(item, "conference"),
                "classification": enum_value(getv(item, "classification")),
                "venue_id": as_int(getv(loc, "id") or getv(loc, "venue_id")),
                "latitude": as_float(getv(loc, "latitude")),
                "longitude": as_float(getv(loc, "longitude")),
                "elevation": as_float(getv(loc, "elevation")),
                "capacity": as_float(getv(loc, "capacity")),
                "grass": as_bool_int(getv(loc, "grass")),
                "dome": as_bool_int(getv(loc, "dome")),
            }
        )
    return rows


def flatten_coaches(payload: Any) -> list[dict[str, Any]]:
    rows = []
    for coach in _iter(payload):
        coach_id = as_int(getv(coach, "id"))
        first = getv(coach, "first_name")
        last = getv(coach, "last_name")
        seasons = getv(coach, "seasons") or []
        if seasons:
            for season in _iter(seasons):
                school = getv(season, "school") or getv(season, "team")
                year = as_int(getv(season, "year"))
                if not school or year is None:
                    continue
                rows.append(
                    {
                        "coach_id": coach_id,
                        "first_name": first,
                        "last_name": last,
                        "school": school,
                        "year": year,
                        "games": as_int(getv(season, "games")),
                        "wins": as_int(getv(season, "wins")),
                        "losses": as_int(getv(season, "losses")),
                        "sp_overall": as_float(getv(season, "sp_overall")),
                        "sp_offense": as_float(getv(season, "sp_offense")),
                        "sp_defense": as_float(getv(season, "sp_defense")),
                        "srs": as_float(getv(season, "srs")),
                    }
                )
        else:
            school = getv(coach, "school") or getv(coach, "team")
            year = as_int(getv(coach, "year"))
            if school and year is not None:
                rows.append(
                    {
                        "coach_id": coach_id,
                        "first_name": first,
                        "last_name": last or getv(coach, "season_type"),
                        "school": school,
                        "year": year,
                        "games": as_int(getv(coach, "games")),
                        "wins": None,
                        "losses": as_int(getv(coach, "losses")),
                        "sp_overall": as_float(getv(coach, "sp_overall")),
                        "sp_offense": as_float(getv(coach, "sp_offense")),
                        "sp_defense": as_float(getv(coach, "sp_defense")),
                        "srs": None,
                    }
                )
    return rows


def flatten_wepa(payload: Any) -> list[dict[str, Any]]:
    rows = []
    for item in _iter(payload):
        team = getv(item, "team")
        year = as_int(getv(item, "year"))
        if not team or year is None:
            continue
        epa = getv(item, "epa") or {}
        allowed = getv(item, "epa_allowed") or {}
        success = getv(item, "success_rate") or {}
        success_allowed = getv(item, "success_rate_allowed") or {}
        rows.append(
            {
                "year": year,
                "team": team,
                "conference": getv(item, "conference"),
                "epa_total": as_float(getv(epa, "total")),
                "epa_passing": as_float(getv(epa, "passing")),
                "epa_rushing": as_float(getv(epa, "rushing")),
                "epa_allowed_total": as_float(getv(allowed, "total")),
                "success_rate": as_float(getv(success, "total")),
                "success_rate_allowed": as_float(getv(success_allowed, "total")),
                "explosiveness": as_float(getv(item, "explosiveness")),
                "explosiveness_allowed": as_float(getv(item, "explosiveness_allowed")),
            }
        )
    return rows


def flatten_fpi(payload: Any) -> list[dict[str, Any]]:
    rows = []
    for item in _iter(payload):
        team = getv(item, "team")
        year = as_int(getv(item, "year") or getv(item, "season"))
        if not team or year is None:
            continue
        eff = getv(item, "efficiencies") or {}
        rows.append(
            {
                "year": year,
                "team": team,
                "conference": getv(item, "conference"),
                "fpi": as_float(getv(item, "fpi") or getv(item, "rating")),
                "fpi_rank": as_int(getv(item, "fpi_rank") or getv(item, "rank")),
                "offense": as_float(getv(eff, "offense")),
                "defense": as_float(getv(eff, "defense")),
                "special_teams": as_float(getv(eff, "special_teams") or getv(eff, "specialTeams")),
                "overall_eff": as_float(getv(eff, "overall")),
                "source": getv(item, "source") or "cfbd",
                "as_of": iso_date(getv(item, "as_of")),
            }
        )
    return rows


def flatten_player_season_stats(payload: Any) -> list[dict[str, Any]]:
    rows = []
    for item in _iter(payload):
        player = getv(item, "player") or getv(item, "name") or getv(item, "athlete")
        if isinstance(player, dict):
            player_id = as_int(getv(player, "id")) or getv(player, "id")
            position = getv(player, "position")
            player = getv(player, "name") or getv(player, "display_name")
        else:
            player_id = getv(item, "player_id") or getv(item, "athlete_id") or getv(item, "id")
            position = getv(item, "position")
        team = getv(item, "team") or getv(item, "school")
        season = as_int(getv(item, "season") or getv(item, "year"))
        if not player or not team or season is None:
            continue
        stats = getv(item, "stat") or getv(item, "stats") or item
        category = (getv(item, "category") or getv(item, "stat_type") or "all")
        if isinstance(category, str):
            category = category.lower()
        else:
            category = "all"
        rows.append(
            {
                "season": season,
                "player_id": None if player_id is None else str(player_id),
                "player": player,
                "position": position if not isinstance(position, dict) else getv(position, "abbreviation"),
                "team": team,
                "conference": getv(item, "conference"),
                "category": category,
                "games": as_float(getv(item, "games") or getv(stats, "games")),
                "attempts": as_float(
                    getv(stats, "attempts")
                    or getv(item, "passing_attempts")
                    or getv(item, "rushing_attempts")
                ),
                "completions": as_float(getv(stats, "completions") or getv(item, "passing_completions")),
                "yards": as_float(
                    getv(stats, "yards") or getv(item, "passing_yards") or getv(item, "rushing_yards")
                ),
                "touchdowns": as_float(
                    getv(stats, "touchdowns") or getv(item, "passing_tds") or getv(item, "rushing_tds")
                ),
                "interceptions": as_float(getv(stats, "interceptions") or getv(item, "passing_ints")),
                "yards_per_attempt": as_float(getv(stats, "yards_per_attempt") or getv(item, "yards_per_pass")),
                "yards_per_carry": as_float(getv(stats, "yards_per_carry") or getv(item, "yards_per_rush")),
                "rating": as_float(getv(stats, "rating") or getv(item, "qbr") or getv(item, "passer_rating")),
            }
        )
    return rows


def flatten_player_ppa(payload: Any) -> list[dict[str, Any]]:
    rows = []
    for item in _iter(payload):
        player = getv(item, "name") or getv(item, "player")
        team = getv(item, "team") or getv(item, "school")
        season = as_int(getv(item, "season") or getv(item, "year"))
        if not player or not team or season is None:
            continue
        avg = getv(item, "average_ppa") or getv(item, "averagePPA") or {}
        if not isinstance(avg, dict):
            avg = {"all": avg}
        usage = getv(item, "usage") or {}
        rows.append(
            {
                "season": season,
                "player_id": None if getv(item, "player_id") is None and getv(item, "id") is None else str(
                    getv(item, "player_id") or getv(item, "id")
                ),
                "player": player,
                "position": getv(item, "position"),
                "team": team,
                "conference": getv(item, "conference"),
                "average_ppa": as_float(getv(avg, "all") or getv(avg, "overall") or getv(item, "average_ppa")),
                "passing": as_float(getv(avg, "pass") or getv(avg, "passing")),
                "rushing": as_float(getv(avg, "rush") or getv(avg, "rushing")),
                "usage": as_float(getv(usage, "overall") if isinstance(usage, dict) else usage),
            }
        )
    return rows


def flatten_portal(payload: Any) -> list[dict[str, Any]]:
    rows = []
    for item in _iter(payload):
        first = getv(item, "first_name") or ""
        last = getv(item, "last_name") or ""
        player = getv(item, "player") or getv(item, "name") or f"{first} {last}".strip()
        season = as_int(getv(item, "season") or getv(item, "year"))
        origin = getv(item, "origin") or getv(item, "from_team") or getv(item, "previous")
        if not player or season is None:
            continue
        rows.append(
            {
                "season": season,
                "player_id": None if getv(item, "player_id") is None else str(getv(item, "player_id")),
                "player": player,
                "position": getv(item, "position"),
                "origin": origin,
                "destination": getv(item, "destination") or getv(item, "to_team") or getv(item, "school"),
                "stars": as_float(getv(item, "stars") or getv(item, "rating")),
                "eligibility": getv(item, "eligibility"),
                "transfer_date": iso_date(getv(item, "transfer_date") or getv(item, "date")),
            }
        )
    return rows


def flatten_pregame_wp(payload: Any) -> list[dict[str, Any]]:
    rows = []
    for item in _iter(payload):
        game_id = as_int(getv(item, "game_id"))
        if game_id is None:
            continue
        rows.append(
            {
                "game_id": game_id,
                "season": as_int(getv(item, "season")),
                "week": as_int(getv(item, "week")),
                "season_type": enum_value(getv(item, "season_type")),
                "home_team": getv(item, "home_team"),
                "away_team": getv(item, "away_team"),
                "spread": as_float(getv(item, "spread")),
                "home_win_probability": as_float(getv(item, "home_win_probability")),
            }
        )
    return rows
