"""Leakage-safe as-of features. Same-game stats and end-of-season ratings are never used."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cfb_model import config, store
from cfb_model.util import haversine_miles, shrink

# Closing spread is evaluation-only. Opening spread is optional for a blend model.
FEATURE_COLS = [
    "week",
    "neutral_site",
    "conference_game",
    "elo_diff",
    "talent_diff",
    "recruiting_3yr_diff",
    "returning_diff",
    "prior_sp_diff",
    "prior_core_diff",
    "prior_wepa_diff",
    "games_played_home",
    "games_played_away",
    "games_played_diff",
    "rest_days_home",
    "rest_days_away",
    "rest_diff",
    "travel_miles_away",
    "travel_miles_home",
    "elevation_diff",
    "grass",
    "dome",
    "capacity",
    "noise_bucket",
    "temperature",
    "precipitation",
    "wind_speed",
    "margin_std_diff",
    "margin_l4_diff",
    "pf_std_diff",
    "pa_std_diff",
    "prior_margin_diff",
    "off_ppa_std_diff",
    "def_ppa_std_diff",
    "off_success_std_diff",
    "def_success_std_diff",
    "off_expl_std_diff",
    "def_expl_std_diff",
    "havoc_std_diff",
    "off_ppa_shrunk_diff",
    "def_ppa_shrunk_diff",
    "margin_shrunk_diff",
    "prior_fpi_diff",
    "run_fit_diff",
    "off_rush_2ply_diff",
    "def_rush_2ply_diff",
    "coach_aggression_diff",
    "coach_rush_share_diff",
    "coach_tempo_diff",
    "qb_ppa_diff",
    "qb_dynamic_diff",
    "qb_punch_diff",
    "qb_ypa_diff",
    "qb_rush_share_diff",
    "injury_load_diff",
    "qb_injury_diff",
    "ol_injury_diff",
    "skill_injury_diff",
]

BLEND_FEATURE_COLS = FEATURE_COLS + ["open_spread"]
TARGET_COL = "home_margin"
SERVICE_ACADEMIES = {"Army", "Navy", "Air Force"}

STAT_FIELDS = [
    "margin",
    "points_for",
    "points_against",
    "off_ppa",
    "def_ppa",
    "off_passing_ppa",
    "off_rushing_ppa",
    "off_success",
    "def_success",
    "off_expl",
    "def_expl",
    "havoc",
    "opp_elo",
]


def build_features(
    conn=None,
    *,
    db_path: Path | None = None,
    fbs_only: bool = True,
) -> pd.DataFrame:
    own = conn is None
    conn = conn or store.init_schema(store.connect(db_path) if db_path else None)
    try:
        games = store.read_table(conn, "games")
        if games.empty:
            raise ValueError("games table is empty — run ingest or migrate first")
        tables = {
            name: store.read_table(conn, name)
            for name in [
                "lines",
                "ppa_games",
                "advanced_game_stats",
                "havoc_games",
                "team_game_stats",
                "talent",
                "recruiting_teams",
                "returning_production",
                "core_ratings",
                "sp_ratings",
                "wepa_season",
                "venues",
                "teams",
                "coaches_seasons",
                "pregame_wp",
                "weather",
                "fpi_ratings",
                "injury_reports",
                "player_season_stats",
                "player_ppa",
                "transfer_portal",
            ]
        }
    finally:
        if own:
            conn.close()

    games = _prepare_games(games, tables["teams"], fbs_only=fbs_only)
    games = _attach_lines(games, tables["lines"])
    games = _attach_pregame_wp(games, tables["pregame_wp"])
    games = _attach_weather(games, tables["weather"])
    games = _attach_venues(games, tables["venues"], tables["teams"])

    long = _team_game_panel(games, tables)
    rolling = _rolling_as_of(long)
    home = _prefix(rolling, "home").rename(columns={"team": "home_team", "game_id": "id"})
    away = _prefix(rolling, "away").rename(columns={"team": "away_team", "game_id": "id"})
    frame = games.merge(home, on=["id", "home_team"], how="left")
    frame = frame.merge(away, on=["id", "away_team"], how="left")

    frame = _attach_priors(frame, tables)
    frame = _context_features(frame)
    frame = _diffs_and_shrinkage(frame)
    frame = _attach_identity_layers(frame, games, tables)
    frame[TARGET_COL] = frame["home_points"] - frame["away_points"]
    frame["week_bucket"] = np.where(frame["week"].fillna(99) <= 4, "early", "midlate")
    frame["is_service_academy"] = (
        frame["home_team"].isin(SERVICE_ACADEMIES) | frame["away_team"].isin(SERVICE_ACADEMIES)
    ).astype(int)
    return frame


def available_features(frame: pd.DataFrame, blend: bool = False) -> list[str]:
    cols = BLEND_FEATURE_COLS if blend else FEATURE_COLS
    return [c for c in cols if c in frame.columns]


def _prepare_games(games: pd.DataFrame, teams: pd.DataFrame, fbs_only: bool) -> pd.DataFrame:
    frame = games.copy()
    frame["start_date"] = pd.to_datetime(frame["start_date"], utc=True, errors="coerce")
    frame["completed"] = frame["completed"].fillna(0).astype(int)
    for col in ("home_classification", "away_classification"):
        if col in frame:
            frame[col] = frame[col].astype(str).str.lower()
    if fbs_only:
        fbs_names = set()
        if not teams.empty and "school" in teams:
            fbs_names = set(teams["school"].dropna())
        home_fbs = frame.get("home_classification", pd.Series(index=frame.index)).eq("fbs") | frame["home_team"].isin(fbs_names)
        away_fbs = frame.get("away_classification", pd.Series(index=frame.index)).eq("fbs") | frame["away_team"].isin(fbs_names)
        if fbs_names or "home_classification" in frame:
            frame = frame[home_fbs & away_fbs].copy()
    frame = frame.dropna(subset=["home_team", "away_team", "start_date"])
    return frame.sort_values(["start_date", "id"]).reset_index(drop=True)


def _attach_lines(games: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    if lines.empty:
        games["close_spread"] = np.nan
        games["open_spread"] = np.nan
        games["close_total"] = np.nan
        return games
    grouped = lines.groupby("game_id").agg(
        close_spread=("spread", "median"),
        open_spread=("spread_open", "median"),
        close_total=("over_under", "median"),
    )
    return games.merge(grouped, left_on="id", right_index=True, how="left")


def _attach_pregame_wp(games: pd.DataFrame, wp: pd.DataFrame) -> pd.DataFrame:
    if wp.empty:
        games["cfbd_home_wp"] = np.nan
        games["cfbd_pregame_spread"] = np.nan
        return games
    slim = wp.rename(columns={"home_win_probability": "cfbd_home_wp", "spread": "cfbd_pregame_spread"})
    keep = ["game_id", "cfbd_home_wp", "cfbd_pregame_spread"]
    slim = slim[keep].drop_duplicates("game_id")
    return games.merge(slim, left_on="id", right_on="game_id", how="left").drop(columns=["game_id"], errors="ignore")


def _attach_weather(games: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    if weather.empty:
        for col in ("temperature", "precipitation", "wind_speed"):
            games[col] = np.nan
        games["indoor"] = 0
        return games
    return games.merge(weather, left_on="id", right_on="game_id", how="left").drop(columns=["game_id"], errors="ignore")


def _attach_venues(games: pd.DataFrame, venues: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    venue_cols = ["id", "latitude", "longitude", "elevation", "capacity", "grass", "dome"]
    if not venues.empty:
        v = venues[[c for c in venue_cols if c in venues.columns]].rename(
            columns={
                "id": "venue_id",
                "latitude": "venue_lat",
                "longitude": "venue_lon",
                "elevation": "venue_elevation",
                "capacity": "capacity",
                "grass": "grass",
                "dome": "dome",
            }
        )
        games = games.merge(v, on="venue_id", how="left")
    if not teams.empty:
        home_geo = teams.rename(
            columns={
                "school": "home_team",
                "latitude": "home_lat",
                "longitude": "home_lon",
                "elevation": "home_elevation",
            }
        )
        away_geo = teams.rename(
            columns={
                "school": "away_team",
                "latitude": "away_lat",
                "longitude": "away_lon",
                "elevation": "away_elevation",
            }
        )
        keep_h = [c for c in ("home_team", "home_lat", "home_lon", "home_elevation") if c in home_geo]
        keep_a = [c for c in ("away_team", "away_lat", "away_lon", "away_elevation") if c in away_geo]
        games = games.merge(home_geo[keep_h].drop_duplicates("home_team"), on="home_team", how="left")
        games = games.merge(away_geo[keep_a].drop_duplicates("away_team"), on="away_team", how="left")
        if "grass" not in games.columns and "grass" in teams.columns:
            pass
        if "capacity" not in games.columns and "capacity" in teams.columns:
            cap = teams[["school", "capacity", "grass", "dome"]].rename(columns={"school": "home_team"})
            games = games.merge(cap.drop_duplicates("home_team"), on="home_team", how="left")
    for col in ("grass", "dome", "capacity", "venue_lat", "venue_lon", "venue_elevation"):
        if col not in games.columns:
            games[col] = np.nan
    return games


def _team_game_panel(games: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "game_id": games["id"],
            "season": games["season"],
            "week": games["week"],
            "start_date": games["start_date"],
            "team": games["home_team"],
            "opponent": games["away_team"],
            "is_home": 1,
            "points_for": games["home_points"],
            "points_against": games["away_points"],
            "elo": games["home_pregame_elo"],
            "opp_elo": games["away_pregame_elo"],
        }
    )
    away = pd.DataFrame(
        {
            "game_id": games["id"],
            "season": games["season"],
            "week": games["week"],
            "start_date": games["start_date"],
            "team": games["away_team"],
            "opponent": games["home_team"],
            "is_home": 0,
            "points_for": games["away_points"],
            "points_against": games["home_points"],
            "elo": games["away_pregame_elo"],
            "opp_elo": games["home_pregame_elo"],
        }
    )
    panel = pd.concat([home, away], ignore_index=True)
    panel["margin"] = panel["points_for"] - panel["points_against"]

    ppa = tables["ppa_games"]
    if not ppa.empty:
        ppa_keep = ppa.rename(
            columns={
                "off_overall": "off_ppa",
                "def_overall": "def_ppa",
                "off_passing": "off_passing_ppa",
                "off_rushing": "off_rushing_ppa",
            }
        )
        cols = ["game_id", "team", "off_ppa", "def_ppa", "off_passing_ppa", "off_rushing_ppa"]
        panel = panel.merge(ppa_keep[cols], on=["game_id", "team"], how="left")
    adv = tables["advanced_game_stats"]
    if not adv.empty:
        adv_keep = adv.rename(
            columns={
                "off_success_rate": "off_success",
                "def_success_rate": "def_success",
                "off_explosiveness": "off_expl",
                "def_explosiveness": "def_expl",
            }
        )
        cols = ["game_id", "team", "off_success", "def_success", "off_expl", "def_expl"]
        present = [c for c in cols if c in adv_keep.columns]
        panel = panel.merge(adv_keep[present], on=["game_id", "team"], how="left")
    havoc = tables["havoc_games"]
    if not havoc.empty:
        h = havoc.rename(columns={"off_havoc": "havoc"})[["game_id", "team", "havoc"]]
        panel = panel.merge(h, on=["game_id", "team"], how="left")
    for field in STAT_FIELDS:
        if field not in panel.columns:
            panel[field] = np.nan
    return panel.sort_values(["team", "start_date", "game_id"])


def _rolling_as_of(panel: pd.DataFrame) -> pd.DataFrame:
    """Shift by one game so kickoff features never include the current game."""
    frame = panel.copy()
    grouped = frame.groupby(["team", "season"], group_keys=False)
    frame["games_played"] = grouped.cumcount()
    for field in STAT_FIELDS:
        frame[f"{field}_raw"] = grouped[field].shift(1)
        frame[f"{field}_std"] = grouped[field].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )
        frame[f"{field}_l4"] = grouped[field].transform(
            lambda s: s.shift(1).rolling(4, min_periods=1).mean()
        )
    prev_date = frame.groupby("team")["start_date"].shift(1)
    frame["rest_days"] = (frame["start_date"] - prev_date).dt.total_seconds() / 86400.0

    season_summary = (
        frame.groupby(["team", "season"])
        .agg(prior_margin=("margin", "mean"), prior_off_ppa=("off_ppa", "mean"), prior_def_ppa=("def_ppa", "mean"))
        .reset_index()
    )
    season_summary["season"] = season_summary["season"] + 1
    frame = frame.merge(season_summary, on=["team", "season"], how="left")
    return frame


def _prefix(rolling: pd.DataFrame, side: str) -> pd.DataFrame:
    keep = ["game_id", "team", "games_played", "rest_days", "prior_margin", "prior_off_ppa", "prior_def_ppa"]
    keep += [c for c in rolling.columns if c.endswith("_std") or c.endswith("_l4")]
    slim = rolling[keep].copy()
    rename = {c: f"{side}_{c}" for c in slim.columns if c not in {"game_id", "team"}}
    return slim.rename(columns=rename)


def _attach_priors(frame: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    talent = tables["talent"]
    if not talent.empty:
        t = talent.rename(columns={"year": "season", "team": "home_team", "talent": "home_talent"})
        frame = frame.merge(t, on=["season", "home_team"], how="left")
        t = talent.rename(columns={"year": "season", "team": "away_team", "talent": "away_talent"})
        frame = frame.merge(t, on=["season", "away_team"], how="left")
    else:
        frame["home_talent"] = np.nan
        frame["away_talent"] = np.nan

    rec = tables["recruiting_teams"]
    if not rec.empty:
        rec = rec.copy()
        rec["points"] = rec["points"].astype(float)
        rolling = (
            rec.sort_values(["team", "year"])
            .groupby("team")["points"]
            .transform(lambda s: s.rolling(3, min_periods=1).mean())
        )
        rec["recruiting_3yr"] = rolling
        rh = rec.rename(columns={"year": "season", "team": "home_team", "recruiting_3yr": "home_recruiting_3yr"})
        frame = frame.merge(rh[["season", "home_team", "home_recruiting_3yr"]], on=["season", "home_team"], how="left")
        ra = rec.rename(columns={"year": "season", "team": "away_team", "recruiting_3yr": "away_recruiting_3yr"})
        frame = frame.merge(ra[["season", "away_team", "away_recruiting_3yr"]], on=["season", "away_team"], how="left")
    else:
        frame["home_recruiting_3yr"] = np.nan
        frame["away_recruiting_3yr"] = np.nan

    ret = tables["returning_production"]
    if not ret.empty:
        rh = ret.rename(columns={"team": "home_team", "percent_ppa": "home_returning"})
        frame = frame.merge(rh[["season", "home_team", "home_returning"]], on=["season", "home_team"], how="left")
        ra = ret.rename(columns={"team": "away_team", "percent_ppa": "away_returning"})
        frame = frame.merge(ra[["season", "away_team", "away_returning"]], on=["season", "away_team"], how="left")
    else:
        frame["home_returning"] = np.nan
        frame["away_returning"] = np.nan

    # Prior-season only ratings (end-of-season snapshots leak if used in-year).
    for table_name, value_col, home_name, away_name in [
        ("sp_ratings", "rating", "home_prior_sp", "away_prior_sp"),
        ("core_ratings", "overall", "home_prior_core", "away_prior_core"),
        ("wepa_season", "epa_total", "home_prior_wepa", "away_prior_wepa"),
        ("fpi_ratings", "fpi", "home_prior_fpi", "away_prior_fpi"),
    ]:
        table = tables[table_name]
        frame[home_name] = np.nan
        frame[away_name] = np.nan
        if table.empty or value_col not in table.columns:
            continue
        year_col = "year" if "year" in table.columns else "season"
        prior = table[[year_col, "team", value_col]].copy()
        prior[year_col] = prior[year_col] + 1
        prior = prior.rename(columns={year_col: "season", value_col: "val"})
        if table_name in {"core_ratings", "fpi_ratings"}:
            prior = prior.drop_duplicates(["season", "team"], keep="last")
        hh = prior.rename(columns={"team": "home_team", "val": home_name})
        frame = frame.drop(columns=[home_name]).merge(hh, on=["season", "home_team"], how="left")
        aa = prior.rename(columns={"team": "away_team", "val": away_name})
        frame = frame.drop(columns=[away_name]).merge(aa, on=["season", "away_team"], how="left")
    return frame


def _context_features(frame: pd.DataFrame) -> pd.DataFrame:
    venue_lat = frame["venue_lat"] if "venue_lat" in frame else frame.get("home_lat")
    venue_lon = frame["venue_lon"] if "venue_lon" in frame else frame.get("home_lon")
    frame["travel_miles_away"] = [
        haversine_miles(alat, alon, vlat, vlon)
        for alat, alon, vlat, vlon in zip(
            frame.get("away_lat", pd.Series(np.nan, index=frame.index)),
            frame.get("away_lon", pd.Series(np.nan, index=frame.index)),
            venue_lat if venue_lat is not None else pd.Series(np.nan, index=frame.index),
            venue_lon if venue_lon is not None else pd.Series(np.nan, index=frame.index),
        )
    ]
    home_travel = []
    for ns, hlat, hlon, vlat, vlon in zip(
        frame.get("neutral_site", pd.Series(0, index=frame.index)).fillna(0),
        frame.get("home_lat", pd.Series(np.nan, index=frame.index)),
        frame.get("home_lon", pd.Series(np.nan, index=frame.index)),
        venue_lat if venue_lat is not None else pd.Series(np.nan, index=frame.index),
        venue_lon if venue_lon is not None else pd.Series(np.nan, index=frame.index),
    ):
        if ns:
            home_travel.append(haversine_miles(hlat, hlon, vlat, vlon))
        else:
            home_travel.append(0.0)
    frame["travel_miles_home"] = home_travel
    home_el = frame.get("home_elevation", pd.Series(np.nan, index=frame.index))
    venue_el = frame.get("venue_elevation", home_el)
    away_el = frame.get("away_elevation", pd.Series(np.nan, index=frame.index))
    frame["elevation_diff"] = pd.to_numeric(venue_el, errors="coerce") - pd.to_numeric(away_el, errors="coerce")
    if "dome" in frame:
        indoor = frame.get("indoor", pd.Series(0, index=frame.index)).fillna(0)
        frame.loc[(frame["dome"] == 1) | (indoor == 1), ["precipitation", "wind_speed"]] = 0
    cap = pd.to_numeric(frame.get("capacity"), errors="coerce")
    try:
        frame["noise_bucket"] = pd.qcut(cap.rank(method="first"), 3, labels=[1, 2, 3]).astype(float)
    except (ValueError, TypeError):
        frame["noise_bucket"] = np.nan
    frame["grass"] = pd.to_numeric(frame.get("grass"), errors="coerce").fillna(0)
    frame["dome"] = pd.to_numeric(frame.get("dome"), errors="coerce").fillna(0)
    frame["neutral_site"] = pd.to_numeric(frame.get("neutral_site"), errors="coerce").fillna(0)
    frame["conference_game"] = pd.to_numeric(frame.get("conference_game"), errors="coerce").fillna(0)
    return frame


def _diffs_and_shrinkage(frame: pd.DataFrame) -> pd.DataFrame:
    frame["elo_diff"] = pd.to_numeric(frame.get("home_pregame_elo"), errors="coerce") - pd.to_numeric(
        frame.get("away_pregame_elo"), errors="coerce"
    )
    frame["talent_diff"] = frame.get("home_talent") - frame.get("away_talent")
    frame["recruiting_3yr_diff"] = frame.get("home_recruiting_3yr") - frame.get("away_recruiting_3yr")
    frame["returning_diff"] = frame.get("home_returning") - frame.get("away_returning")
    frame["prior_sp_diff"] = frame.get("home_prior_sp") - frame.get("away_prior_sp")
    frame["prior_core_diff"] = frame.get("home_prior_core") - frame.get("away_prior_core")
    frame["prior_wepa_diff"] = frame.get("home_prior_wepa") - frame.get("away_prior_wepa")
    frame["prior_fpi_diff"] = frame.get("home_prior_fpi") - frame.get("away_prior_fpi")
    frame["games_played_home"] = frame.get("home_games_played")
    frame["games_played_away"] = frame.get("away_games_played")
    frame["games_played_diff"] = frame["games_played_home"] - frame["games_played_away"]
    frame["rest_days_home"] = frame.get("home_rest_days")
    frame["rest_days_away"] = frame.get("away_rest_days")
    frame["rest_diff"] = frame["rest_days_home"] - frame["rest_days_away"]

    pairs = [
        ("margin_std", "margin_std_diff"),
        ("margin_l4", "margin_l4_diff"),
        ("points_for_std", "pf_std_diff"),
        ("points_against_std", "pa_std_diff"),
        ("prior_margin", "prior_margin_diff"),
        ("off_ppa_std", "off_ppa_std_diff"),
        ("def_ppa_std", "def_ppa_std_diff"),
        ("off_success_std", "off_success_std_diff"),
        ("def_success_std", "def_success_std_diff"),
        ("off_expl_std", "off_expl_std_diff"),
        ("def_expl_std", "def_expl_std_diff"),
        ("havoc_std", "havoc_std_diff"),
    ]
    for stem, out in pairs:
        home = frame.get(f"home_{stem}")
        away = frame.get(f"away_{stem}")
        if home is None or away is None:
            frame[out] = np.nan
        else:
            frame[out] = pd.to_numeric(home, errors="coerce") - pd.to_numeric(away, errors="coerce")

    k = config.SHRINKAGE_K
    frame["off_ppa_shrunk_diff"] = [
        (shrink(h, hp, n_h or 0, k) or 0) - (shrink(a, ap, n_a or 0, k) or 0)
        if any(v is not None and not (isinstance(v, float) and np.isnan(v)) for v in (h, a, hp, ap))
        else np.nan
        for h, a, hp, ap, n_h, n_a in zip(
            frame.get("home_off_ppa_std", pd.Series(np.nan, index=frame.index)),
            frame.get("away_off_ppa_std", pd.Series(np.nan, index=frame.index)),
            frame.get("home_prior_off_ppa", pd.Series(np.nan, index=frame.index)),
            frame.get("away_prior_off_ppa", pd.Series(np.nan, index=frame.index)),
            frame.get("home_games_played", pd.Series(0, index=frame.index)),
            frame.get("away_games_played", pd.Series(0, index=frame.index)),
        )
    ]
    frame["def_ppa_shrunk_diff"] = [
        (shrink(h, hp, n_h or 0, k) or 0) - (shrink(a, ap, n_a or 0, k) or 0)
        if any(pd.notna(v) for v in (h, a, hp, ap))
        else np.nan
        for h, a, hp, ap, n_h, n_a in zip(
            frame.get("home_def_ppa_std", pd.Series(np.nan, index=frame.index)),
            frame.get("away_def_ppa_std", pd.Series(np.nan, index=frame.index)),
            frame.get("home_prior_def_ppa", pd.Series(np.nan, index=frame.index)),
            frame.get("away_prior_def_ppa", pd.Series(np.nan, index=frame.index)),
            frame.get("home_games_played", pd.Series(0, index=frame.index)),
            frame.get("away_games_played", pd.Series(0, index=frame.index)),
        )
    ]
    frame["margin_shrunk_diff"] = [
        (shrink(h, hp, n_h or 0, k) or 0) - (shrink(a, ap, n_a or 0, k) or 0)
        if any(pd.notna(v) for v in (h, a, hp, ap))
        else np.nan
        for h, a, hp, ap, n_h, n_a in zip(
            frame.get("home_margin_std", pd.Series(np.nan, index=frame.index)),
            frame.get("away_margin_std", pd.Series(np.nan, index=frame.index)),
            frame.get("home_prior_margin", pd.Series(np.nan, index=frame.index)),
            frame.get("away_prior_margin", pd.Series(np.nan, index=frame.index)),
            frame.get("home_games_played", pd.Series(0, index=frame.index)),
            frame.get("away_games_played", pd.Series(0, index=frame.index)),
        )
    ]
    return frame


def _attach_identity_layers(frame: pd.DataFrame, games: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Coach, scheme, QB (who starts), and as-of injuries. All leakage-safe."""
    try:
        from cfb_model.identity import attach_scheme_features

        frame = attach_scheme_features(frame, games, tables.get("team_game_stats"))
    except Exception:
        for col in ("run_fit_diff", "off_rush_2ply_diff", "def_rush_2ply_diff"):
            if col not in frame.columns:
                frame[col] = np.nan
    try:
        from cfb_model.coaches import attach_coaches, build_coach_season_table

        profiles = build_coach_season_table(
            tables.get("coaches_seasons", pd.DataFrame()),
            games,
            tables.get("team_game_stats", pd.DataFrame()),
            tables.get("advanced_game_stats"),
            tables.get("ppa_games"),
            tables.get("teams"),
        )
        frame = attach_coaches(frame, profiles)
    except Exception:
        pass
    try:
        from cfb_model.players import attach_quarterbacks, build_qb_season_table

        qb_table = build_qb_season_table(
            tables.get("player_season_stats", pd.DataFrame()),
            tables.get("player_ppa"),
            tables.get("transfer_portal"),
            tables.get("talent"),
        )
        live = None
        if not tables.get("player_season_stats", pd.DataFrame()).empty:
            stats = tables["player_season_stats"]
            if "season" in stats.columns:
                current = stats[stats["season"] == config.CURRENT_SEASON]
                if not current.empty and "completed" in frame.columns and frame["completed"].fillna(0).astype(int).eq(0).any():
                    live = current
        injuries = tables.get("injury_reports", pd.DataFrame())
        frame = attach_quarterbacks(frame, qb_table, injuries=injuries, live_stats=live)
    except Exception:
        for col in ("qb_ppa_diff", "qb_dynamic_diff", "qb_punch_diff", "qb_out_points_diff"):
            if col not in frame.columns:
                frame[col] = np.nan
    try:
        from cfb_model.injuries import attach_injuries

        frame = attach_injuries(frame, tables.get("injury_reports", pd.DataFrame()))
    except Exception:
        if "injury_load_diff" not in frame.columns:
            frame["injury_load_diff"] = np.nan
    return frame
