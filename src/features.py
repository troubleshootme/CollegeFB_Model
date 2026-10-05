from __future__ import annotations

import re
import sqlite3
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.config import DB_PATH

STAT_NUM_COLS = [
    "points",
    "totalYards",
    "rushingYards",
    "netPassingYards",
    "rushingAttempts",
    "firstDowns",
    "turnovers",
    "sacks",
    "tacklesForLoss",
    "yardsPerRushAttempt",
    "yardsPerPass",
]

FEATURE_COLS = [
    "week",
    "conference_game",
    "neutral_site",
    "home_pregame_elo",
    "away_pregame_elo",
    "elo_diff",
    "home_talent",
    "away_talent",
    "talent_diff",
    "home_sp",
    "away_sp",
    "sp_diff",
    "home_sp_off",
    "away_sp_off",
    "home_sp_def",
    "away_sp_def",
    "home_recruit",
    "away_recruit",
    "recruit_diff",
    "home_returning_ppa",
    "away_returning_ppa",
    "home_returning_usage",
    "away_returning_usage",
    "distance_miles",
    "elevation_diff",
    "home_capacity",
    "home_dome",
    "home_grass",
    "capacity_bucket",
    "home_rest_days",
    "away_rest_days",
    "home_points_l4",
    "away_points_l4",
    "home_points_allowed_l4",
    "away_points_allowed_l4",
    "home_yards_l4",
    "away_yards_l4",
    "home_yards_allowed_l4",
    "away_yards_allowed_l4",
    "home_to_margin_l4",
    "away_to_margin_l4",
    "home_third_down_l4",
    "away_third_down_l4",
    "home_ypp_l4",
    "away_ypp_l4",
    "home_points_prior",
    "away_points_prior",
    "home_points_allowed_prior",
    "away_points_allowed_prior",
    "home_yards_prior",
    "away_yards_prior",
    "home_yards_allowed_prior",
    "away_yards_allowed_prior",
    "home_success_prior",
    "away_success_prior",
    "home_def_success_prior",
    "away_def_success_prior",
    "wx_temp_max",
    "wx_temp_min",
]

MARKET_COLS = ["spread", "over_under"]


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _parse_efficiency(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value)
    if "-" not in text:
        return np.nan
    made, att = text.split("-", 1)
    try:
        attempts = float(att)
        return float(made) / attempts if attempts else np.nan
    except ValueError:
        return np.nan


def _parse_clock(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value)
    if ":" not in text:
        return pd.to_numeric(text, errors="coerce")
    try:
        minutes, seconds = text.split(":", 1)
        return float(minutes) * 60 + float(seconds)
    except ValueError:
        return np.nan


def _season_fbs_schools(teams: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Trusted (season, school) FBS pairs.

    CFBD can list FCS programs on a future roster before any games are played
    (2026: North Dakota State, Sacramento State). If a season has no completed
    games, keep only schools that already appeared in an earlier roster.
    """
    completed_seasons = set(
        pd.to_numeric(games.loc[games["completed"].astype(bool), "season"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    rows: list[dict] = []
    seen: set[str] = set()
    years = sorted(pd.to_numeric(teams["year"], errors="coerce").dropna().astype(int).unique())
    for year in years:
        raw = set(teams.loc[teams["year"] == year, "school"].dropna().astype(str))
        trusted = raw & seen if year not in completed_seasons and seen else raw
        if year in completed_seasons:
            seen |= raw
        else:
            seen |= trusted
        rows.extend({"season": year, "school": school} for school in sorted(trusted))
    return pd.DataFrame(rows, columns=["season", "school"])


def season_ending_elo(games: pd.DataFrame) -> pd.DataFrame:
    """Last completed postgame Elo for each team-season, shifted to the next season."""
    home = games[["start_date", "season", "home_team", "home_postgame_elo"]].rename(
        columns={"home_team": "team", "home_postgame_elo": "elo"}
    )
    away = games[["start_date", "season", "away_team", "away_postgame_elo"]].rename(
        columns={"away_team": "team", "away_postgame_elo": "elo"}
    )
    long = pd.concat([home, away], ignore_index=True).dropna(subset=["elo"])
    long["start_date"] = pd.to_datetime(long["start_date"], utc=True, errors="coerce")
    last = long.sort_values("start_date").groupby(["team", "season"], as_index=False).tail(1)
    last = last.rename(columns={"elo": "carry_elo"})
    last["season"] = last["season"] + 1
    return last[["team", "season", "carry_elo"]]


def haversine_miles(lat1, lon1, lat2, lon2) -> pd.Series:
    radius = 3958.8
    lat1 = np.radians(pd.to_numeric(lat1, errors="coerce"))
    lon1 = np.radians(pd.to_numeric(lon1, errors="coerce"))
    lat2 = np.radians(pd.to_numeric(lat2, errors="coerce"))
    lon2 = np.radians(pd.to_numeric(lon2, errors="coerce"))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def _parse_efficiency_series(series: pd.Series) -> pd.Series:
    parts = series.astype(str).str.split("-", n=1, expand=True)
    if parts.shape[1] < 2:
        return pd.Series(np.nan, index=series.index)
    made = pd.to_numeric(parts[0], errors="coerce")
    att = pd.to_numeric(parts[1], errors="coerce").replace(0, np.nan)
    return made / att


def _home_spreads(frame: pd.DataFrame) -> pd.Series:
    spread = pd.to_numeric(frame["spread"], errors="coerce")
    formatted = frame["formatted_spread"].fillna("").astype(str)
    parts = formatted.str.rsplit(" ", n=1, expand=True)
    if parts.shape[1] < 2:
        return spread
    favorite = parts[0]
    mag = pd.to_numeric(parts[1], errors="coerce")
    home = frame["home_team"].astype(str)
    away = frame["away_team"].astype(str)
    result = spread.copy()
    result = np.where(favorite.eq(home) & mag.notna(), mag, result)
    result = np.where(favorite.eq(away) & mag.notna(), -mag, result)
    return pd.Series(result, index=frame.index)


_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_table(name: str, con: sqlite3.Connection) -> pd.DataFrame:
    if not _TABLE_NAME.fullmatch(name):
        raise ValueError(f"invalid table name {name!r}")
    return pd.read_sql(f'SELECT * FROM "{name}"', con)


def consensus_lines(lines: pd.DataFrame) -> pd.DataFrame:
    if lines.empty:
        return pd.DataFrame(columns=["game_id", "spread", "over_under", "spread_open", "over_under_open"])
    frame = lines.copy()
    frame["provider_key"] = (
        frame["provider"].fillna("").astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    )
    frame = frame.drop_duplicates(["game_id", "provider_key"], keep="first")
    frame["home_spread"] = _home_spreads(frame)
    frame["is_consensus"] = frame["provider_key"].eq("consensus")
    has_consensus = frame.groupby("game_id")["is_consensus"].transform("any")
    rest = frame.loc[(has_consensus & frame["is_consensus"]) | ~has_consensus]
    grouped = (
        rest.groupby("game_id")
        .agg(
            spread=("home_spread", "median"),
            over_under=("over_under", "median"),
            spread_open=("spread_open", "median"),
            over_under_open=("over_under_open", "median"),
            n_books=("provider_key", "nunique"),
        )
        .reset_index()
    )
    return grouped


def prepare_team_games(stats: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    frame = stats.drop_duplicates(["game_id", "team_name"]).copy()
    if "home_away" in frame.columns:
        frame["home_away"] = frame["home_away"].astype(str).str.lower()
    for col in STAT_NUM_COLS:
        if col in frame.columns:
            frame[col] = _to_float(frame[col])
        else:
            frame[col] = np.nan
    frame["third_down_rate"] = (
        _parse_efficiency_series(frame["thirdDownEff"]) if "thirdDownEff" in frame.columns else np.nan
    )
    frame["fourth_down_rate"] = (
        _parse_efficiency_series(frame["fourthDownEff"]) if "fourthDownEff" in frame.columns else np.nan
    )
    frame["possession_seconds"] = (
        frame["possessionTime"].map(_parse_clock) if "possessionTime" in frame.columns else np.nan
    )
    rush_att = _to_float(frame["rushingAttempts"]) if "rushingAttempts" in frame.columns else 0
    if "completionAttempts" in frame.columns:
        parts = frame["completionAttempts"].astype(str).str.split("-", n=1, expand=True)
        pass_att = pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else np.nan
    else:
        pass_att = np.nan
    plays = pd.to_numeric(rush_att, errors="coerce").fillna(0) + pd.to_numeric(pass_att, errors="coerce").fillna(0)
    frame["ypp"] = frame["totalYards"] / plays.replace(0, np.nan)

    existing_ids = set(frame["game_id"].dropna().unique())
    missing = games[~games["game_id"].isin(existing_ids)].copy()
    if not missing.empty:
        stubs = []
        for side, team_col in (("home", "home_team"), ("away", "away_team")):
            part = missing[["game_id", "season", team_col]].rename(columns={team_col: "team_name"})
            part["home_away"] = side
            stubs.append(part)
        stubs = pd.concat(stubs, ignore_index=True)
        for col in list(STAT_NUM_COLS) + ["thirdDownEff", "fourthDownEff", "possessionTime", "completionAttempts"]:
            if col not in stubs.columns:
                stubs[col] = np.nan
        stubs["third_down_rate"] = np.nan
        stubs["fourth_down_rate"] = np.nan
        stubs["possession_seconds"] = np.nan
        stubs["ypp"] = np.nan
        frame = pd.concat([frame, stubs], ignore_index=True, sort=False)

    meta = games[
        ["game_id", "start_date", "home_team", "away_team", "home_points", "away_points"]
    ].drop_duplicates("game_id")
    frame = frame.merge(meta, on="game_id", how="left")
    opp = frame[["game_id", "team_name", "points", "totalYards", "turnovers"]].rename(
        columns={
            "team_name": "opp_team",
            "points": "points_allowed",
            "totalYards": "yards_allowed",
            "turnovers": "takeaways",
        }
    )
    paired = frame.merge(opp, on="game_id", how="left")
    paired = paired[paired["team_name"] != paired["opp_team"]].copy()
    paired["to_margin"] = paired["takeaways"] - paired["turnovers"]
    paired["start_date"] = pd.to_datetime(paired["start_date"], utc=True, errors="coerce")
    paired = paired.sort_values(["team_name", "start_date"]).reset_index(drop=True)

    rolling_src = {
        "points": "points",
        "points_allowed": "points_allowed",
        "yards": "totalYards",
        "yards_allowed": "yards_allowed",
        "to_margin": "to_margin",
        "third_down": "third_down_rate",
        "ypp": "ypp",
    }
    by_team = paired.groupby("team_name", sort=False)
    for name, col in rolling_src.items():
        paired[f"{name}_l4"] = by_team[col].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
        paired[f"{name}_season"] = paired.groupby(["team_name", "season"], sort=False)[col].transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )

    paired["prev_start"] = by_team["start_date"].shift(1)
    paired["rest_days"] = (paired["start_date"] - paired["prev_start"]).dt.total_seconds() / 86400.0

    prior = (
        paired.groupby(["team_name", "season"], as_index=False)
        .agg(
            points_prior=("points", "mean"),
            points_allowed_prior=("points_allowed", "mean"),
            yards_prior=("totalYards", "mean"),
            yards_allowed_prior=("yards_allowed", "mean"),
            to_margin_prior=("to_margin", "mean"),
            third_down_prior=("third_down_rate", "mean"),
        )
    )
    prior["season"] = prior["season"] + 1
    paired = paired.merge(prior, on=["team_name", "season"], how="left")
    return paired


def _prefix_team_features(team_games: pd.DataFrame, side: str) -> pd.DataFrame:
    cols = [
        "game_id",
        "team_name",
        "rest_days",
        "points_l4",
        "points_allowed_l4",
        "yards_l4",
        "yards_allowed_l4",
        "to_margin_l4",
        "third_down_l4",
        "ypp_l4",
        "points_prior",
        "points_allowed_prior",
        "yards_prior",
        "yards_allowed_prior",
    ]
    frame = team_games[cols].copy()
    rename = {col: f"{side}_{col}" for col in cols if col != "game_id"}
    rename["team_name"] = f"{side}_team_stats"
    rename["rest_days"] = f"{side}_rest_days"
    return frame.rename(columns=rename)


def build_feature_frame(con: sqlite3.Connection | None = None) -> pd.DataFrame:
    close = False
    if con is None:
        con = sqlite3.connect(DB_PATH)
        close = True
    weather = pd.DataFrame()
    venues = pd.DataFrame()
    try:
        games = load_table("games", con)
        stats = load_table("team_game_stats", con)
        lines = load_table("betting_lines", con)
        teams = load_table("fbs_teams", con)
        talent = load_table("talent", con)
        sp_ratings = load_table("sp_ratings", con)
        recruiting = load_table("recruiting", con)
        returning = load_table("returning_production", con)
        advanced = load_table("advanced_season_stats", con)
        try:
            weather = load_table("weather_daily", con)
        except sqlite3.Error:
            weather = pd.DataFrame()
        try:
            venues = load_table("venues", con)
        except sqlite3.Error:
            venues = pd.DataFrame()
    finally:
        if close:
            con.close()

    games = games.drop_duplicates("game_id")
    talent = talent.drop_duplicates(["year", "team"])
    sp_ratings = sp_ratings.drop_duplicates(["year", "team"])
    recruiting = recruiting.drop_duplicates(["year", "team"])
    returning = returning.drop_duplicates(["year", "team"])
    advanced = advanced.drop_duplicates(["year", "team"])
    teams = teams.drop_duplicates(["year", "school"])

    print("Building feature frame...")
    games = games.copy()
    games["start_date"] = pd.to_datetime(games["start_date"], utc=True, errors="coerce")
    games["margin"] = games["home_points"] - games["away_points"]
    games["total_points"] = games["home_points"] + games["away_points"]
    games["home_win"] = (games["home_points"] > games["away_points"]).astype("float")
    roster = _season_fbs_schools(teams, games)
    home_fbs = games.merge(
        roster.rename(columns={"school": "home_team"}).assign(_home_fbs=True),
        on=["season", "home_team"],
        how="left",
    )["_home_fbs"].fillna(False).astype(bool)
    away_fbs = games.merge(
        roster.rename(columns={"school": "away_team"}).assign(_away_fbs=True),
        on=["season", "away_team"],
        how="left",
    )["_away_fbs"].fillna(False).astype(bool)
    games["fbs_vs_fbs"] = (
        games["home_classification"].str.lower().eq("fbs")
        & games["away_classification"].str.lower().eq("fbs")
        & home_fbs.to_numpy()
        & away_fbs.to_numpy()
    )

    team_games = prepare_team_games(stats, games)
    home_feats = _prefix_team_features(team_games[team_games["home_away"] == "home"], "home").drop_duplicates("game_id")
    away_feats = _prefix_team_features(team_games[team_games["home_away"] == "away"], "away").drop_duplicates("game_id")
    frame = games.merge(home_feats, on="game_id", how="left").merge(away_feats, on="game_id", how="left")

    teams_year = teams.rename(
        columns={
            "school": "team",
            "latitude": "lat",
            "longitude": "lon",
            "elevation": "elev",
            "capacity": "capacity",
            "grass": "grass",
            "dome": "dome",
            "year": "season",
        }
    )
    home_teams = teams_year.add_prefix("home_").rename(columns={"home_season": "season", "home_team": "home_team"})
    away_teams = teams_year.add_prefix("away_").rename(columns={"away_season": "season", "away_team": "away_team"})
    frame = frame.merge(
        home_teams[
            [
                "season",
                "home_team",
                "home_lat",
                "home_lon",
                "home_elev",
                "home_capacity",
                "home_grass",
                "home_dome",
            ]
        ],
        on=["season", "home_team"],
        how="left",
    )
    frame = frame.merge(
        away_teams[["season", "away_team", "away_lat", "away_lon", "away_elev"]],
        on=["season", "away_team"],
        how="left",
    )
    if not venues.empty and "venue_id" in frame.columns:
        v = venues.drop_duplicates("venue_id").rename(
            columns={"latitude": "venue_lat", "longitude": "venue_lon", "dome": "venue_dome"}
        )
        keep = [col for col in ["venue_id", "venue_lat", "venue_lon", "venue_dome"] if col in v.columns]
        frame = frame.merge(v[keep], on="venue_id", how="left")
    dest_lat = frame["venue_lat"] if "venue_lat" in frame.columns else frame["home_lat"]
    dest_lon = frame["venue_lon"] if "venue_lon" in frame.columns else frame["home_lon"]
    if "venue_lat" in frame.columns:
        dest_lat = dest_lat.fillna(frame["home_lat"])
        dest_lon = dest_lon.fillna(frame["home_lon"])
    frame["distance_miles"] = haversine_miles(frame["away_lat"], frame["away_lon"], dest_lat, dest_lon)
    frame["elevation_diff"] = _to_float(frame["home_elev"]) - _to_float(frame["away_elev"])
    frame["home_capacity"] = _to_float(frame["home_capacity"])
    venue_dome = _to_float(frame["venue_dome"]) if "venue_dome" in frame.columns else np.nan
    frame["home_dome"] = venue_dome.fillna(_to_float(frame["home_dome"])).fillna(0)
    frame["home_grass"] = _to_float(frame["home_grass"]).fillna(0)
    frame["capacity_bucket"] = pd.cut(
        frame["home_capacity"],
        bins=[-np.inf, 30000, 50000, 80000, np.inf],
        labels=[0, 1, 2, 3],
    ).astype("float")

    talent = talent.rename(columns={"year": "season", "team": "talent_team", "talent": "talent"})
    talent_prior = talent.copy()
    talent_prior["season"] = talent_prior["season"] + 1
    frame = frame.merge(
        talent.rename(columns={"talent_team": "home_team", "talent": "home_talent"}),
        on=["season", "home_team"],
        how="left",
    )
    frame = frame.merge(
        talent.rename(columns={"talent_team": "away_team", "talent": "away_talent"}),
        on=["season", "away_team"],
        how="left",
    )
    frame = frame.merge(
        talent_prior.rename(columns={"talent_team": "home_team", "talent": "home_talent_prior"})[
            ["season", "home_team", "home_talent_prior"]
        ],
        on=["season", "home_team"],
        how="left",
    )
    frame = frame.merge(
        talent_prior.rename(columns={"talent_team": "away_team", "talent": "away_talent_prior"})[
            ["season", "away_team", "away_talent_prior"]
        ],
        on=["season", "away_team"],
        how="left",
    )
    frame["home_talent"] = _to_float(frame["home_talent"]).fillna(_to_float(frame["home_talent_prior"]))
    frame["away_talent"] = _to_float(frame["away_talent"]).fillna(_to_float(frame["away_talent_prior"]))
    frame["talent_diff"] = frame["home_talent"] - frame["away_talent"]

    # Prior-season SP+ avoids using same-year ratings that include later weeks.
    sp_prior = sp_ratings.rename(columns={"year": "season", "team": "sp_team"})
    sp_prior["season"] = sp_prior["season"] + 1
    frame = frame.merge(
        sp_prior.rename(
            columns={
                "sp_team": "home_team",
                "sp_overall": "home_sp",
                "sp_offense": "home_sp_off",
                "sp_defense": "home_sp_def",
            }
        )[["season", "home_team", "home_sp", "home_sp_off", "home_sp_def"]],
        on=["season", "home_team"],
        how="left",
    )
    frame = frame.merge(
        sp_prior.rename(
            columns={
                "sp_team": "away_team",
                "sp_overall": "away_sp",
                "sp_offense": "away_sp_off",
                "sp_defense": "away_sp_def",
            }
        )[["season", "away_team", "away_sp", "away_sp_off", "away_sp_def"]],
        on=["season", "away_team"],
        how="left",
    )
    frame["sp_diff"] = frame["home_sp"] - frame["away_sp"]

    rec = recruiting.rename(columns={"year": "season", "team": "rec_team", "recruit_points": "recruit"})
    frame = frame.merge(
        rec.rename(columns={"rec_team": "home_team", "recruit": "home_recruit"})[["season", "home_team", "home_recruit"]],
        on=["season", "home_team"],
        how="left",
    )
    frame = frame.merge(
        rec.rename(columns={"rec_team": "away_team", "recruit": "away_recruit"})[["season", "away_team", "away_recruit"]],
        on=["season", "away_team"],
        how="left",
    )
    frame["recruit_diff"] = frame["home_recruit"] - frame["away_recruit"]

    ret = returning.rename(
        columns={
            "year": "season",
            "team": "ret_team",
            "returning_total_ppa": "returning_ppa",
            "returning_usage": "returning_usage",
        }
    )
    frame = frame.merge(
        ret.rename(columns={"ret_team": "home_team", "returning_ppa": "home_returning_ppa", "returning_usage": "home_returning_usage"})[
            ["season", "home_team", "home_returning_ppa", "home_returning_usage"]
        ],
        on=["season", "home_team"],
        how="left",
    )
    frame = frame.merge(
        ret.rename(columns={"ret_team": "away_team", "returning_ppa": "away_returning_ppa", "returning_usage": "away_returning_usage"})[
            ["season", "away_team", "away_returning_ppa", "away_returning_usage"]
        ],
        on=["season", "away_team"],
        how="left",
    )

    adv_prior = advanced.rename(columns={"year": "season", "team": "adv_team"})
    adv_prior["season"] = adv_prior["season"] + 1
    frame = frame.merge(
        adv_prior.rename(
            columns={
                "adv_team": "home_team",
                "off_success_rate": "home_success_prior",
                "def_success_rate": "home_def_success_prior",
            }
        )[["season", "home_team", "home_success_prior", "home_def_success_prior"]],
        on=["season", "home_team"],
        how="left",
    )
    frame = frame.merge(
        adv_prior.rename(
            columns={
                "adv_team": "away_team",
                "off_success_rate": "away_success_prior",
                "def_success_rate": "away_def_success_prior",
            }
        )[["season", "away_team", "away_success_prior", "away_def_success_prior"]],
        on=["season", "away_team"],
        how="left",
    )

    frame = frame.merge(consensus_lines(lines), on="game_id", how="left")
    carry = season_ending_elo(games)
    frame = frame.merge(
        carry.rename(columns={"team": "home_team", "carry_elo": "home_carry_elo"}),
        on=["season", "home_team"],
        how="left",
    )
    frame = frame.merge(
        carry.rename(columns={"team": "away_team", "carry_elo": "away_carry_elo"}),
        on=["season", "away_team"],
        how="left",
    )
    home_elo = _to_float(frame["home_pregame_elo"])
    away_elo = _to_float(frame["away_pregame_elo"])
    frame["home_pregame_elo"] = home_elo.where(home_elo.ne(1500)).fillna(_to_float(frame["home_carry_elo"]))
    frame["away_pregame_elo"] = away_elo.where(away_elo.ne(1500)).fillna(_to_float(frame["away_carry_elo"]))
    frame["elo_diff"] = _to_float(frame["home_pregame_elo"]) - _to_float(frame["away_pregame_elo"])
    frame["conference_game"] = frame["conference_game"].astype("float")
    frame["neutral_site"] = frame["neutral_site"].astype("float")
    return _attach_weather(frame, weather)


def _guess_tz(lat: float, lon: float) -> ZoneInfo:
    if pd.isna(lon):
        return ZoneInfo("America/Chicago")
    lon = float(lon)
    lat = float(lat) if not pd.isna(lat) else 0.0
    if lon < -150 or (lat < 25 and lon < -140):
        return ZoneInfo("Pacific/Honolulu")
    if lon > -25:
        return ZoneInfo("Europe/Dublin")
    if lon < -115:
        return ZoneInfo("America/Los_Angeles")
    if lon < -102:
        return ZoneInfo("America/Denver")
    if lon < -87:
        return ZoneInfo("America/Chicago")
    return ZoneInfo("America/New_York")


def _local_game_date(start, lat, lon) -> str | None:
    ts = pd.to_datetime(start, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    try:
        local = ts.tz_convert(_guess_tz(lat, lon))
    except Exception:
        local = ts.tz_convert(ZoneInfo("America/Chicago"))
    return local.strftime("%Y-%m-%d")


def _attach_weather(frame: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ["wx_temp_max", "wx_temp_min", "wx_precip", "wx_wind", "wx_precip_exposed", "wx_wind_exposed"]:
        if col not in out.columns:
            out[col] = np.nan
    if weather.empty or "venue_id" not in out.columns:
        return out
    wx = weather.copy()
    wx["venue_id"] = pd.to_numeric(wx["venue_id"], errors="coerce")
    out["venue_id"] = pd.to_numeric(out["venue_id"], errors="coerce")
    lats = out["venue_lat"] if "venue_lat" in out.columns else pd.Series(np.nan, index=out.index)
    lons = out["venue_lon"] if "venue_lon" in out.columns else pd.Series(np.nan, index=out.index)
    if "home_lat" in out.columns:
        lats = lats.fillna(out["home_lat"])
        lons = lons.fillna(out["home_lon"])
    out["game_date"] = [
        _local_game_date(start, lat, lon)
        for start, lat, lon in zip(out["start_date"], lats, lons, strict=False)
    ]
    wx = wx.rename(
        columns={
            "date": "game_date",
            "temp_max_c": "wx_temp_max",
            "temp_min_c": "wx_temp_min",
            "precip_mm": "wx_precip",
            "wind_kmh": "wx_wind",
        }
    )
    keep = [col for col in ["venue_id", "game_date", "wx_temp_max", "wx_temp_min", "wx_precip", "wx_wind"] if col in wx.columns]
    merged = out.merge(wx[keep], on=["venue_id", "game_date"], how="left", suffixes=("", "_wx"))
    for col in ["wx_temp_max", "wx_temp_min", "wx_precip", "wx_wind"]:
        src = f"{col}_wx"
        if src in merged.columns:
            merged[col] = _to_float(merged[src]).combine_first(_to_float(merged[col]))
            merged = merged.drop(columns=[src])
        else:
            merged[col] = _to_float(merged.get(col))
    # Climatology fill for dates beyond the forecast window.
    wx["mmdd"] = wx["game_date"].astype(str).str.slice(5, 10)
    climo = (
        wx.groupby(["venue_id", "mmdd"], as_index=False)[["wx_temp_max", "wx_temp_min", "wx_precip", "wx_wind"]]
        .mean()
        .rename(
            columns={
                "wx_temp_max": "climo_temp_max",
                "wx_temp_min": "climo_temp_min",
                "wx_precip": "climo_precip",
                "wx_wind": "climo_wind",
            }
        )
    )
    merged["mmdd"] = merged["game_date"].astype(str).str.slice(5, 10)
    merged = merged.merge(climo, on=["venue_id", "mmdd"], how="left")
    future = ~merged["completed"].astype(bool) if "completed" in merged.columns else False
    for col, climo_col in [
        ("wx_temp_max", "climo_temp_max"),
        ("wx_temp_min", "climo_temp_min"),
        ("wx_precip", "climo_precip"),
        ("wx_wind", "climo_wind"),
    ]:
        if isinstance(future, pd.Series):
            merged.loc[future, col] = _to_float(merged.loc[future, col]).fillna(_to_float(merged.loc[future, climo_col]))
        else:
            merged[col] = _to_float(merged[col]).fillna(_to_float(merged[climo_col]))
    merged = _fill_nearest_weather(merged, wx)
    merged = _fill_nearest_climo(merged, wx)
    dome = _to_float(merged.get("home_dome")).fillna(0).clip(0, 1)
    exposed = 1.0 - dome
    merged["wx_precip_exposed"] = _to_float(merged["wx_precip"]) * exposed
    merged["wx_wind_exposed"] = _to_float(merged["wx_wind"]) * exposed
    coverage = float(merged["wx_temp_max"].notna().mean())
    print(f"Weather coverage after merge: {coverage:.1%} of games", flush=True)
    return merged.drop(columns=["mmdd", "climo_temp_max", "climo_temp_min", "climo_precip", "climo_wind"], errors="ignore")


def _fill_nearest_weather(merged: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    """Copy weather from the closest stadium that has an observation that day."""
    if "game_id" not in merged.columns or "home_lat" not in merged.columns:
        return merged
    have = _to_float(merged["wx_temp_max"]).notna()
    search_lat = "venue_lat" if "venue_lat" in merged.columns else "home_lat"
    search_lon = "venue_lon" if "venue_lon" in merged.columns else "home_lon"
    if search_lat not in merged.columns:
        return merged
    coord_lat = merged[search_lat].fillna(merged["home_lat"]) if "home_lat" in merged.columns else merged[search_lat]
    coord_lon = merged[search_lon].fillna(merged["home_lon"]) if "home_lon" in merged.columns else merged[search_lon]
    anchors = (
        merged.loc[have, ["venue_id"]].assign(wx_lat=coord_lat[have], wx_lon=coord_lon[have])
        .dropna()
        .drop_duplicates("venue_id")
        .rename(columns={"venue_id": "wx_venue_id"})
    )
    if anchors.empty:
        return merged
    wx_pts = wx.merge(anchors, left_on="venue_id", right_on="wx_venue_id", how="inner")
    need = merged.loc[~have & merged["game_date"].notna(), ["game_id", "game_date"]].drop_duplicates("game_id")
    need = need.assign(search_lat=coord_lat.reindex(need.index), search_lon=coord_lon.reindex(need.index))
    if need.empty or wx_pts.empty:
        return merged
    cand = need.merge(
        wx_pts[["game_date", "wx_lat", "wx_lon", "wx_temp_max", "wx_temp_min", "wx_precip", "wx_wind"]],
        on="game_date",
        how="left",
    )
    cand["dist"] = haversine_miles(cand["search_lat"], cand["search_lon"], cand["wx_lat"], cand["wx_lon"])
    cand = cand[cand["dist"].notna() & (cand["dist"] <= 250)]
    if cand.empty:
        return merged
    best = cand.sort_values("dist").groupby("game_id", as_index=False).first()
    filled = merged.merge(
        best[["game_id", "wx_temp_max", "wx_temp_min", "wx_precip", "wx_wind"]].rename(
            columns={
                "wx_temp_max": "near_temp_max",
                "wx_temp_min": "near_temp_min",
                "wx_precip": "near_precip",
                "wx_wind": "near_wind",
            }
        ),
        on="game_id",
        how="left",
    )
    for col, near in [
        ("wx_temp_max", "near_temp_max"),
        ("wx_temp_min", "near_temp_min"),
        ("wx_precip", "near_precip"),
        ("wx_wind", "near_wind"),
    ]:
        filled[col] = _to_float(filled[col]).fillna(_to_float(filled[near]))
    return filled.drop(columns=["near_temp_max", "near_temp_min", "near_precip", "near_wind"])


def _fill_nearest_climo(merged: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    """For upcoming games past the forecast window, use nearby stadium climatology."""
    if "completed" not in merged.columns or "game_id" not in merged.columns:
        return merged
    future_missing = (~merged["completed"].astype(bool)) & _to_float(merged["wx_temp_max"]).isna()
    if not future_missing.any():
        return merged
    hist = wx.copy()
    this_year = str(pd.Timestamp.now(tz="UTC").year)
    hist = hist[hist["game_date"].astype(str).str.slice(0, 4) < this_year]
    if hist.empty:
        return merged
    hist["mmdd"] = hist["game_date"].astype(str).str.slice(5, 10)
    climo = hist.groupby(["venue_id", "mmdd"], as_index=False)[["wx_temp_max", "wx_temp_min", "wx_precip", "wx_wind"]].mean()
    site_lat = merged["venue_lat"].fillna(merged["home_lat"]) if "venue_lat" in merged.columns else merged["home_lat"]
    site_lon = merged["venue_lon"].fillna(merged["home_lon"]) if "venue_lon" in merged.columns else merged["home_lon"]
    sites = (
        merged[["venue_id"]]
        .assign(wx_lat=site_lat, wx_lon=site_lon)
        .dropna()
        .drop_duplicates("venue_id")
    )
    climo = climo.merge(sites, on="venue_id", how="inner")
    if climo.empty:
        return merged
    need = merged.loc[future_missing, ["game_id", "game_date"]].drop_duplicates("game_id")
    need = need.assign(search_lat=site_lat.reindex(need.index), search_lon=site_lon.reindex(need.index))
    need["mmdd"] = need["game_date"].astype(str).str.slice(5, 10)
    cand = need.merge(
        climo.rename(
            columns={
                "wx_temp_max": "climo_t",
                "wx_temp_min": "climo_n",
                "wx_precip": "climo_p",
                "wx_wind": "climo_w",
            }
        ),
        on="mmdd",
        how="left",
    )
    cand["dist"] = haversine_miles(cand["search_lat"], cand["search_lon"], cand["wx_lat"], cand["wx_lon"])
    cand = cand[cand["dist"].notna() & (cand["dist"] <= 250)]
    if cand.empty:
        return merged
    best = cand.sort_values("dist").groupby("game_id", as_index=False).first()
    filled = merged.merge(
        best[["game_id", "climo_t", "climo_n", "climo_p", "climo_w"]],
        on="game_id",
        how="left",
    )
    filled["wx_temp_max"] = _to_float(filled["wx_temp_max"]).fillna(_to_float(filled["climo_t"]))
    filled["wx_temp_min"] = _to_float(filled["wx_temp_min"]).fillna(_to_float(filled["climo_n"]))
    filled["wx_precip"] = _to_float(filled["wx_precip"]).fillna(_to_float(filled["climo_p"]))
    filled["wx_wind"] = _to_float(filled["wx_wind"]).fillna(_to_float(filled["climo_w"]))
    return filled.drop(columns=["climo_t", "climo_n", "climo_p", "climo_w"])


def model_matrix(frame: pd.DataFrame, feature_cols: Iterable[str] = FEATURE_COLS) -> pd.DataFrame:
    cols = [col for col in feature_cols if col in frame.columns]
    matrix = frame[cols].copy()
    for col in cols:
        matrix[col] = _to_float(matrix[col])
    return matrix
