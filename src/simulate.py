"""Synthesize and score FBS matchups, including hypotheticals."""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import DB_PATH, MODELS_DIR
from src.features import build_feature_frame, haversine_miles, model_matrix
from src.train import annotate_board

SNAP_FIELDS = [
    "pregame_elo",
    "talent",
    "sp",
    "sp_off",
    "sp_def",
    "recruit",
    "returning_ppa",
    "returning_usage",
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
    "success_prior",
    "def_success_prior",
]

_UNSET = object()
_FRAME_CACHE: dict[str, Any] = {"mtime": None, "frame": None}
_MODEL_CACHE: dict[str, Any] | None = None

MODEL_FILES = {
    "margin": "margin_hgb.joblib",
    "win": "win_hgb.joblib",
    "ats": "ats_residual.joblib",
}


def _norm(value: object) -> str:
    return " ".join(str(value or "").lower().split())


def invalidate_cache() -> None:
    global _MODEL_CACHE
    _FRAME_CACHE["mtime"] = None
    _FRAME_CACHE["frame"] = None
    _MODEL_CACHE = None


def get_feature_frame(force: bool = False) -> pd.DataFrame:
    mtime = DB_PATH.stat().st_mtime if DB_PATH.exists() else None
    if not force and _FRAME_CACHE["frame"] is not None and _FRAME_CACHE["mtime"] == mtime:
        return _FRAME_CACHE["frame"]
    frame = build_feature_frame()
    _FRAME_CACHE["frame"] = frame
    _FRAME_CACHE["mtime"] = mtime
    return frame


def load_models(force: bool = False) -> dict[str, Any]:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None and not force:
        return _MODEL_CACHE
    missing = [name for name in MODEL_FILES.values() if not (MODELS_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {', '.join(missing)}. Train the model before scoring matchups."
        )
    try:
        _MODEL_CACHE = {key: joblib.load(MODELS_DIR / filename) for key, filename in MODEL_FILES.items()}
    except Exception as exc:  # noqa: BLE001 - pickle/version mismatch should become a train error
        _MODEL_CACHE = None
        raise FileNotFoundError(
            f"Saved models could not be loaded ({exc}). Retrain on latest games."
        ) from exc
    return _MODEL_CACHE


def models_ready() -> bool:
    return all((MODELS_DIR / name).exists() for name in MODEL_FILES.values())


def score_slate(slate: pd.DataFrame, models: dict[str, Any] | None = None) -> pd.DataFrame:
    packed = models or load_models()
    out = slate.copy()
    margin = packed["margin"]
    win = packed["win"]
    ats = packed["ats"]
    out["pred_margin"] = margin["model"].predict(model_matrix(out, margin["features"]))
    out["pred_home_win_prob"] = win["model"].predict_proba(model_matrix(out, win["features"]))[:, 1]
    totals = pd.to_numeric(out["over_under"], errors="coerce") if "over_under" in out.columns else pd.Series(np.nan, index=out.index)
    out["pred_home_points"] = totals / 2.0 + out["pred_margin"] / 2.0
    out["pred_away_points"] = totals / 2.0 - out["pred_margin"] / 2.0
    spread = pd.to_numeric(out["spread"], errors="coerce") if "spread" in out.columns else pd.Series(np.nan, index=out.index)
    out["edge_vs_spread"] = out["pred_margin"] + spread
    out["ats_edge"] = np.nan
    lined = out.loc[spread.notna()]
    if not lined.empty:
        out.loc[lined.index, "ats_edge"] = ats["model"].predict(model_matrix(lined, ats["features"]))
    return annotate_board(out)


def _team_rows(frame: pd.DataFrame, team: str) -> pd.DataFrame:
    key = _norm(team)
    home_mask = frame["home_team"].map(_norm).eq(key)
    away_mask = frame["away_team"].map(_norm).eq(key)
    rows = frame.loc[home_mask | away_mask].copy()
    if rows.empty:
        raise KeyError(f"Unknown team {team!r}")
    rows["_start"] = pd.to_datetime(rows["start_date"], utc=True, errors="coerce")
    return rows.sort_values("_start")


def team_snapshot(frame: pd.DataFrame, team: str, as_of: object | None = None) -> dict[str, Any]:
    rows = _team_rows(frame, team)
    if as_of is not None:
        cutoff = pd.to_datetime(as_of, utc=True, errors="coerce")
        rows = rows[rows["_start"] <= cutoff]
        if rows.empty:
            raise KeyError(f"No games for {team!r} before {as_of}")
    upcoming = rows[~rows["completed"].astype(bool)] if "completed" in rows.columns else rows.iloc[0:0]
    source = upcoming.iloc[0] if not upcoming.empty else rows.iloc[-1]
    is_home = _norm(source.get("home_team")) == _norm(team)
    prefix = "home" if is_home else "away"
    display = source["home_team"] if is_home else source["away_team"]
    snap: dict[str, Any] = {
        "team": str(display),
        "team_id": source.get(f"{prefix}_id"),
        "conference": source.get(f"{prefix}_conference"),
        "lat": source.get(f"{prefix}_lat"),
        "lon": source.get(f"{prefix}_lon"),
        "elev": source.get(f"{prefix}_elev"),
        "last_start": source.get("start_date"),
        "season": source.get("season"),
        "capacity": source.get("home_capacity") if is_home else np.nan,
        "dome": source.get("home_dome") if is_home else np.nan,
        "grass": source.get("home_grass") if is_home else np.nan,
        "venue": source.get("venue") if is_home else None,
    }
    for field in SNAP_FIELDS:
        snap[field] = source.get(f"{prefix}_{field}")
    return snap


def _capacity_bucket(capacity: object) -> float:
    value = pd.to_numeric(pd.Series([capacity]), errors="coerce").iloc[0]
    if pd.isna(value):
        return np.nan
    return float(
        pd.cut(
            [value],
            bins=[-np.inf, 30000, 50000, 80000, np.inf],
            labels=[0, 1, 2, 3],
        )[0]
    )


def _rest_days(snap: dict[str, Any], kickoff: pd.Timestamp | None) -> float:
    if kickoff is None:
        value = pd.to_numeric(pd.Series([snap.get("rest_days")]), errors="coerce").iloc[0]
        return float(value) if pd.notna(value) else np.nan
    last = pd.to_datetime(snap.get("last_start"), utc=True, errors="coerce")
    if pd.isna(last):
        return np.nan
    return float((kickoff - last).total_seconds() / 86400.0)


def _combine_snapshots(
    home: dict[str, Any],
    away: dict[str, Any],
    *,
    season: int | None,
    week: int | None,
    kickoff: object | None,
    venue: str | None,
    neutral_site: bool,
    conference_game: bool | None,
    spread: float | None,
    over_under: float | None,
    wx_temp_max: float | None,
    wx_temp_min: float | None,
    wx_precip: float | None,
    wx_wind: float | None,
) -> dict[str, Any]:
    start = pd.to_datetime(kickoff, utc=True, errors="coerce") if kickoff else pd.NaT
    same_conf = _norm(home.get("conference")) == _norm(away.get("conference")) and bool(home.get("conference"))
    conf_flag = float(conference_game) if conference_game is not None else float(same_conf)
    dest_lat = home.get("lat")
    dest_lon = home.get("lon")
    distance = haversine_miles(
        pd.Series([away.get("lat")]),
        pd.Series([away.get("lon")]),
        pd.Series([dest_lat]),
        pd.Series([dest_lon]),
    ).iloc[0]
    home_elo = pd.to_numeric(pd.Series([home.get("pregame_elo")]), errors="coerce").iloc[0]
    away_elo = pd.to_numeric(pd.Series([away.get("pregame_elo")]), errors="coerce").iloc[0]
    home_talent = pd.to_numeric(pd.Series([home.get("talent")]), errors="coerce").iloc[0]
    away_talent = pd.to_numeric(pd.Series([away.get("talent")]), errors="coerce").iloc[0]
    home_sp = pd.to_numeric(pd.Series([home.get("sp")]), errors="coerce").iloc[0]
    away_sp = pd.to_numeric(pd.Series([away.get("sp")]), errors="coerce").iloc[0]
    home_rec = pd.to_numeric(pd.Series([home.get("recruit")]), errors="coerce").iloc[0]
    away_rec = pd.to_numeric(pd.Series([away.get("recruit")]), errors="coerce").iloc[0]
    home_elev = pd.to_numeric(pd.Series([home.get("elev")]), errors="coerce").iloc[0]
    away_elev = pd.to_numeric(pd.Series([away.get("elev")]), errors="coerce").iloc[0]
    row: dict[str, Any] = {
        "game_id": None,
        "status": "hypothetical",
        "completed": False,
        "fbs_vs_fbs": True,
        "season": int(season) if season is not None else int(home.get("season") or away.get("season") or 0) or None,
        "week": int(week) if week is not None else 1,
        "start_date": start if pd.notna(start) else pd.NaT,
        "home_team": home["team"],
        "away_team": away["team"],
        "home_id": home.get("team_id"),
        "away_id": away.get("team_id"),
        "home_conference": home.get("conference"),
        "away_conference": away.get("conference"),
        "conference_game": conf_flag,
        "neutral_site": float(bool(neutral_site)),
        "venue": venue if venue is not None else (None if neutral_site else home.get("venue")),
        "spread": np.nan if spread is None else spread,
        "over_under": np.nan if over_under is None else over_under,
        "home_lat": dest_lat,
        "home_lon": dest_lon,
        "home_elev": home_elev,
        "away_lat": away.get("lat"),
        "away_lon": away.get("lon"),
        "away_elev": away_elev,
        "home_capacity": home.get("capacity"),
        "home_dome": 0.0 if pd.isna(home.get("dome")) else home.get("dome"),
        "home_grass": home.get("grass"),
        "capacity_bucket": _capacity_bucket(home.get("capacity")),
        "distance_miles": float(distance) if pd.notna(distance) else np.nan,
        "elevation_diff": (home_elev - away_elev) if pd.notna(home_elev) and pd.notna(away_elev) else np.nan,
        "elo_diff": (home_elo - away_elo) if pd.notna(home_elo) and pd.notna(away_elo) else np.nan,
        "talent_diff": (home_talent - away_talent) if pd.notna(home_talent) and pd.notna(away_talent) else np.nan,
        "sp_diff": (home_sp - away_sp) if pd.notna(home_sp) and pd.notna(away_sp) else np.nan,
        "recruit_diff": (home_rec - away_rec) if pd.notna(home_rec) and pd.notna(away_rec) else np.nan,
        "home_rest_days": _rest_days(home, start if pd.notna(start) else None),
        "away_rest_days": _rest_days(away, start if pd.notna(start) else None),
        "wx_temp_max": np.nan if wx_temp_max is None else wx_temp_max,
        "wx_temp_min": np.nan if wx_temp_min is None else wx_temp_min,
        "wx_precip": np.nan if wx_precip is None else wx_precip,
        "wx_wind": np.nan if wx_wind is None else wx_wind,
    }
    for field in SNAP_FIELDS:
        if field == "rest_days":
            continue
        row[f"home_{field}"] = home.get(field)
        row[f"away_{field}"] = away.get(field)
    row["home_pregame_elo"] = home_elo
    row["away_pregame_elo"] = away_elo
    row["home_talent"] = home_talent
    row["away_talent"] = away_talent
    if wx_temp_max is None:
        row["wx_temp_max"] = np.nan
    if wx_temp_min is None:
        row["wx_temp_min"] = np.nan
    return row


def simulate_matchup(
    home_team: str,
    away_team: str,
    *,
    frame: pd.DataFrame | None = None,
    models: dict[str, Any] | None = None,
    season: int | None = None,
    week: int | None = None,
    kickoff: object | None = None,
    venue: str | None = None,
    neutral_site: bool = False,
    conference_game: bool | None = None,
    spread: object = _UNSET,
    over_under: object = _UNSET,
    game_id: int | None = None,
    wx_temp_max: object = _UNSET,
    wx_temp_min: object = _UNSET,
    wx_precip: float | None = None,
    wx_wind: float | None = None,
) -> pd.DataFrame:
    board = frame if frame is not None else get_feature_frame()
    if game_id is not None:
        matches = board.loc[pd.to_numeric(board["game_id"], errors="coerce") == int(game_id)].copy()
        if matches.empty:
            raise KeyError(f"Unknown game_id {game_id}")
        row = matches.iloc[[0]].copy()
        if spread is not _UNSET:
            row["spread"] = spread
        if over_under is not _UNSET:
            row["over_under"] = over_under
        if wx_temp_max is not _UNSET:
            row["wx_temp_max"] = wx_temp_max
        if wx_temp_min is not _UNSET:
            row["wx_temp_min"] = wx_temp_min
        if wx_precip is not None:
            row["wx_precip"] = wx_precip
        if wx_wind is not None:
            row["wx_wind"] = wx_wind
        if "status" not in row.columns:
            row["status"] = np.where(row["completed"].astype(bool), "completed", "scheduled")
        scored = score_slate(row, models=models)
        return scored.reset_index(drop=True)

    home = team_snapshot(board, home_team, as_of=kickoff)
    away = team_snapshot(board, away_team, as_of=kickoff)
    pair = board[
        board["home_team"].map(_norm).eq(_norm(home["team"]))
        & board["away_team"].map(_norm).eq(_norm(away["team"]))
    ]
    if not pair.empty:
        source = pair.sort_values("start_date").iloc[-1]
        if spread is _UNSET and pd.notna(source.get("spread")):
            spread = float(pd.to_numeric(source.get("spread"), errors="coerce"))
        if over_under is _UNSET and pd.notna(source.get("over_under")):
            over_under = float(pd.to_numeric(source.get("over_under"), errors="coerce"))
        if wx_temp_max is _UNSET and pd.notna(source.get("wx_temp_max")):
            wx_temp_max = float(pd.to_numeric(source.get("wx_temp_max"), errors="coerce"))
        if wx_temp_min is _UNSET and pd.notna(source.get("wx_temp_min")):
            wx_temp_min = float(pd.to_numeric(source.get("wx_temp_min"), errors="coerce"))
        if venue is None and pd.notna(source.get("venue")):
            venue = source.get("venue")
        if season is None and pd.notna(source.get("season")):
            season = int(source["season"])
        if week is None and pd.notna(source.get("week")):
            week = int(source["week"])
    combined = _combine_snapshots(
        home,
        away,
        season=season,
        week=week,
        kickoff=kickoff,
        venue=venue,
        neutral_site=neutral_site,
        conference_game=conference_game,
        spread=None if spread is _UNSET else spread,
        over_under=None if over_under is _UNSET else over_under,
        wx_temp_max=None if wx_temp_max is _UNSET else wx_temp_max,
        wx_temp_min=None if wx_temp_min is _UNSET else wx_temp_min,
        wx_precip=wx_precip,
        wx_wind=wx_wind,
    )
    slate = pd.DataFrame([combined])
    scored = score_slate(slate, models=models)
    scored["status"] = "hypothetical"
    return scored.reset_index(drop=True)


def _json_num(value: object) -> float | int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (int,)):
        return int(value)
    return value if isinstance(value, (str, bool)) else None


def _iso(value: object) -> str | None:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.isoformat()


def prediction_payload(row: pd.Series, teams: dict[str, dict] | None = None) -> dict[str, Any]:
    teams = teams or {}
    home_name = str(row.get("home_team") or "")
    away_name = str(row.get("away_team") or "")
    home_wp = _json_num(row.get("pred_home_win_prob"))
    away_wp = None if home_wp is None else round(1.0 - float(home_wp), 6)
    home_pts = _json_num(row.get("pred_home_points"))
    away_pts = _json_num(row.get("pred_away_points"))
    status = row.get("status")
    if not status:
        status = "completed" if bool(row.get("completed")) else "scheduled"
    game_id = row.get("game_id")
    try:
        game_id_out = None if game_id is None or pd.isna(game_id) else int(game_id)
    except (TypeError, ValueError):
        game_id_out = None
    cover = row.get("cover_side")
    if cover is not None:
        try:
            if pd.isna(cover):
                cover = None
        except (TypeError, ValueError):
            pass
    return {
        "id": game_id_out,
        "game_id": game_id_out,
        "season": _json_num(row.get("season")),
        "week": _json_num(row.get("week")),
        "status": str(status),
        "start_date": _iso(row.get("start_date")),
        "kick_label": None if pd.isna(row.get("kick_label")) else str(row.get("kick_label") or "") or None,
        "venue": None if pd.isna(row.get("venue")) else row.get("venue"),
        "neutral_site": bool(row.get("neutral_site")),
        "conference_game": bool(row.get("conference_game")),
        "home_team": teams.get(home_name) or {"name": home_name, "school": home_name},
        "away_team": teams.get(away_name) or {"name": away_name, "school": away_name},
        "home_win_prob": home_wp,
        "away_win_prob": away_wp,
        "projected_home_score": None if home_pts is None else round(float(home_pts), 1),
        "projected_away_score": None if away_pts is None else round(float(away_pts), 1),
        "pred_margin": _json_num(row.get("pred_margin")),
        "pred_home_points": home_pts,
        "pred_away_points": away_pts,
        "pick": None if pd.isna(row.get("pick")) else row.get("pick"),
        "pick_prob": _json_num(row.get("pick_prob")),
        "conf": None if pd.isna(row.get("conf")) else row.get("conf"),
        "cover_side": None if cover is None else str(cover),
        "ats_edge": _json_num(row.get("ats_edge")),
        "edge_vs_spread": _json_num(row.get("edge_vs_spread")),
        "spread": _json_num(row.get("spread")),
        "over_under": _json_num(row.get("over_under")),
        "wx_temp_max": _json_num(row.get("wx_temp_max")),
        "wx_temp_min": _json_num(row.get("wx_temp_min")),
        "wx_precip": _json_num(row.get("wx_precip")),
        "wx_wind": _json_num(row.get("wx_wind")),
    }
