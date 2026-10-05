"""Synthesize and score FBS matchups, including hypotheticals."""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from src import quarters as quarters_mod
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
    "mov_x_l4",
    "mov_x_l8",
    "mov_x_season",
    "sos_l4",
]

_UNSET = object()
_FRAME_CACHE: dict[str, Any] = {"mtime": None, "frame": None}
_MODEL_CACHE: dict[str, Any] | None = None
_MODEL_MTIMES: tuple | None = None
_QUARTER_CACHE: dict[str, Any] = {"mtime": None, "model": None}

MODEL_FILES = {
    "margin": "margin_hgb.joblib",
    "win": "win_hgb.joblib",
    "ats": "ats_residual.joblib",
}


HOLDOUT_MODEL_FILES = {
    "margin": "holdout_margin.joblib",
    "win": "holdout_win.joblib",
    "ats": "holdout_ats.joblib",
}


def load_holdout_models() -> dict[str, Any] | None:
    """Models fit only on seasons before the holdout (None if not trained yet)."""
    if not all((MODELS_DIR / name).exists() for name in HOLDOUT_MODEL_FILES.values()):
        return None
    try:
        return {key: joblib.load(MODELS_DIR / name) for key, name in HOLDOUT_MODEL_FILES.items()}
    except Exception:  # noqa: BLE001 - stale/incompatible pickle: caller falls back
        return None


def _norm(value: object) -> str:
    return " ".join(str(value or "").lower().split())


def _model_mtimes() -> tuple:
    return tuple(
        (MODELS_DIR / name).stat().st_mtime_ns if (MODELS_DIR / name).exists() else None
        for name in MODEL_FILES.values()
    )


def load_quarter_model() -> "quarters_mod.QuarterModel | None":
    """Fitted score-by-quarter model, or None before the first training run with quarter data."""
    path = MODELS_DIR / quarters_mod.MODEL_FILE
    mtime = path.stat().st_mtime_ns if path.exists() else None
    if _QUARTER_CACHE["mtime"] != mtime or (mtime is not None and _QUARTER_CACHE["model"] is None):
        _QUARTER_CACHE["model"] = quarters_mod.QuarterModel.load(MODELS_DIR) if mtime is not None else None
        _QUARTER_CACHE["mtime"] = mtime
        quarters_mod._LINE_CACHE.clear()  # cached line scores belong to the previous fit
    return _QUARTER_CACHE["model"]


def invalidate_cache() -> None:
    global _MODEL_CACHE
    _QUARTER_CACHE["mtime"] = None
    _QUARTER_CACHE["model"] = None
    _FRAME_CACHE["mtime"] = None
    _FRAME_CACHE["frame"] = None
    _MODEL_CACHE = None


def cached_feature_frame() -> pd.DataFrame | None:
    frame = _FRAME_CACHE.get("frame")
    return None if frame is None else frame


def get_feature_frame(force: bool = False) -> pd.DataFrame:
    try:
        from src.schedule_sync import sync_schedule_results

        sync_schedule_results()
    except Exception:
        pass
    mtime = DB_PATH.stat().st_mtime if DB_PATH.exists() else None
    if not force and _FRAME_CACHE["frame"] is not None and _FRAME_CACHE["mtime"] == mtime:
        return _FRAME_CACHE["frame"]
    from src.week0 import assign_board_week

    frame = assign_board_week(build_feature_frame())
    _FRAME_CACHE["frame"] = frame
    _FRAME_CACHE["mtime"] = mtime
    return frame


def load_models(force: bool = False) -> dict[str, Any]:
    global _MODEL_CACHE, _MODEL_MTIMES
    # Re-load when a retrain (possibly out-of-band, e.g. cron) replaced the files.
    if _MODEL_CACHE is not None and not force and _MODEL_MTIMES == _model_mtimes():
        return _MODEL_CACHE
    missing = [name for name in MODEL_FILES.values() if not (MODELS_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {', '.join(missing)}. Train the model before scoring matchups."
        )
    try:
        _MODEL_MTIMES = _model_mtimes()
        _MODEL_CACHE = {key: joblib.load(MODELS_DIR / filename) for key, filename in MODEL_FILES.items()}
    except Exception as exc:  # noqa: BLE001 - pickle/version mismatch should become a train error
        _MODEL_CACHE = None
        raise FileNotFoundError(
            f"Saved models could not be loaded ({exc}). Retrain on latest games."
        ) from exc
    return _MODEL_CACHE


def models_ready() -> bool:
    return all((MODELS_DIR / name).exists() for name in MODEL_FILES.values())


def last_completed_season(frame: pd.DataFrame) -> int | None:
    games = frame[frame["fbs_vs_fbs"].astype(bool)] if "fbs_vs_fbs" in frame.columns else frame
    if games.empty or "season" not in games.columns:
        return None
    remaining = games.groupby("season")["completed"].agg(lambda s: int((~s.astype(bool)).sum()))
    done = remaining[remaining == 0]
    if done.empty:
        return None
    return int(done.index.max())


def legal_football_points(value: object) -> int | None:
    """Whole-number football score. 0 is a shutout; 1 is not a scoring play."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        points = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    if points < 0:
        points = 0
    if points == 1:
        points = 0
    return points


def _legal_score_series(values: pd.Series) -> pd.Series:
    nums = pd.to_numeric(values, errors="coerce")
    snapped = nums.round().clip(lower=0)
    return snapped.mask(snapped.eq(1), 0)


def _no_ties(home: pd.Series, away: pd.Series, margin: pd.Series) -> tuple[pd.Series, pd.Series]:
    """College football has no ties: when rounding makes the scores equal, the predicted winner gets +1.

    Rounding is monotonic, so a tie is the only way it can disagree with the sign of the margin.
    A winner at 0 jumps to 2 (a safety), since a team cannot score exactly 1.
    """
    tied = home.notna() & away.notna() & home.eq(away)
    home_wins = pd.to_numeric(margin, errors="coerce").fillna(0.0) >= 0
    bump = pd.Series(np.where(home.eq(0), 2.0, home + 1.0), index=home.index)
    home = home.where(~(tied & home_wins), bump)
    away = away.where(~(tied & ~home_wins), pd.Series(np.where(away.eq(0), 2.0, away + 1.0), index=away.index))
    return home, away


DEFAULT_TOTAL = 54.5


def _numeric_col(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[name], errors="coerce")


def predicted_totals(frame: pd.DataFrame) -> pd.Series:
    """Market over/under when posted; otherwise recent scoring form; else a CFB default."""
    market = _numeric_col(frame, "over_under")
    home_off = _numeric_col(frame, "home_points_l4").fillna(_numeric_col(frame, "home_points_prior"))
    away_off = _numeric_col(frame, "away_points_l4").fillna(_numeric_col(frame, "away_points_prior"))
    home_def = _numeric_col(frame, "home_points_allowed_l4").fillna(
        _numeric_col(frame, "home_points_allowed_prior")
    )
    away_def = _numeric_col(frame, "away_points_allowed_l4").fillna(
        _numeric_col(frame, "away_points_allowed_prior")
    )
    home_exp = ((home_off + away_def) / 2.0).fillna(home_off)
    away_exp = ((away_off + home_def) / 2.0).fillna(away_off)
    estimated = home_exp + away_exp
    return market.fillna(estimated).fillna(DEFAULT_TOTAL)


def score_slate(slate: pd.DataFrame, models: dict[str, Any] | None = None) -> pd.DataFrame:
    packed = models or load_models()
    out = slate.copy()
    margin = packed["margin"]
    win = packed["win"]
    ats = packed["ats"]
    out["pred_margin"] = margin["model"].predict(model_matrix(out, margin["features"]))
    out["pred_home_win_prob"] = win["model"].predict_proba(model_matrix(out, win["features"]))[:, 1]
    totals = predicted_totals(out)
    out["pred_home_points"], out["pred_away_points"] = _no_ties(
        _legal_score_series(totals / 2.0 + out["pred_margin"] / 2.0),
        _legal_score_series(totals / 2.0 - out["pred_margin"] / 2.0),
        out["pred_margin"],
    )
    _attach_quarters(out, totals)
    spread = pd.to_numeric(out["spread"], errors="coerce") if "spread" in out.columns else pd.Series(np.nan, index=out.index)
    out["edge_vs_spread"] = out["pred_margin"] + spread
    out["ats_edge"] = np.nan
    lined = out.loc[spread.notna()]
    if not lined.empty:
        out.loc[lined.index, "ats_edge"] = ats["model"].predict(model_matrix(lined, ats["features"]))
    return annotate_board(out)


def _attach_quarters(out: pd.DataFrame, totals: pd.Series) -> None:
    """Replace rounded scores with a realistic quarter-by-quarter game (in place).

    The displayed final is the sum of the quarters, so it can only be a reachable score.
    Rows without a usable margin/total keep the rounded fallback.
    """
    model = load_quarter_model()
    if model is None or out.empty:
        return
    cols = {f"pred_{side}_q{q}": np.full(len(out), np.nan) for side in ("home", "away") for q in (1, 2, 3, 4)}
    for pos, (_, row) in enumerate(out.iterrows()):
        margin = pd.to_numeric(row.get("pred_margin"), errors="coerce")
        total = pd.to_numeric(totals.iloc[pos], errors="coerce")
        if pd.isna(margin) or pd.isna(total):
            continue
        game = quarters_mod.typical_line_score(
            model, str(row.get("home_team") or ""), str(row.get("away_team") or ""), float(total), float(margin), n_sims=1000
        )
        for q in range(4):
            cols[f"pred_home_q{q + 1}"][pos] = game["home"][q]
            cols[f"pred_away_q{q + 1}"][pos] = game["away"][q]
    for name, values in cols.items():
        out[name] = values
    have = ~pd.isna(cols["pred_home_q1"])
    for side in ("home", "away"):
        total_pts = sum(out[f"pred_{side}_q{q}"] for q in (1, 2, 3, 4))
        out.loc[have, f"pred_{side}_points"] = total_pts[have]


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
    if not upcoming.empty and "season" in rows.columns:
        # Cancelled/never-played games from old seasons stay "uncompleted" forever.
        upcoming = upcoming[upcoming["season"] == rows["season"].max()]
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
    for name in ("mov_x_l4", "mov_x_l8", "mov_x_season", "sos_l4"):
        h = pd.to_numeric(pd.Series([home.get(name)]), errors="coerce").iloc[0]
        a = pd.to_numeric(pd.Series([away.get(name)]), errors="coerce").iloc[0]
        row[f"{name}_diff"] = (h - a) if pd.notna(h) and pd.notna(a) else np.nan
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
    if "completed" in pair.columns:
        # Only a still-scheduled meeting carries a live line/venue/week; an old result's are stale.
        pair = pair[~pair["completed"].astype(bool)]
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


def _score_int(value: object) -> int | None:
    number = _json_num(value)
    if number is None:
        return None
    return int(round(float(number)))


def _result_fields(row: pd.Series) -> dict[str, Any]:
    completed = bool(row.get("completed"))
    home_points = _score_int(row.get("home_points")) if completed else None
    away_points = _score_int(row.get("away_points")) if completed else None
    pred_margin = _json_num(row.get("pred_margin"))
    spread = _json_num(row.get("spread"))
    actual_margin = None
    winner_hit = None
    ats_hit = None
    if home_points is not None and away_points is not None:
        actual_margin = float(home_points - away_points)
        if pred_margin is not None:
            if actual_margin == 0:
                winner_hit = pred_margin == 0
            else:
                winner_hit = (pred_margin >= 0) == (actual_margin > 0)
        if spread is not None and pred_margin is not None:
            actual_cover = actual_margin + float(spread)
            pred_cover = float(pred_margin) + float(spread)
            if actual_cover == 0:
                ats_hit = None
            else:
                ats_hit = (pred_cover > 0) == (actual_cover > 0)
    return {
        "completed": completed,
        "home_points": home_points,
        "away_points": away_points,
        "actual_margin": actual_margin,
        "winner_hit": winner_hit,
        "ats_hit": ats_hit,
    }


def home_spread_label(abbr: str | None, spread: object, home_name: str = "") -> str | None:
    line = _json_num(spread)
    if line is None:
        return None
    line = float(line)
    tag = str(abbr or home_name or "HOME").strip().upper()
    if not tag:
        tag = "HOME"
    if abs(line) < 1e-9:
        signed = "PK"
    else:
        signed = f"{line:.2f}".rstrip("0").rstrip(".")
        if line > 0:
            signed = f"+{signed}"
    return f"{tag} {signed}"


def _quarter_list(row: pd.Series, side: str) -> list[int] | None:
    values = [_json_num(row.get(f"pred_{side}_q{q}")) for q in (1, 2, 3, 4)]
    return None if any(v is None for v in values) else [int(round(v)) for v in values]


def prediction_payload(row: pd.Series, teams: dict[str, dict] | None = None) -> dict[str, Any]:
    teams = teams or {}
    home_name = str(row.get("home_team") or "")
    away_name = str(row.get("away_team") or "")
    home_wp = _json_num(row.get("pred_home_win_prob"))
    away_wp = None if home_wp is None else round(1.0 - float(home_wp), 6)
    home_pts = legal_football_points(row.get("pred_home_points"))
    away_pts = legal_football_points(row.get("pred_away_points"))
    if home_pts is not None and home_pts == away_pts:
        margin = _json_num(row.get("pred_margin"))
        if margin is None and home_wp is not None:
            margin = home_wp - 0.5
        h, a = _no_ties(pd.Series([float(home_pts)]), pd.Series([float(away_pts)]), pd.Series([margin]))
        home_pts, away_pts = int(h.iloc[0]), int(a.iloc[0])
    home_quarters = _quarter_list(row, "home")
    away_quarters = _quarter_list(row, "away")
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
    home_team = teams.get(home_name) or {"name": home_name, "school": home_name}
    away_team = teams.get(away_name) or {"name": away_name, "school": away_name}
    return {
        "id": game_id_out,
        "game_id": game_id_out,
        "season": _json_num(row.get("season")),
        "week": _json_num(row["board_week"] if "board_week" in row.index and pd.notna(row.get("board_week")) else row.get("week")),
        "status": str(status),
        "start_date": _iso(row.get("start_date")),
        "kick_label": None if pd.isna(row.get("kick_label")) else str(row.get("kick_label") or "") or None,
        "venue": None if pd.isna(row.get("venue")) else row.get("venue"),
        "neutral_site": bool(row.get("neutral_site")),
        "conference_game": bool(row.get("conference_game")),
        "home_team": home_team,
        "away_team": away_team,
        "home_win_prob": home_wp,
        "away_win_prob": away_wp,
        "projected_home_score": home_pts,
        "projected_away_score": away_pts,
        "pred_margin": _json_num(row.get("pred_margin")),
        "pred_home_points": home_pts,
        "pred_away_points": away_pts,
        "pred_home_quarters": home_quarters,
        "pred_away_quarters": away_quarters,
        "pick": None if pd.isna(row.get("pick")) else row.get("pick"),
        "pick_prob": _json_num(row.get("pick_prob")),
        "conf": None if pd.isna(row.get("conf")) else row.get("conf"),
        "cover_side": None if cover is None else str(cover),
        "ats_edge": _json_num(row.get("ats_edge")),
        "edge_vs_spread": _json_num(row.get("edge_vs_spread")),
        "spread": _json_num(row.get("spread")),
        "spread_label": home_spread_label(home_team.get("abbreviation"), row.get("spread"), home_name),
        "over_under": _json_num(row.get("over_under")),
        "wx_temp_max": _json_num(row.get("wx_temp_max")),
        "wx_temp_min": _json_num(row.get("wx_temp_min")),
        "wx_precip": _json_num(row.get("wx_precip")),
        "wx_wind": _json_num(row.get("wx_wind")),
        **_result_fields(row),
    }
