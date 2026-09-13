"""Frozen-base catch-up plus a per-week residual layer.

The HGB trees are not refit weekly. After every FBS-vs-FBS game in a week has
a final, a ridge layer is trained on that week's (actual margin − base pick).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from src.config import MODELS_DIR
from src.features import model_matrix

MIN_WEEK_GAMES = 8
WEEK_LAYER_FILE = "week_layer.joblib"
STATE_FILE = "weekly.json"
LOG_FILE = "predictions.jsonl"
WEEK_LAYER_FEATURES = [
    "elo_diff",
    "sp_diff",
    "talent_diff",
    "recruit_diff",
    "week",
    "conference_game",
    "home_points_l4",
    "away_points_l4",
]

_poller_stop = threading.Event()
_poller_thread: threading.Thread | None = None


def _state_path() -> Path:
    return MODELS_DIR / STATE_FILE


def _log_path() -> Path:
    return MODELS_DIR / LOG_FILE


def _layer_path() -> Path:
    return MODELS_DIR / WEEK_LAYER_FILE


def _fbs(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame.iloc[0:0] if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    if "fbs_vs_fbs" not in frame.columns:
        return frame
    return frame[frame["fbs_vs_fbs"].astype(bool)].copy()


def completed_weeks(frame: pd.DataFrame) -> list[tuple[int, int]]:
    """Board weeks whose FBS-vs-FBS games are all final."""
    games = _fbs(frame)
    if games.empty or "season" not in games.columns or "week" not in games.columns:
        return []
    done: list[tuple[int, int]] = []
    grouped = games.groupby(["season", "week"], sort=True)
    for (season, week), slate in grouped:
        if slate.empty:
            continue
        if bool(slate["completed"].astype(bool).all()):
            done.append((int(season), int(week)))
    return done


def load_state() -> dict[str, Any]:
    path = _state_path()
    if not path.exists():
        return {"base_weeks": [], "applied_weeks": [], "latest_week": None}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"base_weeks": [], "applied_weeks": [], "latest_week": None}
    data.setdefault("base_weeks", [])
    data.setdefault("applied_weeks", [])
    data.setdefault("latest_week", None)
    return data


def save_state(state: dict[str, Any]) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = _state_path()
    path.write_text(json.dumps(state, indent=2, default=str))
    return path


def _week_tuples(rows: list) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for row in rows or []:
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            out.append((int(row[0]), int(row[1])))
    return out


def record_base_catchup(frame: pd.DataFrame) -> dict[str, Any]:
    state = load_state()
    state["base_weeks"] = [list(pair) for pair in completed_weeks(frame)]
    state["applied_weeks"] = []
    state["base_ready"] = True
    state["updated_at"] = time.time()
    save_state(state)
    return state


def log_predictions(frame: pd.DataFrame) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = _log_path()
    seen: set[int] = set()
    if dest.exists():
        for line in dest.read_text().splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            gid = payload.get("game_id")
            if gid is not None:
                seen.add(int(gid))
    keep_cols = [
        c
        for c in ["game_id", "season", "week", "home_team", "away_team", "pred_margin", "pred_margin_raw"]
        if c in frame.columns
    ]
    with dest.open("a", encoding="utf-8") as handle:
        for row in frame[keep_cols].to_dict(orient="records"):
            gid = row.get("game_id")
            try:
                gid_int = None if gid is None or pd.isna(gid) else int(gid)
            except (TypeError, ValueError):
                gid_int = None
            if gid_int is None or gid_int in seen:
                continue
            row["game_id"] = gid_int
            handle.write(json.dumps(row, default=str) + "\n")
            seen.add(gid_int)
    return dest


def logged_pred_margin(frame: pd.DataFrame) -> pd.Series:
    dest = _log_path()
    mapping: dict[int, float] = {}
    if dest.exists():
        for line in dest.read_text().splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            gid = payload.get("game_id")
            pred = payload.get("pred_margin_raw", payload.get("pred_margin"))
            if gid is None or pred is None:
                continue
            mapping.setdefault(int(gid), float(pred))
    ids = pd.to_numeric(frame["game_id"], errors="coerce")
    return ids.map(lambda gid: mapping.get(int(gid)) if pd.notna(gid) and int(gid) in mapping else np.nan)


def _week_slate(frame: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    games = _fbs(frame)
    return games[
        (pd.to_numeric(games["season"], errors="coerce") == int(season))
        & (pd.to_numeric(games["week"], errors="coerce") == int(week))
        & games["completed"].astype(bool)
        & pd.to_numeric(games.get("margin"), errors="coerce").notna()
    ].copy()


def _base_margin(slate: pd.DataFrame, models: dict[str, Any]) -> np.ndarray:
    logged = logged_pred_margin(slate)
    packed = models["margin"]
    predicted = packed["model"].predict(model_matrix(slate, packed["features"]))
    out = np.asarray(predicted, dtype=float)
    logged_vals = pd.to_numeric(logged, errors="coerce").to_numpy(dtype=float)
    use = np.isfinite(logged_vals)
    out[use] = logged_vals[use]
    return out


def _ridge_layer() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("model", Ridge(alpha=25.0, fit_intercept=True)),
        ]
    )


def fit_week_layer(
    frame: pd.DataFrame,
    season: int,
    week: int,
    models: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if models is None:
        from src.simulate import load_models

        models = load_models()
    slate = _week_slate(frame, season, week)
    if len(slate) < MIN_WEEK_GAMES:
        raise RuntimeError(f"Need {MIN_WEEK_GAMES} completed FBS games to learn week {season} w{week}")
    y = pd.to_numeric(slate["margin"], errors="coerce").to_numpy(dtype=float)
    base = _base_margin(slate, models)
    residual = y - base
    features = [col for col in WEEK_LAYER_FEATURES if col in slate.columns] or WEEK_LAYER_FEATURES
    matrix = model_matrix(slate, features)
    layer = _ridge_layer()
    layer.fit(matrix, residual)
    adjusted = base + layer.predict(matrix)
    report = {
        "season": int(season),
        "week": int(week),
        "n": int(len(slate)),
        "mae_base": round(float(mean_absolute_error(y, base)), 3),
        "mae_adjusted": round(float(mean_absolute_error(y, adjusted)), 3),
        "winner_accuracy": round(float(((y > 0) == (adjusted > 0)).mean()), 4),
        "mean_residual": round(float(np.nanmean(residual)), 3),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": layer, "features": features, "kind": "week_layer", **report}, _layer_path())
    state = load_state()
    applied = _week_tuples(state.get("applied_weeks"))
    pair = (int(season), int(week))
    if pair not in applied:
        applied.append(pair)
    state["applied_weeks"] = [list(item) for item in applied]
    state["latest_week"] = report
    state["updated_at"] = time.time()
    save_state(state)
    _merge_latest_into_metrics(report)
    return report


def _merge_latest_into_metrics(report: dict[str, Any]) -> None:
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        path.write_text(json.dumps({"latest_week": report}, indent=2, default=str))
        return
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        data = {}
    data.pop("holdout_season", None)
    data["latest_week"] = report
    path.write_text(json.dumps(data, indent=2, default=str))


def load_week_layer() -> dict[str, Any] | None:
    path = _layer_path()
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def apply_week_layer(slate: pd.DataFrame) -> pd.DataFrame:
    packed = load_week_layer()
    if packed is None or slate is None or slate.empty:
        return slate
    out = slate.copy()
    raw = pd.to_numeric(out.get("pred_margin_raw", out.get("pred_margin")), errors="coerce")
    features = packed.get("features") or WEEK_LAYER_FEATURES
    delta = packed["model"].predict(model_matrix(out, features))
    out["pred_margin_raw"] = raw
    out["pred_margin"] = raw.to_numpy(dtype=float) + np.asarray(delta, dtype=float)
    if "over_under" in out.columns:
        totals = pd.to_numeric(out["over_under"], errors="coerce")
        out["pred_home_points"] = totals / 2.0 + out["pred_margin"] / 2.0
        out["pred_away_points"] = totals / 2.0 - out["pred_margin"] / 2.0
    if "spread" in out.columns:
        spread = pd.to_numeric(out["spread"], errors="coerce")
        out["edge_vs_spread"] = out["pred_margin"] + spread
    return out


def tick_weekly(
    *,
    frame: pd.DataFrame | None = None,
    refresh: bool = True,
    models: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if refresh:
        try:
            from src.collect import refresh_live_games

            refresh_live_games()
        except Exception as exc:
            print(f"live score refresh skipped: {exc}")
    if frame is None:
        from src.simulate import get_feature_frame

        frame = get_feature_frame(force=True)
    if models is None:
        from src.simulate import load_models, models_ready

        state = load_state()
        if not models_ready() or not state.get("base_ready"):
            return {"status": "needs_base"}
        models = load_models()
    state = load_state()
    skip = set(_week_tuples(state.get("base_weeks"))) | set(_week_tuples(state.get("applied_weeks")))
    for season, week in completed_weeks(frame):
        if (season, week) in skip:
            continue
        slate = _week_slate(frame, season, week)
        if len(slate) < MIN_WEEK_GAMES:
            continue
        report = fit_week_layer(frame, season, week, models=models)
        return {"status": "learned", **report}
    return {"status": "skipped"}


def autotune_enabled() -> bool:
    return os.getenv("WEEKLY_AUTOTRAIN", "1").strip().lower() not in {"0", "false", "no"}


def start_weekly_poller() -> None:
    global _poller_thread
    if not autotune_enabled():
        return
    if _poller_thread is not None and _poller_thread.is_alive():
        return
    _poller_stop.clear()
    interval = max(30, int(os.getenv("WEEKLY_POLL_SECONDS", "900")))

    def _loop() -> None:
        while True:
            try:
                result = tick_weekly()
                if result.get("status") == "needs_base":
                    from src.jobs import JobConflict, start_job

                    try:
                        start_job("train")
                    except JobConflict:
                        pass
                elif result.get("status") == "learned":
                    from src.simulate import invalidate_cache

                    invalidate_cache()
            except Exception as exc:
                print(f"weekly tick failed: {exc}")
            if _poller_stop.wait(interval):
                break

    _poller_thread = threading.Thread(target=_loop, name="weekly-autotune", daemon=True)
    _poller_thread.start()


def stop_weekly_poller() -> None:
    _poller_stop.set()
