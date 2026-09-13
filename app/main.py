from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas import JobStartRequest, MatchupRequest
from src.config import CORS_ORIGINS, END_YEAR, MODELS_DIR, load_env
from src.jobs import JobConflict, get_job, list_jobs, start_job
from src.weekly import start_weekly_poller, stop_weekly_poller
from src.simulate import (
    get_feature_frame,
    load_models,
    models_ready,
    prediction_payload,
    simulate_matchup,
)
from src.teams import load_team_directory as _load_team_directory


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_env()
    try:
        _load_team_directory(refresh_if_empty=True)
    except Exception:
        pass
    start_weekly_poller()
    yield
    stop_weekly_poller()


app = FastAPI(
    title="College Football Matchup API",
    description="Predict scheduled and hypothetical FBS matchups. Built for cfb2pdf / cfbschedule.",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)


def _cors_origins() -> list[str]:
    extra = os.getenv("CORS_ORIGINS", "")
    origins = list(CORS_ORIGINS)
    if extra:
        origins.extend(part.strip() for part in extra.split(",") if part.strip())
    origins.append("http://localhost:38417")
    origins.append("http://127.0.0.1:38417")
    return list(dict.fromkeys(origins))


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)


def reset_state() -> None:
    """Test hook."""
    load_env()


def load_team_directory() -> dict[str, dict[str, Any]]:
    return _load_team_directory(refresh_if_empty=True)


def _configured_api_key() -> str | None:
    load_env()
    key = os.getenv("MODEL_API_KEY")
    if key is None:
        return None
    key = key.strip()
    return key or None


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    path = request.url.path
    if request.method == "OPTIONS" or path in {
        "/api/health",
        "/api/docs",
        "/api/redoc",
        "/api/openapi.json",
    }:
        return await call_next(request)
    key = _configured_api_key()
    if key and request.headers.get("x-api-key") != key:
        return JSONResponse({"detail": "Invalid or missing API key"}, status_code=401)
    return await call_next(request)


def _require_models() -> None:
    if not models_ready():
        raise HTTPException(status_code=503, detail="Train the model before scoring matchups.")


def score_week(season: int, week: int) -> pd.DataFrame:
    load_models()
    frame = get_feature_frame()
    slate = frame[
        frame["fbs_vs_fbs"].astype(bool)
        & (frame["season"] == int(season))
        & (frame["week"] == int(week))
    ].copy()
    upcoming = slate[~slate["completed"].astype(bool)] if not slate.empty else slate
    use = upcoming if not upcoming.empty else slate
    if use.empty:
        return use
    from src.simulate import score_slate

    return score_slate(use.sort_values("start_date"))


def _default_week() -> tuple[int, int]:
    if not models_ready():
        return END_YEAR, 1
    try:
        frame = get_feature_frame()
    except Exception:
        return END_YEAR, 1
    upcoming = frame[~frame["completed"].astype(bool) & frame["fbs_vs_fbs"].astype(bool)]
    if upcoming.empty:
        return END_YEAR, 1
    latest = int(upcoming["season"].max())
    season_games = upcoming[upcoming["season"] == latest]
    first = season_games.sort_values(["week", "start_date"]).iloc[0]
    return int(first["season"]), int(first["week"])


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "models_ready": models_ready(),
        "service": "cfb-matchup",
    }


@app.get("/api/teams")
def teams() -> dict[str, Any]:
    directory = load_team_directory()
    rows = sorted(directory.values(), key=lambda t: (t.get("conference") or "", t.get("name") or ""))
    return {"teams": rows}


@app.get("/api/weeks")
def weeks(season: int | None = Query(default=None)) -> dict[str, Any]:
    try:
        frame = get_feature_frame()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    games = frame[frame["fbs_vs_fbs"].astype(bool)].copy()
    if season is not None:
        games = games[games["season"] == int(season)]
    if games.empty:
        return {"season": season or END_YEAR, "weeks": []}
    grouped = (
        games.groupby(["season", "week"], as_index=False)
        .agg(games=("game_id", "nunique"), upcoming=("completed", lambda s: int((~s.astype(bool)).sum())))
        .sort_values(["season", "week"])
    )
    return {
        "season": season or int(grouped["season"].max()),
        "weeks": grouped.to_dict(orient="records"),
    }


@app.get("/api/predictions")
def predictions(
    season: int | None = Query(default=None),
    week: int | None = Query(default=None),
) -> dict[str, Any]:
    _require_models()
    if season is None or week is None:
        default_season, default_week = _default_week()
        season = default_season if season is None else season
        week = default_week if week is None else week
    try:
        board = score_week(season, week)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    directory = load_team_directory()
    games = [prediction_payload(row, directory) for _, row in board.iterrows()]
    return {"season": season, "week": week, "games": games}


@app.get("/api/predictions/{game_id}")
def prediction_detail(game_id: int) -> dict[str, Any]:
    _require_models()
    try:
        board = simulate_matchup("ignored", "ignored", game_id=game_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    directory = load_team_directory()
    return prediction_payload(board.iloc[0], directory)


@app.post("/api/matchups")
def create_matchup(body: MatchupRequest) -> dict[str, Any]:
    _require_models()
    payload = body.model_dump(exclude_unset=True)
    home = payload.pop("home_team")
    away = payload.pop("away_team")
    try:
        board = simulate_matchup(home, away, **payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    directory = load_team_directory()
    return prediction_payload(board.iloc[0], directory)


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        return {"ready": False, "metrics": None}
    return {"ready": True, "metrics": json.loads(path.read_text())}


@app.get("/api/metrics/importance")
def feature_importance() -> dict[str, Any]:
    path = MODELS_DIR / "feature_importance.csv"
    if not path.exists():
        return {"features": []}
    frame = pd.read_csv(path)
    return {"features": frame.head(40).to_dict(orient="records")}


@app.get("/api/jobs")
def jobs() -> dict[str, Any]:
    return {"jobs": list_jobs()}


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str) -> dict[str, Any]:
    try:
        return get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown job") from exc


@app.post("/api/jobs/{kind}")
def create_job(kind: str, body: JobStartRequest | None = None) -> dict[str, Any]:
    if kind not in {"collect", "weather", "train", "pipeline"}:
        raise HTTPException(status_code=404, detail="Unknown job kind")
    try:
        return start_job(kind)
    except JobConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
