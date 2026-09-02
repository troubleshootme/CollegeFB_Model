"""Paths, season window, and environment configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = ROOT / "models"
LEARNING_DIR = Path(os.environ.get("CFB_LEARNING_DIR", DATA_DIR / "learning"))
PROFILES_DIR = Path(os.environ.get("CFB_PROFILES_DIR", DATA_DIR / "profiles"))
GAMEBOOKS_DIR = Path(os.environ.get("CFB_GAMEBOOKS_DIR", DATA_DIR / "gamebooks"))
DB_PATH = Path(os.environ.get("CFB_DB_PATH", DATA_DIR / "cfb.db"))
LEGACY_DB_PATH = ROOT / "collegeFootball.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

START_YEAR = int(os.environ.get("CFB_START_YEAR", "2016"))
END_YEAR = int(os.environ.get("CFB_END_YEAR", "2026"))
CURRENT_SEASON = int(os.environ.get("CFB_CURRENT_SEASON", "2026"))

# Default ingest assumes a CFBD Pro / Tier 3 key (~75k calls/month).
# A 2016–2026 full backfill is ~1–2k calls with disk cache.
INGEST_FULL_DEFAULT = True
INGEST_WEATHER_DEFAULT = True

# Form (how good they are this year): current-season results take over fast
# so the rest of the season can move. n=4 games → 67% this year, 33% last year.
SHRINKAGE_K_FORM = 2.0
# Tendencies (run/pass, explosiveness, havoc): stay glued to multi-year identity.
# n=4 → 25% this year; n=12 → 50/50.
SHRINKAGE_K_TENDENCY = 12.0
SHRINKAGE_K = SHRINKAGE_K_FORM
TENDENCY_YEARS = 3
COACH_YEARS = 6
COACH_TENDENCY_K = SHRINKAGE_K_TENDENCY
COACH_AGGRESSION_K = 10.0
QB_YEARS = 3
QB_TENDENCY_K = 6.0
BLOWOUT_LEAD = 21
EARLY_GAMES = 4  # in-season identity is still forming before this many games
ELO_MARGIN_SCALE = 28.0
DEFAULT_MARGIN_SIGMA = 16.0
# A starting QB is ~6.5 points of home margin vs a replacement-level backup.
# Dual-threat QBs cost more when they sit (the run threat vanishes too).
QB_OUT_POINTS = 6.5
QB_DYNAMIC_INJURY_BONUS = 0.45

API_KEY_ENV = "CFBD_API_KEY"


def api_key() -> str | None:
    key = os.environ.get(API_KEY_ENV, "").strip()
    return key or None


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LEARNING_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    GAMEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
