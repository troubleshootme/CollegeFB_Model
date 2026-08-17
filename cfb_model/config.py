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

SHRINKAGE_K = 4.0
ELO_MARGIN_SCALE = 28.0
DEFAULT_MARGIN_SIGMA = 16.0

API_KEY_ENV = "CFBD_API_KEY"


def api_key() -> str | None:
    key = os.environ.get(API_KEY_ENV, "").strip()
    return key or None


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
