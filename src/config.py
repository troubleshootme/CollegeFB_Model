from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "collegeFootball.db"
MODELS_DIR = ROOT / "models"
ENV_PATH = ROOT / ".env"
TEAMS_CACHE_PATH = ROOT / "data" / "fbs_teams_cache.json"

START_YEAR = 2013
END_YEAR = 2026
SEASON_TYPES = ("regular", "postseason")
WEEKS = range(1, 17)
API_BASE = "https://api.collegefootballdata.com"
HOST_PORT = 38417
CORS_ORIGINS = [
    "https://cfb2pdf.com",
    "http://cfb2pdf.com",
    "http://localhost:47392",
    "http://127.0.0.1:47392",
]

# HTTP to api.collegefootballdata.com is forbidden unless a collect/ingest/retrain
# job sets this. The API/serve process must leave it unset.
CFBD_ALLOW_CALLS_ENV = "CFBD_ALLOW_CALLS"
_TRUTHY = {"1", "true", "yes", "on"}


class CfbdNotAllowed(RuntimeError):
    """Serve-path code tried to call CFBD. Set CFBD_ALLOW_CALLS=1 only on train jobs."""


class CfbdQuotaExhausted(RuntimeError):
    """CFBD remainingCalls is 0; refuse further HTTP so the shared monthly pool is not hammered."""


def load_env(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def cfbd_calls_allowed() -> bool:
    load_env()
    return os.getenv(CFBD_ALLOW_CALLS_ENV, "").strip().lower() in _TRUTHY


def enable_cfbd_calls() -> None:
    """Mark this process as a train/collect job. Never call from the API/serve path."""
    os.environ[CFBD_ALLOW_CALLS_ENV] = "1"


def require_cfbd_allowed() -> None:
    if not cfbd_calls_allowed():
        raise CfbdNotAllowed(
            "CFBD HTTP is disabled outside collect/ingest/retrain. "
            f"Set {CFBD_ALLOW_CALLS_ENV}=1 only for those jobs — never in the API process."
        )


def api_key() -> str:
    load_env()
    key = os.getenv("CFBD_API_KEY") or os.getenv("API_KEY")
    if not key:
        raise RuntimeError("Set CFBD_API_KEY in .env")
    return key
