from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "collegeFootball.db"
MODELS_DIR = ROOT / "models"
ENV_PATH = ROOT / ".env"

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


def load_env(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def api_key() -> str:
    load_env()
    key = os.getenv("CFBD_API_KEY") or os.getenv("API_KEY")
    if not key:
        raise RuntimeError("Set CFBD_API_KEY in .env")
    return key
