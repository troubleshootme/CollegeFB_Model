from __future__ import annotations

import random
import threading
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import API_BASE, api_key

_LOCAL = threading.local()


def _session() -> requests.Session:
    session = getattr(_LOCAL, "session", None)
    if session is not None:
        return session
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(
        {
            "Authorization": f"Bearer {api_key()}",
            "Accept": "application/json",
        }
    )
    _LOCAL.session = session
    return session


def get_json(path: str, params: dict[str, Any] | None = None, attempts: int = 6) -> list | dict:
    url = f"{API_BASE}{path}"
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = _session().get(url, params=params, timeout=90)
            if response.status_code == 404:
                return []
            if response.status_code == 400:
                print(f"  CFBD 400 {path} {params}", flush=True)
                return []
            if response.status_code == 429:
                time.sleep(2 ** attempt + random.random())
                continue
            response.raise_for_status()
            if not response.content:
                return []
            return response.json()
        except Exception as exc:  # noqa: BLE001 - retry transient API failures
            last_error = exc
            time.sleep(2 ** attempt + random.random())
    raise RuntimeError(f"Failed GET {path} {params}: {last_error}")
