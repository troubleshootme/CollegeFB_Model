from __future__ import annotations

import random
import threading
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import API_BASE, CfbdQuotaExhausted, api_key, require_cfbd_allowed

_LOCAL = threading.local()
_REMAINING: dict[str, int | None] = {"value": None}

_REMAINING_HEADERS = (
    "remainingcalls",
    "remaining-calls",
    "x-remaining-calls",
    "x-calls-remaining",
    "x-ratelimit-remaining",
)


def remaining_calls() -> int | None:
    return _REMAINING["value"]


def set_remaining_calls(value: int | None) -> None:
    _REMAINING["value"] = value


def _parse_remaining(value: object) -> int | None:
    if value is None:
        return None
    try:
        remaining = int(float(value))
    except (TypeError, ValueError):
        return None
    return remaining


def _record_remaining(response: requests.Response, payload: object = None) -> None:
    headers = {str(key).lower(): value for key, value in response.headers.items()}
    for name in _REMAINING_HEADERS:
        parsed = _parse_remaining(headers.get(name))
        if parsed is not None:
            set_remaining_calls(parsed)
            return
    if isinstance(payload, dict):
        for key in ("remainingCalls", "remaining_calls"):
            parsed = _parse_remaining(payload.get(key))
            if parsed is not None:
                set_remaining_calls(parsed)
                return


def _guard_quota() -> None:
    require_cfbd_allowed()
    remaining = remaining_calls()
    if remaining is not None and remaining <= 0:
        raise CfbdQuotaExhausted(
            "CFBD remainingCalls is 0; refusing further requests so the shared monthly quota is not exhausted."
        )


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
    _guard_quota()
    url = f"{API_BASE}{path}"
    last_error: Exception | None = None
    for attempt in range(attempts):
        _guard_quota()
        try:
            response = _session().get(url, params=params, timeout=90)
            payload: list | dict | None = None
            if response.content:
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
            _record_remaining(response, payload)
            if remaining_calls() is not None and remaining_calls() <= 0 and path.rstrip("/") != "/info":
                # Recorded 0 after this response: allow the payload through, next call dies.
                pass
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
            return payload if payload is not None else []
        except CfbdQuotaExhausted:
            raise
        except Exception as exc:  # noqa: BLE001 - retry transient API failures
            last_error = exc
            time.sleep(2 ** attempt + random.random())
    raise RuntimeError(f"Failed GET {path} {params}: {last_error}")


def ping_quota() -> int | None:
    """Read remainingCalls from /info. Train jobs should call this once before a pull."""
    payload = get_json("/info")
    if isinstance(payload, dict):
        remaining = _parse_remaining(payload.get("remainingCalls") or payload.get("remaining_calls"))
        if remaining is not None:
            set_remaining_calls(remaining)
    _guard_quota()
    return remaining_calls()
