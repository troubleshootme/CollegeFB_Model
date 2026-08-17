"""CFBD v2 client with disk cache, retries, and /info feature detection."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import cfbd
from cfbd.exceptions import ApiException

from cfb_model import config
from cfb_model.util import to_plain

T = TypeVar("T")


class CfbdClient:
    def __init__(self, api_key: str | None = None, cache_dir: Path | None = None) -> None:
        self.api_key = api_key if api_key is not None else config.api_key()
        self.cache_dir = cache_dir or config.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._user_info: dict[str, Any] | None = None
        self._configuration = cfbd.Configuration(access_token=self.api_key or "")
        self._client = cfbd.ApiClient(self._configuration)
        self.games = cfbd.GamesApi(self._client)
        self.betting = cfbd.BettingApi(self._client)
        self.metrics = cfbd.MetricsApi(self._client)
        self.ratings = cfbd.RatingsApi(self._client)
        self.stats = cfbd.StatsApi(self._client)
        self.teams = cfbd.TeamsApi(self._client)
        self.players = cfbd.PlayersApi(self._client)
        self.coaches = cfbd.CoachesApi(self._client)
        self.venues = cfbd.VenuesApi(self._client)
        self.recruiting = cfbd.RecruitingApi(self._client)
        self.adjusted = cfbd.AdjustedMetricsApi(self._client)
        self.info_api = cfbd.InfoApi(self._client)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def __enter__(self) -> "CfbdClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def require_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                f"Set {config.API_KEY_ENV} in the environment or a .env file "
                "(see .env.example). Get a key at https://collegefootballdata.com/key"
            )

    def user_info(self, refresh: bool = False) -> dict[str, Any] | None:
        if self._user_info is not None and not refresh:
            return self._user_info
        if not self.api_key:
            return None
        try:
            info = self._retry(lambda: self.info_api.get_user_info())
            self._user_info = to_plain(info) or {}
        except Exception as exc:
            print(f"warning: GET /info failed ({exc})")
            self._user_info = {}
        return self._user_info

    def has_feature(self, name: str) -> bool:
        info = self.user_info() or {}
        features = info.get("features") or {}
        if isinstance(features, dict) and features:
            return bool(features.get(name) or features.get(name.replace("-", "_")))
        # Unknown feature map: try premium endpoints on large quotas (Pro / 75k).
        remaining = info.get("remaining_calls")
        limit = info.get("monthly_limit")
        try:
            if remaining is not None and float(remaining) >= 5000:
                return True
            if limit is not None and float(limit) >= 30000:
                return True
        except (TypeError, ValueError):
            pass
        return False

    def remaining_calls(self) -> int | None:
        info = self.user_info() or {}
        remaining = info.get("remaining_calls")
        try:
            return int(remaining) if remaining is not None else None
        except (TypeError, ValueError):
            return None

    def cached_call(self, cache_key: str, fn: Callable[[], T], use_cache: bool = True) -> Any:
        path = self._cache_path(cache_key)
        if use_cache and path.exists():
            with path.open() as handle:
                return json.load(handle)
        result = self._retry(fn)
        payload = to_plain(result)
        if use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as handle:
                json.dump(payload, handle)
        return payload

    def _cache_path(self, cache_key: str) -> Path:
        digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:16]
        safe = re_safe(cache_key)
        return self.cache_dir / f"{safe}_{digest}.json"

    def _retry(self, fn: Callable[[], T], attempts: int = 5) -> T:
        delay = 2.0
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                return fn()
            except ApiException as exc:
                last_exc = exc
                status = getattr(exc, "status", None)
                if status in {429, 500, 502, 503, 504} and attempt < attempts - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
            except Exception as exc:
                last_exc = exc
                message = str(exc).lower()
                if any(token in message for token in ("429", "503", "timeout", "temporarily")) and attempt < attempts - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise last_exc or RuntimeError("CFBD request failed")


def re_safe(key: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_=" else "_" for ch in key)[:80]


def classification_fbs():
    from cfbd.models.division_classification import DivisionClassification

    return DivisionClassification.FBS


def season_type(name: str):
    from cfbd.models.season_type import SeasonType

    mapping = {
        "regular": SeasonType.REGULAR,
        "postseason": SeasonType.POSTSEASON,
        "both": SeasonType.BOTH,
    }
    return mapping[name]
