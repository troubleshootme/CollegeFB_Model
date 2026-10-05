"""Historical weather from Open-Meteo using stadium coordinates."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import requests

from cfb_model import store
from cfb_model.client import CfbdClient
from cfb_model.util import as_float, getv

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def ingest_weather(
    conn,
    *,
    client: CfbdClient | None = None,
    use_cache: bool = True,
    timeout: int = 30,
) -> int:
    """Prefer CFBD /games/weather when the key has the feature; else Open-Meteo."""
    if client is not None and client.has_feature("weather"):
        try:
            return _from_cfbd(conn, client, use_cache)
        except Exception as exc:
            print(f"CFBD weather failed ({exc}); falling back to Open-Meteo")
    return _from_open_meteo(conn, timeout=timeout)


def _from_cfbd(conn, client: CfbdClient, use_cache: bool) -> int:
    games = store.read_table(conn, "games")
    if games.empty:
        return 0
    rows = []
    for year in sorted(games["season"].dropna().unique()):
        payload = client.cached_call(
            f"cfbd_weather_{int(year)}",
            lambda year=int(year): client.games.get_weather(year=int(year)),
            use_cache,
        )
        for item in payload or []:
            game_id = getv(item, "game_id") or getv(item, "id")
            if game_id is None:
                continue
            rows.append(
                {
                    "game_id": int(game_id),
                    "temperature": as_float(getv(item, "temperature")),
                    "precipitation": as_float(getv(item, "precipitation")),
                    "wind_speed": as_float(getv(item, "wind_speed")),
                    "indoor": 1 if getv(item, "game_indoors") else 0,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return 0
    return store.replace_all(conn, "weather", frame)


def _from_open_meteo(conn, timeout: int = 30) -> int:
    games = store.read_table(conn, "games")
    venues = store.read_table(conn, "venues")
    teams = store.read_table(conn, "teams")
    if games.empty:
        return 0
    venue_lookup = venues.set_index("id") if not venues.empty and "id" in venues else pd.DataFrame()
    team_lookup = teams.set_index("id") if not teams.empty and "id" in teams else pd.DataFrame()
    existing = store.read_table(conn, "weather")
    have = set(existing["game_id"].tolist()) if not existing.empty else set()
    rows = []
    session = requests.Session()
    for game in games.itertuples(index=False):
        if game.id in have:
            continue
        dome = _lookup(venue_lookup, game.venue_id, "dome") or _lookup(team_lookup, game.home_id, "dome")
        lat = _lookup(venue_lookup, game.venue_id, "latitude") or _lookup(team_lookup, game.home_id, "latitude")
        lon = _lookup(venue_lookup, game.venue_id, "longitude") or _lookup(team_lookup, game.home_id, "longitude")
        if dome:
            rows.append(
                {"game_id": game.id, "temperature": 70.0, "precipitation": 0.0, "wind_speed": 0.0, "indoor": 1}
            )
            continue
        date = _game_date(game.start_date)
        if lat is None or lon is None or date is None:
            continue
        daily = _fetch_daily(session, float(lat), float(lon), date, timeout=timeout)
        if daily is None:
            continue
        rows.append(
            {
                "game_id": game.id,
                "temperature": daily.get("temperature"),
                "precipitation": daily.get("precipitation"),
                "wind_speed": daily.get("wind_speed"),
                "indoor": 0,
            }
        )
    if not rows:
        return 0
    return store.replace_rows(conn, "weather", pd.DataFrame(rows))


def _lookup(frame: pd.DataFrame, key, column: str):
    if frame is None or frame.empty or key is None or key not in frame.index:
        return None
    try:
        value = frame.loc[key, column]
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        return value if pd.notna(value) else None
    except Exception:
        return None


def _game_date(start_date) -> str | None:
    if start_date is None or (isinstance(start_date, float) and pd.isna(start_date)):
        return None
    text = str(start_date)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return text[:10] if len(text) >= 10 else None


def _fetch_daily(session: requests.Session, lat: float, lon: float, date: str, timeout: int) -> dict | None:
    params = {
        "latitude": round(lat, 3),
        "longitude": round(lon, 3),
        "start_date": date,
        "end_date": date,
        "daily": "temperature_2m_mean,precipitation_sum,wind_speed_10m_max",
        "timezone": "auto",
    }
    try:
        response = session.get(ARCHIVE_URL, params=params, timeout=timeout)
        if response.status_code >= 400:
            response = session.get(FORECAST_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        daily = payload.get("daily") or {}
        temps = daily.get("temperature_2m_mean") or []
        precips = daily.get("precipitation_sum") or []
        winds = daily.get("wind_speed_10m_max") or []
        return {
            "temperature": temps[0] if temps else None,
            "precipitation": precips[0] if precips else None,
            "wind_speed": winds[0] if winds else None,
        }
    except Exception:
        return None
