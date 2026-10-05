from __future__ import annotations

import sqlite3
import threading
import time
from datetime import date, timedelta

import pandas as pd
import requests

from src.config import DB_PATH

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_VARS = "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,weather_code"
_RATE_LOCK = threading.Lock()
_NEXT_REQUEST = {"archive": 0.0, "forecast": 0.0}
ARCHIVE_INTERVAL_SEC = 6.5
FORECAST_INTERVAL_SEC = 0.4


def _at(seq: list | None, i: int):
    if not seq or i >= len(seq):
        return None
    return seq[i]


def _parse_daily(venue_id: int, payload: dict) -> list[dict]:
    daily = payload.get("daily") or {}
    times = daily.get("time") or []
    rows = []
    for i, day in enumerate(times):
        rows.append(
            {
                "venue_id": int(venue_id),
                "date": day,
                "temp_max_c": _at(daily.get("temperature_2m_max"), i),
                "temp_min_c": _at(daily.get("temperature_2m_min"), i),
                "precip_mm": _at(daily.get("precipitation_sum"), i),
                "wind_kmh": _at(daily.get("wind_speed_10m_max"), i),
                "weather_code": _at(daily.get("weather_code"), i),
            }
        )
    return rows


def _throttle(kind: str) -> None:
    interval = ARCHIVE_INTERVAL_SEC if kind == "archive" else FORECAST_INTERVAL_SEC
    with _RATE_LOCK:
        now = time.time()
        wait = _NEXT_REQUEST[kind] - now
        _NEXT_REQUEST[kind] = max(now, _NEXT_REQUEST[kind]) + interval
    if wait > 0:
        time.sleep(wait)


class DailyQuotaExceeded(RuntimeError):
    pass


def _get(url: str, params: dict, kind: str, attempts: int = 6) -> dict | None:
    last_err = None
    for attempt in range(attempts):
        _throttle(kind)
        try:
            response = requests.get(url, params=params, timeout=90)
            if response.status_code == 429:
                last_err = f"429 {response.text[:120]}"
                if "Daily" in response.text:
                    raise DailyQuotaExceeded(last_err)
                print(f"    {kind} rate limited; sleeping 75s", flush=True)
                time.sleep(75)
                continue
            if not response.ok:
                last_err = f"{response.status_code} {response.text[:160]}"
                time.sleep(2 * (attempt + 1))
                continue
            return response.json()
        except DailyQuotaExceeded:
            raise
        except requests.RequestException as exc:
            last_err = str(exc)
            time.sleep(2 * (attempt + 1))
    if last_err:
        print(f"    {kind} failed {params.get('latitude')},{params.get('longitude')}: {last_err}", flush=True)
    return None


def _season_windows(archive_end: str) -> list[tuple[str, str]]:
    """Aug-Jan football windows. Long ranges are weighted ~days/28 against Open-Meteo's daily cap."""
    windows = []
    end = date.fromisoformat(archive_end)
    for year in range(2018, end.year + 1):
        start = date(year, 8, 1)
        stop = date(year + 1, 1, 20)
        if start > end:
            continue
        windows.append((start.isoformat(), min(stop, end).isoformat()))
    return windows


def _venue_weather(venue_id: int, lat: float, lon: float, archive_end: str, need_archive: bool) -> list[dict]:
    rows: list[dict] = []
    if need_archive:
        for start, end in _season_windows(archive_end):
            payload = _get(
                ARCHIVE_URL,
                {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": start,
                    "end_date": end,
                    "daily": DAILY_VARS,
                    "timezone": "auto",
                },
                "archive",
            )
            if payload:
                rows.extend(_parse_daily(venue_id, payload))
    forecast = _get(
        FORECAST_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "daily": DAILY_VARS,
            "forecast_days": 16,
            "timezone": "auto",
        },
        "forecast",
    )
    if forecast:
        rows.extend(_parse_daily(venue_id, forecast))
    return rows


def stadium_locations(con: sqlite3.Connection) -> pd.DataFrame:
    venues = pd.read_sql("SELECT venue_id, latitude, longitude FROM venues", con)
    used = pd.read_sql("SELECT DISTINCT venue_id FROM games WHERE venue_id IS NOT NULL", con)
    frame = used.merge(venues, on="venue_id", how="left")
    missing = frame["latitude"].isna() | frame["longitude"].isna()
    if missing.any():
        teams = pd.read_sql(
            "SELECT venue_id, latitude, longitude FROM fbs_teams WHERE latitude IS NOT NULL",
            con,
        ).drop_duplicates("venue_id")
        frame = frame.merge(teams, on="venue_id", how="left", suffixes=("", "_team"))
        frame["latitude"] = frame["latitude"].fillna(frame["latitude_team"])
        frame["longitude"] = frame["longitude"].fillna(frame["longitude_team"])
    return frame.dropna(subset=["latitude", "longitude"]).drop_duplicates("venue_id")


def _existing_weather(con: sqlite3.Connection) -> pd.DataFrame:
    try:
        return pd.read_sql("SELECT * FROM weather_daily", con)
    except sqlite3.Error:
        return pd.DataFrame()


def _write_weather(frame: pd.DataFrame) -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.execute("PRAGMA journal_mode=WAL")
        frame.to_sql("_weather_daily_new", con, if_exists="replace", index=False)
        con.execute("DROP TABLE IF EXISTS weather_daily")
        con.execute("ALTER TABLE _weather_daily_new RENAME TO weather_daily")
        con.execute("CREATE INDEX IF NOT EXISTS ix_weather_venue_date ON weather_daily(venue_id, date)")
        con.commit()


def _merged(existing: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    fresh = pd.DataFrame(rows)
    parts = [df for df in (existing, fresh) if not df.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).drop_duplicates(["venue_id", "date"], keep="last")


def fetch_weather() -> None:
    archive_end = (date.today() - timedelta(days=1)).isoformat()
    with sqlite3.connect(DB_PATH) as con:
        locations = stadium_locations(con)
        existing = _existing_weather(con)
    complete: set[int] = set()
    if not existing.empty:
        coverage = existing.groupby("venue_id")["date"].agg(["min", "count"])
        complete = set(
            coverage[(coverage["min"] <= "2018-08-15") & (coverage["count"] >= 1000)].index.astype(int)
        )
    print(
        f"Fetching Open-Meteo weather for {len(locations)} stadiums "
        f"({len(complete)} already complete) through {archive_end} + 16-day forecast",
        flush=True,
    )
    rows: list[dict] = []
    jobs = [
        (int(row.venue_id), float(row.latitude), float(row.longitude), int(row.venue_id) not in complete)
        for row in locations.itertuples()
    ]
    archive_ok = True
    done = 0
    for vid, lat, lon, need_archive in jobs:
        try:
            part = _venue_weather(vid, lat, lon, archive_end, need_archive and archive_ok)
        except DailyQuotaExceeded:
            archive_ok = False
            print("  Open-Meteo daily cap hit; remaining venues forecast-only", flush=True)
            try:
                part = _venue_weather(vid, lat, lon, archive_end, False)
            except DailyQuotaExceeded:
                part = []
        except Exception as exc:
            print(f"    venue {vid} crashed: {exc}", flush=True)
            part = []
        rows.extend(part)
        done += 1
        if done == 1 or done % 15 == 0 or done == len(jobs):
            print(f"  weather {done}/{len(jobs)}", flush=True)
        if done % 25 == 0:
            frame = _merged(existing, rows)
            _write_weather(frame)
            print(f"  checkpoint {len(frame):,} rows", flush=True)
    frame = _merged(existing, rows)
    if frame.empty:
        print("  no weather rows returned", flush=True)
        return
    _write_weather(frame)
    print(f"  wrote weather_daily: {len(frame):,} rows across {frame['venue_id'].nunique()} venues", flush=True)


if __name__ == "__main__":
    fetch_weather()
