"""CFBD quota guards: never re-pull frozen seasons, never week-loop the future."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from cfb_model import store


def live_seasons(today: date | None = None) -> set[int]:
    today = today or date.today()
    if today.month == 1:
        return {today.year - 1, today.year}
    return {today.year}


def stored_seasons(conn) -> set[int]:
    if not store.table_exists(conn, "games"):
        return set()
    rows = conn.execute("SELECT DISTINCT season FROM games WHERE season IS NOT NULL").fetchall()
    return {int(season) for (season,) in rows if season is not None}


def table_has_rows(conn, name: str) -> bool:
    if not store.table_exists(conn, name):
        return False
    return conn.execute(f'SELECT 1 FROM "{name}" LIMIT 1').fetchone() is not None


def years_to_ingest(
    start_year: int,
    end_year: int,
    existing: set[int],
    *,
    today: date | None = None,
    force: bool = False,
) -> list[int]:
    """Historical seasons already in SQLite are skipped. Live season always refreshes."""
    if force:
        return list(range(start_year, end_year + 1))
    live = live_seasons(today)
    return [year for year in range(start_year, end_year + 1) if year in live or year not in existing]


def cache_key(base: str, year: int, *, today: date | None = None) -> str:
    """Live-season cache expires daily so new weeks land without re-pulling 2016."""
    today = today or date.today()
    if year in live_seasons(today):
        return f"{base}_{today.isoformat()}"
    return base


def weeks_to_fetch(games: list[dict], *, today: date | None = None) -> list[int]:
    """Only completed weeks plus games kicking off within 8 days. Future empty weeks are free quota."""
    today = today or date.today()
    horizon = datetime(today.year, today.month, today.day, tzinfo=timezone.utc) + timedelta(days=8)
    weeks: set[int] = set()
    for game in games:
        week = game.get("week")
        if week is None:
            continue
        if game.get("completed"):
            weeks.add(int(week))
            continue
        start = _as_dt(game.get("start_date"))
        if start is not None and start <= horizon:
            weeks.add(int(week))
    return sorted(weeks)


def _as_dt(value) -> datetime | None:
    if value is None:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed
