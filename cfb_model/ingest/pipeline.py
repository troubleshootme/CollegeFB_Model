"""Pull CFBD endpoints into the local SQLite warehouse."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from cfb_model import config, store
from cfb_model.client import CfbdClient, classification_fbs, season_type
from cfb_model.ingest import flatten as F
from cfb_model.quota import (
    cache_key,
    live_seasons,
    stored_seasons,
    table_has_rows,
    weeks_to_fetch,
    years_to_ingest,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _save_year(conn, table: str, rows: list[dict], year_col: str, year: int) -> int:
    frame = _frame(rows)
    if frame.empty:
        return 0
    return store.replace_rows(conn, table, frame, year_col=year_col, year=year)


def run_ingest(
    start_year: int | None = None,
    end_year: int | None = None,
    *,
    include_weather: bool | None = None,
    include_injuries: bool = True,
    full: bool | None = None,
    use_cache: bool = True,
    force: bool = False,
    client: CfbdClient | None = None,
) -> dict[str, int]:
    """Download 2016–current FBS data.

    Defaults to a Pro-tier (75k/month) pull: week-level havoc, box scores, Elo,
    plus CFBD weather / Open-Meteo and WEPA when the key allows it.
    Pass full=False for a year-level-only lite ingest.
    """
    start_year = start_year or config.START_YEAR
    end_year = end_year or config.END_YEAR
    if full is None:
        full = config.INGEST_FULL_DEFAULT
    if include_weather is None:
        include_weather = config.INGEST_WEATHER_DEFAULT
    own_client = client is None
    client = client or CfbdClient()
    client.require_key()
    info = client.user_info() or {}
    print(
        f"CFBD tier={info.get('tier_name')} remaining={info.get('remaining_calls')} "
        f"limit={info.get('monthly_limit')}"
    )
    counts: dict[str, int] = {}
    conn = store.init_schema()
    try:
        existing = stored_seasons(conn)
        years = years_to_ingest(start_year, end_year, existing, force=force)
        print(f"ingest years {years} (force={force})")
        if force or not table_has_rows(conn, "venues"):
            counts["venues"] = _ingest_venues(client, conn, use_cache)
        else:
            counts["venues"] = 0
        counts["teams"] = _ingest_teams(client, conn, end_year, use_cache)
        counts["coaches"] = _ingest_coaches(client, conn, start_year, end_year, use_cache)
        for year in years:
            print(f"ingest year {year}")
            year_counts = _ingest_year(client, conn, year, use_cache=use_cache, full=full, force=force)
            for key, value in year_counts.items():
                counts[key] = counts.get(key, 0) + value
        if include_weather:
            from cfb_model.weather import ingest_weather

            counts["weather"] = ingest_weather(conn, client=client, use_cache=use_cache)
        if include_injuries:
            from cfb_model.injuries import ingest_injuries

            try:
                counts["injury_reports"] = ingest_injuries(conn)
            except Exception as exc:
                print(f"injuries skipped: {exc}")
                counts["injury_reports"] = 0
        store.set_meta(conn, "last_ingest", datetime.now(timezone.utc).isoformat())
        store.set_meta(conn, "start_year", str(start_year))
        store.set_meta(conn, "end_year", str(end_year))
    finally:
        conn.close()
        if own_client:
            client.close()
    return counts


def _ingest_venues(client: CfbdClient, conn, use_cache: bool) -> int:
    payload = client.cached_call("venues", lambda: client.venues.get_venues(), use_cache)
    rows = F.flatten_venues(payload)
    return store.replace_all(conn, "venues", _frame(rows))


def _ingest_teams(client: CfbdClient, conn, year: int, use_cache: bool) -> int:
    payload = client.cached_call(
        f"teams_fbs_{year}",
        lambda: client.teams.get_fbs_teams(year=year),
        use_cache,
    )
    rows = F.flatten_teams(payload)
    return store.replace_all(conn, "teams", _frame(rows))


def _ingest_coaches(client: CfbdClient, conn, start: int, end: int, use_cache: bool) -> int:
    payload = client.cached_call(
        f"coaches_{start}_{end}",
        lambda: client.coaches.get_coaches(min_year=start, max_year=end),
        use_cache,
    )
    rows = F.flatten_coaches(payload)
    return store.replace_all(conn, "coaches_seasons", _frame(rows))


def _ingest_year(client: CfbdClient, conn, year: int, use_cache: bool, full: bool = False, force: bool = False) -> dict[str, int]:
    fbs = classification_fbs()
    counts: dict[str, int] = {}
    live = year in live_seasons()

    games: list[dict] = []
    for kind in ("regular", "postseason"):
        st = season_type(kind)
        payload = client.cached_call(
            cache_key(f"games_{year}_{kind}_fbs", year),
            lambda st=st: client.games.get_games(year=year, season_type=st, classification=fbs),
            use_cache,
        )
        games.extend(F.flatten_games(payload))
    counts["games"] = _save_year(conn, "games", games, "season", year)

    lines = []
    for kind in ("regular", "postseason"):
        st = season_type(kind)
        payload = client.cached_call(
            f"lines_{year}_{kind}",
            lambda st=st: client.betting.get_lines(year=year, season_type=st),
            use_cache,
        )
        lines.extend(F.flatten_lines(payload))
    # lines table has no season column; delete via games of this year
    if lines:
        ids = tuple({row["game_id"] for row in lines})
        # replace all lines for this year's games
        year_ids = [g["id"] for g in games]
        if year_ids:
            conn.executemany("DELETE FROM lines WHERE game_id = ?", [(i,) for i in year_ids])
            conn.commit()
        counts["lines"] = store.replace_rows(conn, "lines", _frame(lines))
    else:
        counts["lines"] = 0

    ppa = []
    for kind in ("regular", "postseason"):
        st = season_type(kind)
        payload = client.cached_call(
            f"ppa_games_{year}_{kind}",
            lambda st=st: client.metrics.get_predicted_points_added_by_game(
                year=year,
                season_type=st,
                exclude_garbage_time=True,
                classification=fbs,
            ),
            use_cache,
        )
        ppa.extend(F.flatten_ppa(payload))
    counts["ppa_games"] = _save_year(conn, "ppa_games", ppa, "season", year)

    advanced = []
    for kind in ("regular", "postseason"):
        st = season_type(kind)
        payload = client.cached_call(
            f"adv_games_{year}_{kind}",
            lambda st=st: client.stats.get_advanced_game_stats(
                year=year, season_type=st, exclude_garbage_time=True
            ),
            use_cache,
        )
        advanced.extend(F.flatten_advanced(payload))
    counts["advanced_game_stats"] = _save_year(conn, "advanced_game_stats", advanced, "season", year)

    weeks = sorted({g.get("week") for g in games if g.get("week") is not None})
    if live and not force:
        weeks = weeks_to_fetch(games) or weeks
    havoc: list[dict] = []
    try:
        payload = client.cached_call(
            f"havoc_{year}",
            lambda: client.stats.get_game_havoc_stats(year=year),
            use_cache,
        )
        havoc.extend(F.flatten_havoc(payload))
    except Exception as exc:
        print(f"  havoc year-level skipped: {exc}")
    if full and not havoc:
        for week in weeks:
            try:
                payload = client.cached_call(
                    f"havoc_{year}_{week}",
                    lambda week=week: client.stats.get_game_havoc_stats(year=year, week=week),
                    use_cache,
                )
                havoc.extend(F.flatten_havoc(payload))
            except Exception as exc:
                print(f"  havoc week {week} skipped: {exc}")
                break
    if havoc:
        conn.execute(
            "DELETE FROM havoc_games WHERE game_id IN (SELECT id FROM games WHERE season = ?)",
            (year,),
        )
        conn.commit()
        counts["havoc_games"] = store.replace_rows(conn, "havoc_games", _frame(havoc))
    else:
        counts["havoc_games"] = 0

    box_rows: list[dict] = []
    if full:
        for week in weeks:
            try:
                payload = client.cached_call(
                    f"team_stats_{year}_{week}",
                    lambda week=week: client.games.get_game_team_stats(
                        year=year, week=week, classification=fbs
                    ),
                    use_cache,
                )
                box_rows.extend(F.flatten_team_game_stats(payload))
            except Exception as exc:
                print(f"  team stats week {week} skipped: {exc}")
                break
    if box_rows:
        conn.execute(
            "DELETE FROM team_game_stats WHERE game_id IN (SELECT id FROM games WHERE season = ?)",
            (year,),
        )
        conn.commit()
        counts["team_game_stats"] = store.replace_rows(conn, "team_game_stats", _frame(box_rows))
    else:
        counts["team_game_stats"] = 0

    elo_rows: list[dict] = []
    if full:
        for week in weeks:
            try:
                payload = client.cached_call(
                    f"elo_{year}_{week}",
                    lambda week=week: client.ratings.get_elo(
                        year=year, week=week, season_type=season_type("regular")
                    ),
                    use_cache,
                )
                elo_rows.extend(F.flatten_elo(payload, year=year, week=week, season_type="regular"))
            except Exception as exc:
                print(f"  elo week {week} skipped: {exc}")
                break
    counts["elo_weekly"] = _save_year(conn, "elo_weekly", elo_rows, "year", year)

    try:
        core = client.cached_call(f"core_{year}", lambda: client.ratings.get_core(year=year), use_cache)
        counts["core_ratings"] = _save_year(conn, "core_ratings", F.flatten_core(core), "year", year)
    except Exception as exc:
        print(f"  CORE skipped: {exc}")
        counts["core_ratings"] = 0

    try:
        sp = client.cached_call(f"sp_{year}", lambda: client.ratings.get_sp(year=year), use_cache)
        counts["sp_ratings"] = _save_year(conn, "sp_ratings", F.flatten_sp(sp), "year", year)
    except Exception as exc:
        print(f"  SP+ skipped: {exc}")
        counts["sp_ratings"] = 0

    talent = client.cached_call(f"talent_{year}", lambda: client.teams.get_talent(year=year), use_cache)
    counts["talent"] = _save_year(conn, "talent", F.flatten_talent(talent), "year", year)

    recruiting = client.cached_call(
        f"recruiting_{year}",
        lambda: client.recruiting.get_team_recruiting_rankings(year=year),
        use_cache,
    )
    counts["recruiting_teams"] = _save_year(
        conn, "recruiting_teams", F.flatten_recruiting(recruiting), "year", year
    )

    returning = client.cached_call(
        f"returning_{year}",
        lambda: client.players.get_returning_production(year=year),
        use_cache,
    )
    counts["returning_production"] = _save_year(
        conn, "returning_production", F.flatten_returning(returning), "season", year
    )

    wp = []
    for kind in ("regular", "postseason"):
        st = season_type(kind)
        payload = client.cached_call(
            f"pregame_wp_{year}_{kind}",
            lambda st=st: client.metrics.get_pregame_win_probabilities(year=year, season_type=st),
            use_cache,
        )
        wp.extend(F.flatten_pregame_wp(payload))
    if wp:
        conn.execute(
            "DELETE FROM pregame_wp WHERE game_id IN (SELECT id FROM games WHERE season = ?)",
            (year,),
        )
        conn.commit()
        counts["pregame_wp"] = store.replace_rows(conn, "pregame_wp", _frame(wp))
    else:
        counts["pregame_wp"] = 0

    try:
        wepa = client.cached_call(
            cache_key(f"wepa_{year}", year),
            lambda: client.adjusted.get_adjusted_team_season_stats(year=year),
            use_cache,
        )
        counts["wepa_season"] = _save_year(conn, "wepa_season", F.flatten_wepa(wepa), "year", year)
    except Exception as exc:
        print(f"  WEPA skipped: {exc}")
        counts["wepa_season"] = 0

    counts.update(_ingest_player_year(client, conn, year, use_cache))

    remaining = client.remaining_calls()
    if remaining is not None:
        print(f"  remaining API calls: {remaining}")
    return counts


def _try_api(obj, names: list[str], **kwargs):
    last = None
    for name in names:
        fn = getattr(obj, name, None)
        if fn is None:
            continue
        try:
            return fn(**kwargs)
        except TypeError as exc:
            last = exc
            continue
        except Exception as exc:
            last = exc
            continue
    if last:
        raise last
    raise AttributeError(f"none of {names} on {type(obj)}")


def _ingest_player_year(client: CfbdClient, conn, year: int, use_cache: bool) -> dict[str, int]:
    """Year-level QB identity sources: season stats, player PPA, portal, FPI."""
    counts: dict[str, int] = {}
    try:
        payload = client.cached_call(
            cache_key(f"player_stats_{year}", year),
            lambda: _try_api(
                client.stats,
                ["get_player_season_stats", "get_season_player_stats"],
                year=year,
            ),
            use_cache,
        )
        rows = F.flatten_player_season_stats(payload)
        if not rows:
            for category in ("passing", "rushing"):
                extra = client.cached_call(
                    cache_key(f"player_stats_{year}_{category}", year),
                    lambda category=category: _try_api(
                        client.stats,
                        ["get_player_season_stats", "get_season_player_stats"],
                        year=year,
                        category=category,
                    ),
                    use_cache,
                )
                rows.extend(F.flatten_player_season_stats(extra))
        counts["player_season_stats"] = _save_year(conn, "player_season_stats", rows, "season", year)
    except Exception as exc:
        print(f"  player season stats skipped: {exc}")
        counts["player_season_stats"] = 0

    try:
        payload = client.cached_call(
            cache_key(f"player_ppa_{year}", year),
            lambda: _try_api(
                client.metrics,
                [
                    "get_predicted_points_added_by_player",
                    "get_player_ppa",
                    "get_predicted_points_added_players",
                ],
                year=year,
            ),
            use_cache,
        )
        counts["player_ppa"] = _save_year(conn, "player_ppa", F.flatten_player_ppa(payload), "season", year)
    except Exception as exc:
        print(f"  player PPA skipped: {exc}")
        counts["player_ppa"] = 0

    try:
        payload = client.cached_call(
            cache_key(f"portal_{year}", year),
            lambda: _try_api(client.players, ["get_transfer_portal", "get_portal"], year=year),
            use_cache,
        )
        counts["transfer_portal"] = _save_year(conn, "transfer_portal", F.flatten_portal(payload), "season", year)
    except Exception as exc:
        print(f"  portal skipped: {exc}")
        counts["transfer_portal"] = 0

    try:
        payload = client.cached_call(
            cache_key(f"fpi_{year}", year),
            lambda: _try_api(client.ratings, ["get_fpi", "get_fpi_ratings"], year=year),
            use_cache,
        )
        counts["fpi_ratings"] = _save_year(conn, "fpi_ratings", F.flatten_fpi(payload), "year", year)
    except Exception as exc:
        print(f"  FPI skipped: {exc}")
        counts["fpi_ratings"] = 0
    return counts
