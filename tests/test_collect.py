from datetime import date

import pandas as pd

from src import collect as collect_mod


def test_live_seasons_only_current_year_after_january():
    assert collect_mod.live_seasons(date(2026, 9, 2)) == {2026}
    assert collect_mod.live_seasons(date(2026, 2, 1)) == {2026}
    assert collect_mod.live_seasons(date(2026, 8, 1)) == {2026}


def test_live_seasons_keeps_prior_year_open_in_january_for_postseason():
    assert collect_mod.live_seasons(date(2026, 1, 15)) == {2025, 2026}


def test_seasons_to_fetch_skips_stored_historical_years():
    existing = set(range(2013, 2027))
    years = collect_mod.seasons_to_fetch(2013, 2026, existing, today=date(2026, 9, 2))
    assert years == [2026]


def test_seasons_to_fetch_backfills_missing_historical_years():
    existing = {2013, 2014, 2016, 2026}
    years = collect_mod.seasons_to_fetch(2013, 2026, existing, today=date(2026, 9, 2))
    assert 2015 in years
    assert 2025 in years
    assert 2013 not in years
    assert 2026 in years


def test_seasons_to_fetch_empty_database_collects_full_range():
    years = collect_mod.seasons_to_fetch(2020, 2022, set(), today=date(2026, 9, 2))
    assert years == [2020, 2021, 2022]


def test_merge_year_frames_replaces_only_years_that_arrived():
    existing = pd.DataFrame({"season": [2024, 2025, 2026], "game_id": [1, 2, 3]})
    incoming = pd.DataFrame({"season": [2026, 2026], "game_id": [30, 31]})
    merged = collect_mod.merge_year_frames(existing, incoming, "season", [2026])
    assert sorted(merged["game_id"].tolist()) == [1, 2, 30, 31]


def test_merge_year_frames_keeps_existing_when_fetch_returns_empty():
    existing = pd.DataFrame({"season": [2025, 2026], "game_id": [1, 2]})
    merged = collect_mod.merge_year_frames(existing, pd.DataFrame(), "season", [2026])
    assert merged["game_id"].tolist() == [1, 2]


def test_collect_does_not_call_cfbd_for_stored_past_seasons(tmp_path, monkeypatch):
    db = tmp_path / "collegeFootball.db"
    monkeypatch.setattr(collect_mod, "DB_PATH", db)

    existing_games = pd.DataFrame(
        {
            "game_id": [10, 20, 30],
            "season": [2024, 2025, 2026],
            "week": [1, 1, 1],
            "home_team": ["A", "B", "C"],
        }
    )
    existing_stats = pd.DataFrame({"game_id": [10, 20, 30], "season": [2024, 2025, 2026], "team": ["A", "B", "C"]})
    existing_lines = pd.DataFrame({"game_id": [10, 20, 30], "season": [2024, 2025, 2026], "spread": [-3, -7, -1]})
    existing_teams = pd.DataFrame({"year": [2024, 2025, 2026], "team": ["A", "B", "C"], "team_id": [1, 2, 3]})

    import sqlite3

    with sqlite3.connect(db) as con:
        existing_games.to_sql("games", con, index=False)
        existing_stats.to_sql("team_game_stats", con, index=False)
        existing_lines.to_sql("betting_lines", con, index=False)
        existing_teams.to_sql("fbs_teams", con, index=False)
        pd.DataFrame({"venue_id": [1], "name": ["Bryant-Denny"]}).to_sql("venues", con, index=False)

    year_calls: list[tuple[str, int]] = []
    venues_calls = []

    def fake_games(year: int) -> pd.DataFrame:
        year_calls.append(("games", year))
        return pd.DataFrame(
            {"game_id": [300 + year], "season": [year], "week": [2], "home_team": [f"new-{year}"]}
        )

    def track(name: str):
        def inner(year: int) -> pd.DataFrame:
            year_calls.append((name, year))
            return pd.DataFrame()

        return inner

    monkeypatch.setattr(collect_mod, "fetch_games", fake_games)
    monkeypatch.setattr(collect_mod, "fetch_team_stats", track("team_stats"))
    monkeypatch.setattr(collect_mod, "fetch_lines", track("lines"))
    monkeypatch.setattr(collect_mod, "fetch_fbs_teams", track("fbs_teams"))
    monkeypatch.setattr(collect_mod, "fetch_venues", lambda: venues_calls.append("venues") or pd.DataFrame())
    monkeypatch.setattr(collect_mod, "fetch_talent", track("talent"))
    monkeypatch.setattr(collect_mod, "fetch_sp", track("sp"))
    monkeypatch.setattr(collect_mod, "fetch_recruiting", track("recruiting"))
    monkeypatch.setattr(collect_mod, "fetch_returning", track("returning"))
    monkeypatch.setattr(collect_mod, "fetch_coaches", track("coaches"))
    monkeypatch.setattr(collect_mod, "fetch_ppa", track("ppa"))
    monkeypatch.setattr(collect_mod, "fetch_advanced_season", track("advanced"))
    monkeypatch.setattr(collect_mod, "date", type("D", (), {"today": staticmethod(lambda: date(2026, 9, 2))}))

    collect_mod.collect(2024, 2026)

    assert year_calls == [
        ("games", 2026),
        ("team_stats", 2026),
        ("lines", 2026),
        ("fbs_teams", 2026),
        ("talent", 2026),
        ("sp", 2026),
        ("recruiting", 2026),
        ("returning", 2026),
        ("coaches", 2026),
        ("ppa", 2026),
        ("advanced", 2026),
    ]
    assert venues_calls == []
    with sqlite3.connect(db) as con:
        games = pd.read_sql("SELECT game_id, season FROM games ORDER BY season, game_id", con)
    assert games["game_id"].tolist() == [10, 20, 300 + 2026]
