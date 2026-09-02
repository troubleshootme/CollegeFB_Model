"""Shared synthetic warehouse for unit tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from cfb_model import store


def _ts(season: int, week: int) -> str:
    start = datetime(season, 8, 24, 19, 0, tzinfo=timezone.utc)
    return (start + timedelta(days=7 * (week - 1))).isoformat()


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "cfb.db"
    monkeypatch.setattr("cfb_model.config.DB_PATH", path)
    monkeypatch.setattr("cfb_model.config.MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr("cfb_model.config.DATA_DIR", tmp_path)
    monkeypatch.setattr("cfb_model.config.LEARNING_DIR", tmp_path / "learning")
    (tmp_path / "learning").mkdir(parents=True, exist_ok=True)
    conn = store.init_schema(store.connect(path))
    seed_warehouse(conn)
    conn.close()
    return path


def seed_warehouse(conn) -> None:
    teams = ["Alpha", "Bravo", "Charlie", "Delta"]
    team_rows = []
    for i, name in enumerate(teams, start=1):
        team_rows.append(
            {
                "id": i,
                "school": name,
                "conference": "Test",
                "classification": "fbs",
                "venue_id": i,
                "latitude": 30 + i,
                "longitude": -90 - i,
                "elevation": 100 * i,
                "capacity": 40000 + 10000 * i,
                "grass": i % 2,
                "dome": 0,
            }
        )
    store.replace_all(conn, "teams", pd.DataFrame(team_rows))
    venues = pd.DataFrame(
        {
            "id": [r["id"] for r in team_rows],
            "name": [r["school"] for r in team_rows],
            "city": ["X"] * 4,
            "state": ["TX"] * 4,
            "timezone": ["America/Chicago"] * 4,
            "latitude": [r["latitude"] for r in team_rows],
            "longitude": [r["longitude"] for r in team_rows],
            "elevation": [r["elevation"] for r in team_rows],
            "capacity": [r["capacity"] for r in team_rows],
            "grass": [r["grass"] for r in team_rows],
            "dome": [0] * 4,
        }
    )
    store.replace_all(conn, "venues", venues)

    games = []
    ppa = []
    lines = []
    talent = []
    gid = 1000
    strength = {"Alpha": 14, "Bravo": 4, "Charlie": -2, "Delta": -10}
    for season in (2022, 2023, 2024):
        for name, talent_pts in [("Alpha", 900), ("Bravo", 800), ("Charlie", 700), ("Delta", 600)]:
            talent.append({"year": season, "team": name, "talent": talent_pts + (season - 2022) * 5})
        for week in range(1, 9):
            pairs = [("Alpha", "Bravo"), ("Charlie", "Delta"), ("Alpha", "Charlie"), ("Bravo", "Delta")]
            home, away = pairs[(week - 1) % 4]
            if week % 2 == 0:
                home, away = away, home
            home_pts = 28 + strength[home] / 2 - strength[away] / 4 + (week % 3)
            away_pts = 21 + strength[away] / 2 - strength[home] / 4
            games.append(
                {
                    "id": gid,
                    "season": season,
                    "week": week,
                    "season_type": "regular",
                    "start_date": _ts(season, week),
                    "completed": 1,
                    "neutral_site": 0,
                    "conference_game": 1,
                    "venue_id": teams.index(home) + 1,
                    "venue": home,
                    "home_id": teams.index(home) + 1,
                    "home_team": home,
                    "home_conference": "Test",
                    "home_classification": "fbs",
                    "home_points": home_pts,
                    "home_pregame_elo": 1500 + strength[home] * 8,
                    "home_postgame_elo": 1500 + strength[home] * 8,
                    "home_postgame_wp": None,
                    "away_id": teams.index(away) + 1,
                    "away_team": away,
                    "away_conference": "Test",
                    "away_classification": "fbs",
                    "away_points": away_pts,
                    "away_pregame_elo": 1500 + strength[away] * 8,
                    "away_postgame_elo": 1500 + strength[away] * 8,
                    "away_postgame_wp": None,
                    "excitement_index": None,
                }
            )
            spread = -((home_pts - away_pts) / 2)
            lines.append(
                {
                    "game_id": gid,
                    "provider": "consensus",
                    "spread": spread,
                    "spread_open": spread + 0.5,
                    "over_under": 55,
                    "over_under_open": 54,
                    "home_moneyline": None,
                    "away_moneyline": None,
                }
            )
            for team, opp, off in (
                (home, away, 0.2 + strength[home] / 100),
                (away, home, 0.05 + strength[away] / 100),
            ):
                ppa.append(
                    {
                        "game_id": gid,
                        "season": season,
                        "week": week,
                        "season_type": "regular",
                        "team": team,
                        "conference": "Test",
                        "opponent": opp,
                        "off_overall": off,
                        "off_passing": off,
                        "off_rushing": off / 2,
                        "off_first_down": off,
                        "off_second_down": off,
                        "off_third_down": off,
                        "def_overall": -off / 2,
                        "def_passing": -off / 2,
                        "def_rushing": -off / 3,
                    }
                )
            gid += 1
    store.replace_all(conn, "games", pd.DataFrame(games))
    store.replace_all(conn, "lines", pd.DataFrame(lines))
    store.replace_all(conn, "ppa_games", pd.DataFrame(ppa))
    store.replace_all(conn, "talent", pd.DataFrame(talent))
