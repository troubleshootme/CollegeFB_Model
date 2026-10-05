from __future__ import annotations

import pandas as pd

from src.schedule_sync import apply_schedule_results, team_key


def test_team_key_normalizes_accents_and_punctuation():
    assert team_key("San José State") == team_key("San Jose State")
    assert team_key("Hawai'i") == team_key("Hawaii")


def test_apply_schedule_results_writes_finals_without_touching_other_games():
    games = pd.DataFrame(
        [
            {
                "game_id": 1,
                "season": 2026,
                "week": 1,
                "home_team": "Rutgers",
                "away_team": "Massachusetts",
                "completed": False,
                "home_points": None,
                "away_points": None,
            },
            {
                "game_id": 2,
                "season": 2026,
                "week": 1,
                "home_team": "Ohio State",
                "away_team": "Ball State",
                "completed": False,
                "home_points": None,
                "away_points": None,
            },
        ]
    )
    results = [
        {
            "season": 2026,
            "week": 1,
            "home_team": "Rutgers",
            "away_team": "Massachusetts",
            "home_score": 21,
            "away_score": 37,
            "status": "completed",
        },
        {
            "season": 2026,
            "week": 1,
            "home_team": "Ohio State",
            "away_team": "Ball State",
            "home_score": 0,
            "away_score": 0,
            "status": "scheduled",
        },
    ]
    out, updated = apply_schedule_results(games, results)
    assert updated == 1
    rutgers = out.loc[out["game_id"] == 1].iloc[0]
    assert bool(rutgers["completed"]) is True
    assert int(rutgers["home_points"]) == 21
    assert int(rutgers["away_points"]) == 37
    ohio = out.loc[out["game_id"] == 2].iloc[0]
    assert bool(ohio["completed"]) is False
    assert pd.isna(ohio["home_points"])
