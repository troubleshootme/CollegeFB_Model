from cfb_model.ingest.flatten import flatten_games, flatten_legacy_box, flatten_lines
from cfb_model.migrate import _reconstruct_lines
import pandas as pd


def test_flatten_games_keeps_id():
    rows = flatten_games(
        [
            {
                "id": 401111,
                "season": 2024,
                "week": 1,
                "seasonType": "regular",
                "startDate": "2024-08-31T19:00:00Z",
                "completed": True,
                "neutralSite": False,
                "conferenceGame": True,
                "homeId": 1,
                "homeTeam": "Michigan",
                "homeClassification": "fbs",
                "homePoints": 31,
                "awayId": 2,
                "awayTeam": "Ohio State",
                "awayClassification": "fbs",
                "awayPoints": 24,
                "homePregameElo": 1800,
                "awayPregameElo": 1750,
            }
        ]
    )
    assert rows[0]["id"] == 401111
    assert rows[0]["home_team"] == "Michigan"
    assert rows[0]["home_points"] == 31


def test_flatten_lines_one_row_per_provider():
    rows = flatten_lines(
        [
            {
                "id": 9,
                "lines": [
                    {"provider": "Bovada", "spread": -7.5, "spreadOpen": -6.5, "overUnder": 50},
                    {"provider": "consensus", "spread": -7.0},
                ],
            }
        ]
    )
    assert len(rows) == 2
    assert {r["provider"] for r in rows} == {"Bovada", "consensus"}
    assert rows[0]["game_id"] == 9


def test_legacy_box_parses_text_stats():
    row = flatten_legacy_box(
        {
            "game_id": 1,
            "team_name": "Michigan",
            "home_away": "home",
            "thirdDownEff": "5-12",
            "fourthDownEff": "1-2",
            "possessionTime": "35:42",
            "completionAttempts": "18-30",
            "totalYards": 412,
        }
    )
    assert row["third_down_conv"] == 5
    assert row["third_down_att"] == 12
    assert row["possession_seconds"] == 35 * 60 + 42
    assert row["completions"] == 18


def test_reconstruct_orphan_bet_rows():
    bets = pd.DataFrame(
        [
            {
                "game_id": 55,
                "season": 2024,
                "week": 1,
                "home_team": "A",
                "away_team": "B",
                "provider0": None,
                "spread0": None,
            },
            {
                "game_id": None,
                "season": None,
                "week": None,
                "home_team": None,
                "away_team": None,
                "provider0": "Bovada",
                "spread0": -3.5,
                "over_under0": 48,
            },
        ]
    )
    id_map, lines = _reconstruct_lines(bets, 2024)
    assert id_map.iloc[0]["id"] == 55
    assert lines.iloc[0]["spread"] == -3.5
    assert lines.iloc[0]["game_id"] == 55
