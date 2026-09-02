"""Injury snapshots are as-of kickoff and drop ESPN's stale leftover rows."""

from __future__ import annotations

import pandas as pd

from cfb_model.injuries import attach_injuries, flatten_espn_injuries, map_espn_team
from cfb_model.players import position_weight


def _payload():
    return {
        "timestamp": "2026-09-02T12:00:00Z",
        "season": {"year": 2026},
        "injuries": [
            {
                "id": "52",
                "displayName": "Florida State Seminoles",
                "injuries": [
                    {
                        "status": "Out",
                        "date": "2026-08-30T12:00Z",
                        "shortComment": "Starter (ankle) is out.",
                        "athlete": {
                            "displayName": "Jordan Dual",
                            "id": "99",
                            "position": {"abbreviation": "QB"},
                        },
                    }
                ],
            },
            {
                "id": "57",
                "displayName": "Florida Gators",
                "injuries": [
                    {
                        "status": "Out",
                        "date": "2020-11-21T18:31Z",
                        "shortComment": "stale leftover",
                        "athlete": {
                            "displayName": "Old Player",
                            "position": {"abbreviation": "WR"},
                        },
                    }
                ],
            },
        ],
    }


def test_flatten_drops_stale_and_keeps_current_qb():
    rows = flatten_espn_injuries(_payload(), as_of="2026-09-02T12:00:00Z", season=2026, schools=["Florida State", "Florida"])
    names = {r["player"] for r in rows}
    assert "Jordan Dual" in names
    assert "Old Player" not in names
    qb = next(r for r in rows if r["player"] == "Jordan Dual")
    assert qb["team"] == "Florida State"
    assert qb["position_weight"] == position_weight("QB")
    assert qb["load"] >= 1.0


def test_map_espn_mascot_to_cfbd_school():
    assert map_espn_team("Florida Gators", ["Florida", "Florida State"]) == "Florida"
    assert map_espn_team("Florida State Seminoles", ["Florida", "Florida State"]) == "Florida State"


def test_attach_uses_snapshot_before_kickoff_only():
    reports = pd.DataFrame(
        [
            {
                "snapshot_id": "early",
                "as_of": "2026-09-01T10:00:00+00:00",
                "season": 2026,
                "team": "Alpha",
                "player": "Jordan Dual",
                "player_id": "1",
                "position": "QB",
                "status": "Out",
                "load": 1.2,
                "team_raw": "Alpha",
            },
            {
                "snapshot_id": "late",
                "as_of": "2026-09-07T10:00:00+00:00",
                "season": 2026,
                "team": "Alpha",
                "player": "Jordan Dual",
                "player_id": "1",
                "position": "QB",
                "status": "Out",
                "load": 9.9,
                "team_raw": "Alpha",
            },
        ]
    )
    games = pd.DataFrame(
        {
            "id": [1],
            "home_team": ["Alpha"],
            "away_team": ["Bravo"],
            "start_date": ["2026-09-06T19:00:00+00:00"],
        }
    )
    out = attach_injuries(games, reports).iloc[0]
    assert abs(out["home_injury_load"] - 1.2) < 1e-9
    assert out["home_injury_load"] != 9.9
    assert out["home_qb_injury"] > 0
