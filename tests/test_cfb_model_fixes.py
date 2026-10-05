"""Regression tests for bugs found in the cfb_model review."""

from __future__ import annotations

import datetime as dt
import json
import math

import numpy as np
import pandas as pd
import pytest

from cfb_model.players import attach_quarterbacks
from cfb_model.util import haversine_miles


def test_haversine_missing_coordinates_is_none_not_half_the_planet():
    assert haversine_miles(float("nan"), -90.0, 30.0, -90.0) is None
    assert haversine_miles(None, -90.0, 30.0, -90.0) is None
    assert haversine_miles(30.0, -90.0, 30.0, -90.0) == pytest.approx(0.0)


def test_cached_call_survives_datetimes_and_truncated_cache(tmp_path, monkeypatch):
    pytest.importorskip("cfbd")
    from cfb_model.client import CfbdClient

    client = CfbdClient.__new__(CfbdClient)
    client.cache_dir = tmp_path
    calls = []

    def fetch():
        calls.append(1)
        return {"start_date": dt.datetime(2026, 9, 5, 19, 0)}

    first = client.cached_call("games", fetch)
    assert "2026-09-05" in str(first["start_date"])
    path = client._cache_path("games")
    assert json.loads(path.read_text())["start_date"]
    path.write_text('{"start_da')  # a crash mid-write used to poison every later run
    client.cached_call("games", fetch)
    assert len(calls) == 2


def _qb_profiles():
    rows = []
    for team, name, ppa in (("Alpha", "Starter", 0.30), ("Alpha", "Backup", 0.02)):
        rows.append(
            {"season": 2026, "team": team, "qb_rank": 1 if name == "Starter" else 2, "player_key": f"name:{name.lower()}",
             "qb_name": name, "qb_style": "pocket passer", "qb_ppa_overall": ppa, "qb_dynamic": 0.5,
             "qb_punch": 0.1, "qb_pass_ypa": 8.0, "qb_qb_play_rush_share": 0.1, "qb_seasons_prior": 2}
        )
    rows.append({**rows[0], "team": "Bravo", "player_key": "name:bravo", "qb_name": "Bravo QB", "qb_rank": 1})
    return pd.DataFrame(rows)


def test_injury_report_does_not_leak_into_games_already_played():
    games = pd.DataFrame(
        {
            "id": [1, 2],
            "season": [2026, 2026],
            "home_team": ["Alpha", "Alpha"],
            "away_team": ["Bravo", "Bravo"],
            "start_date": ["2026-09-05T19:00:00Z", "2026-10-10T19:00:00Z"],
            "completed": [1, 0],
        }
    )
    injuries = pd.DataFrame(
        [{"team": "Alpha", "player": "Starter", "status": "Out", "as_of": "2026-10-06T12:00:00Z"}]
    )
    out = attach_quarterbacks(games, _qb_profiles(), injuries=injuries).sort_values("id")
    assert out["home_qb_starter_p"].tolist()[0] == 1.0  # report postdates the played game
    assert out["home_qb_starter_p"].tolist()[1] == 0.0  # report precedes the upcoming game


def test_injury_name_match_requires_same_team():
    games = pd.DataFrame(
        {"id": [1], "season": [2026], "home_team": ["Alpha"], "away_team": ["Bravo"], "completed": [0]}
    )
    injuries = pd.DataFrame([{"team": "Zulu", "player": "Starter", "status": "Out"}])
    out = attach_quarterbacks(games, _qb_profiles(), injuries=injuries).iloc[0]
    assert out["home_qb_starter_p"] == 1.0
