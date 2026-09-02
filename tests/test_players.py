"""QB identity: follow the person, position weights, dual-threat injury cost."""

from __future__ import annotations

import pandas as pd

from cfb_model.players import (
    apply_portal,
    attach_quarterbacks,
    build_qb_season_table,
    classify_qb,
    dynamic_score,
    player_injury_load,
    position_weight,
    qb_out_points,
)


def test_quarterback_outweighs_every_other_position():
    assert position_weight("QB") == 1.0
    assert position_weight("QB") > position_weight("LT") > position_weight("WR")
    assert position_weight("WR") > position_weight("CB") > position_weight("K")
    assert position_weight("K") > position_weight("P")


def test_dynamic_qb_classified_dual_threat():
    dyn = dynamic_score(rush_att=140, pass_att=280, rush_yards=850, games=12, ppa_rush=0.18)
    assert dyn >= 0.55
    assert classify_qb(dyn, pass_ypa=8.2, int_rate=0.02, rush_ypg=70) == "dual-threat"
    pocket = dynamic_score(rush_att=20, pass_att=400, rush_yards=40, games=12, ppa_rush=0.0)
    assert classify_qb(pocket, pass_ypa=8.0, int_rate=0.018, rush_ypg=3) == "pocket passer"


def test_dual_threat_out_costs_more_than_pocket():
    pocket = qb_out_points(0.05, "Out")
    dual = qb_out_points(0.80, "Out")
    assert dual > pocket
    assert dual >= pocket * 1.3
    assert player_injury_load("QB", "Out", dynamic=0.8) > player_injury_load("WR", "Out")


def test_portal_follows_the_qb_to_the_new_school():
    leaders = pd.DataFrame(
        {
            "season": [2023, 2023],
            "team": ["Air Raid U", "Air Raid U"],
            "player": ["Alex Dual", "Pat Backup"],
            "player_key": ["name:alex dual", "name:pat backup"],
            "position": ["QB", "QB"],
            "pass_att": [380, 40],
            "qb_rank": [1, 2],
        }
    )
    portal = pd.DataFrame(
        {
            "season": [2024],
            "player": ["Alex Dual"],
            "position": ["QB"],
            "origin": ["Air Raid U"],
            "destination": ["Ground U"],
        }
    )
    moved = apply_portal(leaders, portal)
    dest = moved[(moved["season"] == 2024) & (moved["team"] == "Ground U")]
    assert not dest.empty
    assert dest.iloc[0]["player"] == "Alex Dual"
    origin = moved[(moved["season"] == 2024) & (moved["team"] == "Air Raid U") & (moved["player"] == "Alex Dual")]
    assert origin.empty


def test_build_qb_table_uses_prior_season_only():
    stats = pd.DataFrame(
        [
            {
                "season": 2022,
                "player": "Alex Dual",
                "position": "QB",
                "team": "Air Raid U",
                "category": "passing",
                "attempts": 300,
                "completions": 190,
                "yards": 2800,
                "touchdowns": 24,
                "interceptions": 8,
                "games": 12,
            },
            {
                "season": 2022,
                "player": "Alex Dual",
                "position": "QB",
                "team": "Air Raid U",
                "category": "rushing",
                "attempts": 120,
                "yards": 780,
                "touchdowns": 10,
                "games": 12,
            },
            {
                "season": 2023,
                "player": "Alex Dual",
                "position": "QB",
                "team": "Air Raid U",
                "category": "passing",
                "attempts": 320,
                "completions": 200,
                "yards": 3100,
                "touchdowns": 28,
                "interceptions": 7,
                "games": 13,
            },
            {
                "season": 2023,
                "player": "Alex Dual",
                "position": "QB",
                "team": "Air Raid U",
                "category": "rushing",
                "attempts": 130,
                "yards": 900,
                "touchdowns": 12,
                "games": 13,
            },
        ]
    )
    table = build_qb_season_table(stats)
    row_2023 = table[(table["season"] == 2023) & (table["qb_rank"] == 1)].iloc[0]
    # 2023 profile is 2022 stats, not the 900-yard 2023 season.
    assert row_2023["qb_name"] == "Alex Dual"
    assert row_2023["qb_rush_ypg"] < 80  # 780/12 = 65 from 2022
    assert row_2023["qb_style"] == "dual-threat"


def test_injury_blends_backup_when_starter_is_out():
    profiles = pd.DataFrame(
        [
            {
                "season": 2024,
                "team": "Alpha",
                "qb_rank": 1,
                "player_key": "name:starter",
                "qb_name": "Starter Star",
                "qb_style": "dual-threat",
                "qb_ppa_overall": 0.30,
                "qb_dynamic": 0.7,
                "qb_punch": 0.8,
                "qb_pass_ypa": 9.0,
                "qb_qb_play_rush_share": 0.25,
                "qb_seasons_prior": 2,
            },
            {
                "season": 2024,
                "team": "Alpha",
                "qb_rank": 2,
                "player_key": "name:backup",
                "qb_name": "Backup",
                "qb_style": "game manager",
                "qb_ppa_overall": 0.02,
                "qb_dynamic": 0.1,
                "qb_punch": -0.2,
                "qb_pass_ypa": 6.5,
                "qb_qb_play_rush_share": 0.05,
                "qb_seasons_prior": 1,
            },
            {
                "season": 2024,
                "team": "Bravo",
                "qb_rank": 1,
                "player_key": "name:bravo",
                "qb_name": "Bravo QB",
                "qb_style": "pocket passer",
                "qb_ppa_overall": 0.10,
                "qb_dynamic": 0.1,
                "qb_punch": 0.0,
                "qb_pass_ypa": 7.5,
                "qb_qb_play_rush_share": 0.08,
                "qb_seasons_prior": 2,
            },
        ]
    )
    games = pd.DataFrame(
        {
            "id": [1],
            "season": [2024],
            "home_team": ["Alpha"],
            "away_team": ["Bravo"],
            "completed": [0],
        }
    )
    injuries = pd.DataFrame(
        [{"team": "Alpha", "player": "Starter Star", "status": "Out", "position": "QB", "load": 1.0}]
    )
    out = attach_quarterbacks(games, profiles, injuries=injuries).iloc[0]
    assert out["home_qb"] == "Starter Star"
    assert out["home_qb_starter_p"] == 0.0
    assert out["home_qb_out"] == 1.0
    assert out["home_qb_out_points"] > 6
    # Blended toward backup PPA
    assert out["home_qb_ppa_overall"] < 0.15
