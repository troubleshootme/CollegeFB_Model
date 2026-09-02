from __future__ import annotations

import pandas as pd

from cfb_model.models import annotate_board
from cfb_model.sim import ratings_from_row, simulate_game, simulate_matchup


def _row(**overrides) -> pd.Series:
    base = {
        "home_team": "Florida State",
        "away_team": "SMU",
        "venue": "Doak Campbell Stadium",
        "week": 1,
        "season": 2026,
        "pred_margin": 1.0,
        "close_spread": 3.0,
        "close_total": 53.0,
        "home_off_ppa_std": 0.12,
        "away_off_ppa_std": 0.18,
        "home_def_ppa_std": -0.04,
        "away_def_ppa_std": -0.08,
        "home_off_success_std": 0.44,
        "away_off_success_std": 0.46,
        "home_def_success_std": 0.41,
        "away_def_success_std": 0.39,
        "home_off_expl_std": 1.15,
        "away_off_expl_std": 1.22,
        "home_havoc_std": 0.14,
        "away_havoc_std": 0.16,
        "home_off_rushing_ppa_std": 0.08,
        "away_off_rushing_ppa_std": 0.06,
        "home_off_passing_ppa_std": 0.14,
        "away_off_passing_ppa_std": 0.20,
        "home_games_played": 0,
        "away_games_played": 0,
        "precipitation": 0.0,
        "wind_speed": 8.0,
        "temperature": 28.0,
        "dome": 0,
    }
    base.update(overrides)
    return pd.Series(base)


def test_simulate_game_is_deterministic_with_seed():
    home, away = ratings_from_row(_row())
    a = simulate_game(home, away, seed=7, pred_margin=1, total=53)
    b = simulate_game(home, away, seed=7, pred_margin=1, total=53)
    assert a["home"]["total"] == b["home"]["total"]
    assert a["away"]["total"] == b["away"]["total"]
    assert [p["description"] for p in a["plays"]] == [p["description"] for p in b["plays"]]


def test_four_quarters_sum_to_final_and_box_matches_plays():
    home, away = ratings_from_row(_row())
    game = simulate_game(home, away, seed=11, pred_margin=3, total=55)
    for side in ("home", "away"):
        box = game[side]
        scored = box["q1"] + box["q2"] + box["q3"] + box["q4"] + box["ot"]
        assert scored == box["total"]
        assert box["plays"] >= 20
        assert box["third_att"] >= box["third_conv"]
        rush_from_plays = sum(
            p["yards"]
            for p in game["plays"]
            if p["play_type"] == "rush" and p["possession"] == box["name"]
        )
        assert box["rush_yds"] == rush_from_plays
        pass_from_plays = sum(
            p["yards"]
            for p in game["plays"]
            if p["play_type"] in {"pass", "sack"} and p["possession"] == box["name"]
        )
        assert box["pass_yds"] == pass_from_plays


def test_play_by_play_has_clock_down_and_score():
    home, away = ratings_from_row(_row())
    game = simulate_game(home, away, seed=3, pred_margin=-3, total=50)
    assert len(game["plays"]) >= 40
    assert len(game["drives"]) >= 8
    snap = game["plays"][5]
    assert snap["quarter"] in {1, 2, 3, 4, 5}
    assert ":" in snap["clock"]
    assert snap["down"] in {0, 1, 2, 3, 4}
    assert "home_score" in snap and "away_score" in snap
    last = game["plays"][-1]
    assert last["home_score"] == game["home"]["total"]
    assert last["away_score"] == game["away"]["total"]


def test_ensemble_wp_tracks_large_home_edge():
    row = _row(pred_margin=21.0, close_spread=-17.5, close_total=58.0)
    result = simulate_matchup(row, n_sims=80, seed=42)
    assert 0.7 <= result["win_prob"] <= 1.0
    assert result["gamebook"]["home"]["total"] >= 0
    assert "box" in result["gamebook"] or "rush_att" in result["gamebook"]["home"]
    assert result["cover_side"] in {"Florida State", "SMU", None}


def test_annotate_board_prints_favorite_line_and_dog_cover():
    frame = pd.DataFrame(
        {
            "home_team": ["Florida State"],
            "away_team": ["SMU"],
            "pred_margin": [1.0],
            "pred_home_wp": [0.52],
            "close_spread": [3.0],
            "start_date": [pd.Timestamp("2026-09-07")],
        }
    )
    out = annotate_board(frame).iloc[0]
    assert out["spread_label"] == "SMU -3"
    assert out["pick"] == "Florida State"
    assert out["cover_side"] == "Florida State"
