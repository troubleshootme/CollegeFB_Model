from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLS, haversine_miles
from src.simulate import prediction_payload, simulate_matchup, team_snapshot


class _Reg:
    def predict(self, matrix):
        return np.full(len(matrix), 7.5)


class _Clf:
    def predict_proba(self, matrix):
        home = np.full(len(matrix), 0.72)
        return np.column_stack([1.0 - home, home])


def _models():
    return {
        "margin": {"model": _Reg(), "features": FEATURE_COLS},
        "win": {"model": _Clf(), "features": FEATURE_COLS},
        "ats": {"model": _Reg(), "features": FEATURE_COLS + ["spread"]},
    }


def _game(**overrides) -> dict:
    row = {col: np.nan for col in FEATURE_COLS}
    row.update(
        {
            "game_id": 401,
            "season": 2026,
            "week": 2,
            "completed": False,
            "fbs_vs_fbs": True,
            "start_date": pd.Timestamp("2026-09-12T16:00:00Z"),
            "home_team": "Alabama",
            "away_team": "Georgia",
            "home_id": 333,
            "away_id": 61,
            "home_conference": "SEC",
            "away_conference": "SEC",
            "conference_game": 1.0,
            "neutral_site": 0.0,
            "venue": "Bryant-Denny Stadium",
            "home_lat": 33.208,
            "home_lon": -87.550,
            "home_elev": 70.0,
            "away_lat": 33.950,
            "away_lon": -83.373,
            "away_elev": 180.0,
            "home_pregame_elo": 1900.0,
            "away_pregame_elo": 1850.0,
            "home_talent": 990.0,
            "away_talent": 980.0,
            "home_sp": 28.0,
            "away_sp": 26.0,
            "home_sp_off": 40.0,
            "away_sp_off": 38.0,
            "home_sp_def": 8.0,
            "away_sp_def": 10.0,
            "home_recruit": 300.0,
            "away_recruit": 290.0,
            "home_returning_ppa": 0.4,
            "away_returning_ppa": 0.3,
            "home_returning_usage": 0.7,
            "away_returning_usage": 0.6,
            "home_capacity": 100077.0,
            "home_dome": 0.0,
            "home_grass": 1.0,
            "capacity_bucket": 3.0,
            "home_rest_days": 7.0,
            "away_rest_days": 8.0,
            "home_points_l4": 38.0,
            "away_points_l4": 34.0,
            "home_points_allowed_l4": 14.0,
            "away_points_allowed_l4": 18.0,
            "home_yards_l4": 480.0,
            "away_yards_l4": 450.0,
            "home_yards_allowed_l4": 280.0,
            "away_yards_allowed_l4": 300.0,
            "home_to_margin_l4": 1.0,
            "away_to_margin_l4": 0.5,
            "home_third_down_l4": 0.48,
            "away_third_down_l4": 0.44,
            "home_ypp_l4": 7.1,
            "away_ypp_l4": 6.8,
            "home_points_prior": 36.0,
            "away_points_prior": 33.0,
            "home_points_allowed_prior": 16.0,
            "away_points_allowed_prior": 19.0,
            "home_yards_prior": 470.0,
            "away_yards_prior": 440.0,
            "home_yards_allowed_prior": 290.0,
            "away_yards_allowed_prior": 310.0,
            "home_success_prior": 0.48,
            "away_success_prior": 0.46,
            "home_def_success_prior": 0.38,
            "away_def_success_prior": 0.40,
            "wx_temp_max": 31.0,
            "wx_temp_min": 21.0,
            "spread": -3.5,
            "over_under": 54.5,
        }
    )
    row.update(overrides)
    return row


def _frame(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def test_team_snapshot_reads_home_side_when_team_is_home():
    frame = _frame(_game())
    snap = team_snapshot(frame, "Alabama")
    assert snap["team"] == "Alabama"
    assert snap["talent"] == 990.0
    assert snap["pregame_elo"] == 1900.0
    assert snap["points_l4"] == 38.0
    assert snap["lat"] == 33.208
    assert snap["conference"] == "SEC"


def test_team_snapshot_reads_away_side_when_team_is_away():
    frame = _frame(_game())
    snap = team_snapshot(frame, "Georgia")
    assert snap["team"] == "Georgia"
    assert snap["talent"] == 980.0
    assert snap["points_l4"] == 34.0
    assert snap["lat"] == 33.950


def test_simulate_matchup_combines_snapshots_and_scores(monkeypatch):
    alabama_home = _game()
    ohio_home = _game(
        game_id=402,
        home_team="Ohio State",
        away_team="Michigan",
        home_id=194,
        away_id=130,
        home_conference="Big Ten",
        away_conference="Big Ten",
        venue="Ohio Stadium",
        home_lat=40.002,
        home_lon=-83.019,
        home_elev="220",
        home_pregame_elo=1880.0,
        home_talent=985.0,
        home_points_l4=41.0,
        home_capacity=102780.0,
        spread=-14.0,
        over_under=48.0,
    )
    frame = _frame(alabama_home, ohio_home)
    result = simulate_matchup(
        "Ohio State",
        "Alabama",
        frame=frame,
        models=_models(),
        spread=-6.5,
        over_under=55.0,
        week=4,
    )
    assert len(result) == 1
    row = result.iloc[0]
    assert row["home_team"] == "Ohio State"
    assert row["away_team"] == "Alabama"
    assert row["home_talent"] == 985.0
    assert row["away_talent"] == 990.0
    assert row["talent_diff"] == pytest.approx(-5.0)
    expected_dist = float(
        haversine_miles(pd.Series([33.208]), pd.Series([-87.550]), pd.Series([40.002]), pd.Series([-83.019])).iloc[0]
    )
    assert row["distance_miles"] == pytest.approx(expected_dist, rel=1e-3)
    assert row["pred_margin"] == pytest.approx(7.5)
    assert row["pred_home_win_prob"] == pytest.approx(0.72)
    assert row["pred_home_points"] == pytest.approx(31.25)
    assert row["pred_away_points"] == pytest.approx(23.75)
    assert row["pick"] == "Ohio State"
    assert row["status"] == "hypothetical"
    assert row["week"] == 4


def test_simulate_matchup_omits_split_scores_without_over_under():
    frame = _frame(_game())
    row = simulate_matchup("Alabama", "Georgia", frame=frame, models=_models(), over_under=None, spread=None).iloc[0]
    assert pd.isna(row["pred_home_points"])
    assert pd.isna(row["pred_away_points"])
    assert row["pred_margin"] == pytest.approx(7.5)
    assert pd.isna(row["ats_edge"])


def test_simulate_matchup_scores_scheduled_game_with_overrides():
    frame = _frame(_game())
    row = simulate_matchup(
        "Alabama",
        "Georgia",
        frame=frame,
        models=_models(),
        game_id=401,
        spread=-10.0,
        wx_temp_max=12.0,
    ).iloc[0]
    assert int(row["game_id"]) == 401
    assert row["spread"] == pytest.approx(-10.0)
    assert row["wx_temp_max"] == pytest.approx(12.0)
    assert row["pred_margin"] == pytest.approx(7.5)
    assert row["status"] != "hypothetical"


def test_prediction_payload_matches_cfb2pdf_field_names():
    frame = _frame(_game())
    row = simulate_matchup("Alabama", "Georgia", frame=frame, models=_models()).iloc[0]
    teams = {
        "Alabama": {
            "id": 333,
            "name": "Alabama",
            "abbreviation": "ALA",
            "logo_url": "https://cdn.example/ala.png",
            "primary_color": "#9E1B32",
            "secondary_color": "#FFFFFF",
            "conference": "SEC",
        },
        "Georgia": {
            "id": 61,
            "name": "Georgia",
            "abbreviation": "UGA",
            "logo_url": "https://cdn.example/uga.png",
            "primary_color": "#BA0C2F",
            "secondary_color": "#000000",
            "conference": "SEC",
        },
    }
    payload = prediction_payload(row, teams)
    assert payload["home_win_prob"] == pytest.approx(0.72)
    assert payload["away_win_prob"] == pytest.approx(0.28)
    assert payload["projected_home_score"] == pytest.approx(31.0, abs=0.6)
    assert payload["projected_away_score"] == pytest.approx(23.5, abs=0.6)
    assert payload["home_team"]["logo_url"].endswith("ala.png")
    assert payload["away_team"]["primary_color"] == "#BA0C2F"
    assert payload["pick"] == "Alabama"
    assert "ats_edge" in payload
    assert payload["home_win_prob"] + payload["away_win_prob"] == pytest.approx(1.0)
