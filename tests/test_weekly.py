from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLS
from src.simulate import score_slate
from src.train import train
from src.weekly import (
    apply_week_layer,
    completed_weeks,
    fit_week_layer,
    log_predictions,
    logged_pred_margin,
    record_base_catchup,
    tick_weekly,
)


class _Reg:
    def predict(self, matrix):
        return np.zeros(len(matrix))


class _Clf:
    def predict_proba(self, matrix):
        home = np.full(len(matrix), 0.5)
        return np.column_stack([1.0 - home, home])


def _models():
    return {
        "margin": {"model": _Reg(), "features": FEATURE_COLS},
        "win": {"model": _Clf(), "features": FEATURE_COLS},
        "ats": {"model": _Reg(), "features": FEATURE_COLS + ["spread"]},
    }


def _row(*, game_id, week, completed=True, fbs=True, margin=10.0, season=2026, **extra):
    row = {col: 0.0 for col in FEATURE_COLS}
    home = 28.0 if margin >= 0 else 21.0
    away = home - margin
    row.update(
        {
            "game_id": game_id,
            "season": season,
            "week": week,
            "completed": completed,
            "fbs_vs_fbs": fbs,
            "home_team": f"Home{game_id}",
            "away_team": f"Away{game_id}",
            "home_points": home if completed else np.nan,
            "away_points": away if completed else np.nan,
            "margin": margin if completed else np.nan,
            "spread": -3.0,
            "over_under": 55.0,
            "start_date": pd.Timestamp("2026-09-12T16:00:00Z") + pd.Timedelta(hours=game_id),
            "elo_diff": 40.0,
            "sp_diff": 5.0,
            "talent_diff": 20.0,
        }
    )
    row.update(extra)
    return row


def _frame(*rows):
    return pd.DataFrame(list(rows))


def test_completed_weeks_require_all_fbs_games_final():
    frame = _frame(
        _row(game_id=1, week=1, completed=True),
        _row(game_id=2, week=1, completed=True),
        _row(game_id=3, week=2, completed=True),
        _row(game_id=4, week=2, completed=False),
    )
    assert completed_weeks(frame) == [(2026, 1)]


def test_non_fbs_leftover_does_not_block_the_week():
    frame = _frame(
        _row(game_id=1, week=3, completed=True, fbs=True),
        _row(game_id=2, week=3, completed=True, fbs=True),
        _row(game_id=3, week=3, completed=False, fbs=False),
    )
    assert completed_weeks(frame) == [(2026, 3)]


def test_week_layer_trains_only_on_that_week_and_moves_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr("src.weekly.MODELS_DIR", tmp_path)
    monkeypatch.setattr("src.simulate.MODELS_DIR", tmp_path)
    week2 = [_row(game_id=i, week=2, margin=12.0) for i in range(1, 13)]
    week3 = [_row(game_id=20 + i, week=3, completed=False, margin=np.nan) for i in range(8)]
    frame = _frame(*week2, *week3)
    report = fit_week_layer(frame, 2026, 2, models=_models())
    assert report["season"] == 2026
    assert report["week"] == 2
    assert report["n"] == 12
    assert report["mae_base"] == pytest.approx(12.0)
    assert report["mae_adjusted"] < report["mae_base"] - 5

    upcoming = pd.DataFrame(week3)
    scored = score_slate(upcoming, models=_models())
    assert scored["pred_margin"].mean() == pytest.approx(12.0, abs=1.5)


def test_second_tick_does_not_stack_the_same_week(tmp_path, monkeypatch):
    monkeypatch.setattr("src.weekly.MODELS_DIR", tmp_path)
    frame = _frame(*[_row(game_id=i, week=4, margin=8.0) for i in range(1, 10)])
    first = tick_weekly(frame=frame, refresh=False, models=_models())
    second = tick_weekly(frame=frame, refresh=False, models=_models())
    assert first["status"] == "learned"
    assert second["status"] == "skipped"
    upcoming = _frame(_row(game_id=99, week=5, completed=False, margin=np.nan))
    scored = apply_week_layer(score_slate(upcoming, models=_models()))
    # Layer is a single week's residual (~8), not 16 from stacking.
    assert scored["pred_margin"].iloc[0] == pytest.approx(8.0, abs=1.5)


def test_tick_skips_weeks_already_in_the_catchup_base(tmp_path, monkeypatch):
    monkeypatch.setattr("src.weekly.MODELS_DIR", tmp_path)
    frame = _frame(*[_row(game_id=i, week=1, margin=9.0) for i in range(1, 10)])
    record_base_catchup(frame)
    result = tick_weekly(frame=frame, refresh=False, models=_models())
    assert result["status"] == "skipped"
    assert not (tmp_path / "week_layer.joblib").exists()


def test_logged_pregame_prediction_is_the_residual_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr("src.weekly.MODELS_DIR", tmp_path)
    upcoming = _frame(*[_row(game_id=i, week=6, completed=False, margin=np.nan) for i in range(1, 10)])
    scored = score_slate(upcoming, models=_models())
    log_predictions(scored)
    finals = upcoming.copy()
    finals["completed"] = True
    finals["home_points"] = 31.0
    finals["away_points"] = 17.0
    finals["margin"] = 14.0
    report = fit_week_layer(finals, 2026, 6, models=_models())
    logged = logged_pred_margin(finals)
    assert logged.notna().all()
    assert report["mae_base"] == pytest.approx(14.0)


def test_train_has_no_holdout_and_uses_every_completed_game(tmp_path, monkeypatch):
    assert "holdout_season" not in inspect.signature(train).parameters
    monkeypatch.setattr("src.train.MODELS_DIR", tmp_path)
    monkeypatch.setattr("src.weekly.MODELS_DIR", tmp_path)
    monkeypatch.setattr("src.train.permutation_importance", lambda *args, **kwargs: type(
        "R", (), {"importances_mean": np.zeros(len(FEATURE_COLS))}
    )())
    rows = []
    gid = 1
    for season in (2024, 2025, 2026):
        for week in range(1, 6):
            for extra in range(4):
                rows.append(
                    _row(
                        game_id=gid,
                        season=season,
                        week=week,
                        margin=7.0 + extra,
                        completed=True,
                    )
                )
                gid += 1
    rows.append(_row(game_id=gid, season=2026, week=6, completed=False, margin=np.nan))
    frame = _frame(*rows)
    monkeypatch.setattr("src.train.build_feature_frame", lambda: frame)
    results = train()
    assert "holdout_season" not in results
    assert results["train_games"] == 60
    assert results["train_seasons"] == "2024-2026"
    assert (tmp_path / "margin_hgb.joblib").exists()
