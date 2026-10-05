"""Closed loop learns from residuals without peeking at the current score."""

from __future__ import annotations

import pandas as pd

from cfb_model.learn import apply_state, default_state, score_frame, update_state


def _games(pred: float, actual: float, n: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": list(range(n)),
            "week": [6] * n,
            "completed": [1] * n,
            "home_margin": [actual] * n,
            "pred_margin": [pred] * n,
            "pred_margin_raw": [pred] * n,
            "home_games_played": [6] * n,
            "away_games_played": [6] * n,
        }
    )


def test_positive_residual_raises_next_home_margin():
    # Actual 10, pred 3 → we underrated homes. Next card should add points.
    scored = score_frame(_games(pred=3.0, actual=10.0))
    assert scored["residual"].mean() == 7.0
    state = update_state(default_state(), scored)
    assert state["bias"]["overall"] > 0
    nxt = pd.DataFrame(
        {
            "pred_margin": [3.0],
            "week": [7],
            "home_games_played": [7],
            "away_games_played": [7],
            "home_margin": [99.0],  # must not leak into the correction
        }
    )
    adjusted = apply_state(nxt, state, injury_prior=False)
    assert adjusted["pred_margin"].iloc[0] > 3.0
    assert adjusted["pred_margin_raw"].iloc[0] == 3.0
    # Current game's 99-point margin never entered the state.
    assert state["bias"]["overall"] < 5


def test_score_frame_ignores_unplayed_games():
    frame = _games(3, 10, n=4)
    frame.loc[0, "completed"] = 0
    frame.loc[0, "home_margin"] = 40
    scored = score_frame(frame)
    assert len(scored) == 3
    assert 40 not in set(scored["home_margin"])
