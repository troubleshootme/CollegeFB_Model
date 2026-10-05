import pandas as pd

from cfb_model.features import build_features
from cfb_model.store import connect, read_table


def test_rolling_stats_exclude_current_game(db_path):
    conn = connect(db_path)
    try:
        frame = build_features(conn)
    finally:
        conn.close()

    # First game of a season for a team must have NaN in-season rolling margin.
    firsts = frame[(frame["home_games_played"] == 0) | (frame["away_games_played"] == 0)]
    assert not firsts.empty
    home_first = frame[frame["home_games_played"] == 0]
    assert home_first["home_margin_std"].isna().all()

    # Reconstruct the actual prior-game margin for a later Alpha home game and
    # confirm the feature does not equal the current game's margin.
    later = frame[(frame["home_team"] == "Alpha") & (frame["home_games_played"] >= 2)].iloc[0]
    assert pd.notna(later["home_margin_std"])
    current_margin = later["home_points"] - later["away_points"]
    assert abs(later["home_margin_std"] - current_margin) > 1e-6


def test_close_spread_not_in_feature_matrix(db_path):
    from cfb_model.features import FEATURE_COLS, available_features

    conn = connect(db_path)
    try:
        frame = build_features(conn)
    finally:
        conn.close()
    cols = available_features(frame, blend=False)
    assert "close_spread" not in cols
    assert "close_spread" not in FEATURE_COLS
    assert "close_spread" in frame.columns
