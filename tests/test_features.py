import pytest

from cfb_model.features import TARGET_COL, available_features, build_features
from cfb_model.store import connect


def test_feature_rows_match_games(db_path):
    conn = connect(db_path)
    try:
        frame = build_features(conn)
    finally:
        conn.close()
    assert len(frame) > 20
    assert TARGET_COL in frame
    assert frame["elo_diff"].notna().all()
    cols = available_features(frame)
    assert "elo_diff" in cols
    assert "talent_diff" in cols
    assert frame["completed"].eq(1).all()


def test_travel_and_rest_populated(db_path):
    conn = connect(db_path)
    try:
        frame = build_features(conn)
    finally:
        conn.close()
    later = frame[frame["week"] > 1]
    assert later["travel_miles_away"].notna().any()
    assert later["rest_days_home"].notna().any()


def _adj_games(margin_first=14.0, opp_elo=1300.0):
    import pandas as pd

    rows = []
    # Team A plays B, then C, then D (all at neutral sites so only Elo matters).
    for i, (opp, elo, margin) in enumerate([("B", opp_elo, margin_first), ("C", 1500.0, 3.0), ("D", 1500.0, 0.0)]):
        rows.append(
            {
                "game_id": i + 1,
                "season": 2025,
                "start_date": f"2025-09-{6 + 7 * i:02d}T20:00:00Z",
                "home_team": "A",
                "away_team": opp,
                "home_points": 20.0 + margin,
                "away_points": 20.0,
                "home_pregame_elo": 1500.0,
                "away_pregame_elo": elo,
                "elo_diff": 1500.0 - elo,
                "neutral_site": 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_opponent_adjusted_form_rewards_beating_strong_teams_more():
    from src.features import add_opponent_adjusted_form

    weak = add_opponent_adjusted_form(_adj_games(opp_elo=1200.0))  # a 14-point win over a weak team
    strong = add_opponent_adjusted_form(_adj_games(opp_elo=1700.0))  # the same 14 points over a strong one
    second_weak = weak.loc[weak["game_id"] == 2, "home_mov_x_l4"].iloc[0]
    second_strong = strong.loc[strong["game_id"] == 2, "home_mov_x_l4"].iloc[0]
    assert second_strong > second_weak  # same raw margin, tougher opponent => better form
    # Elo expects +12.84 against a 1200 team (so +14 is barely better) and -8.56 against a 1700 team (+14 is far better)
    assert second_weak == pytest.approx(14.0 - 12.84)
    assert second_strong == pytest.approx(14.0 + 8.56)
    assert weak.loc[weak["game_id"] == 2, "home_sos_l4"].iloc[0] == pytest.approx(1200.0)


def test_opponent_adjusted_form_never_uses_the_game_itself():
    from src.features import add_opponent_adjusted_form

    base = add_opponent_adjusted_form(_adj_games(margin_first=14.0))
    flipped = add_opponent_adjusted_form(_adj_games(margin_first=-30.0))
    first = lambda f: f.loc[f["game_id"] == 1, ["home_mov_x_l4", "home_mov_x_l8", "home_mov_x_season", "home_sos_l4"]]
    assert first(base).isna().all().all() and first(flipped).isna().all().all()  # no history yet
    # ...but its result does reach later games
    assert base.loc[base["game_id"] == 2, "home_mov_x_l4"].iloc[0] != flipped.loc[flipped["game_id"] == 2, "home_mov_x_l4"].iloc[0]
