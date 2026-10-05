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
