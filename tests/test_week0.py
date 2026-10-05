from __future__ import annotations

import pandas as pd

from src.week0 import assign_board_week, slate_for_week


def _rows(*games: dict) -> pd.DataFrame:
    return pd.DataFrame(list(games))


def test_2026_opening_cluster_moves_to_week_0_and_keeps_model_week():
    frame = _rows(
        {
            "season": 2026,
            "week": 1,
            "season_type": "regular",
            "start_date": "2026-08-30T04:00:00Z",
            "home_team": "Stanford",
            "away_team": "Hawai'i",
        },
        {
            "season": 2026,
            "week": 1,
            "season_type": "regular",
            "start_date": "2026-08-29T16:00:00Z",
            "home_team": "TCU",
            "away_team": "North Carolina",
        },
        {
            "season": 2026,
            "week": 1,
            "season_type": "regular",
            "start_date": "2026-09-05T01:00:00Z",
            "home_team": "Stanford",
            "away_team": "Miami",
        },
        {
            "season": 2026,
            "week": 1,
            "season_type": "regular",
            "start_date": "2026-09-04T00:00:00Z",
            "home_team": "Rutgers",
            "away_team": "Massachusetts",
        },
    )
    out = assign_board_week(frame)
    assert list(out["week"]) == [1, 1, 1, 1]
    hawaii = out.loc[out["away_team"] == "Hawai'i"].iloc[0]
    miami = out.loc[out["away_team"] == "Miami"].iloc[0]
    tcu = out.loc[out["home_team"] == "TCU"].iloc[0]
    rutgers = out.loc[out["home_team"] == "Rutgers"].iloc[0]
    assert int(hawaii["board_week"]) == 0
    assert int(tcu["board_week"]) == 0
    assert int(miami["board_week"]) == 1
    assert int(rutgers["board_week"]) == 1


def test_postseason_week_1_is_not_week_0():
    frame = _rows(
        {
            "season": 2025,
            "week": 1,
            "season_type": "postseason",
            "start_date": "2025-12-20T17:00:00Z",
            "home_team": "Oregon",
            "away_team": "Ohio State",
        },
        {
            "season": 2025,
            "week": 1,
            "season_type": "regular",
            "start_date": "2025-08-23T19:00:00Z",
            "home_team": "Kansas State",
            "away_team": "Iowa State",
        },
        {
            "season": 2025,
            "week": 1,
            "season_type": "regular",
            "start_date": "2025-08-28T23:00:00Z",
            "home_team": "Alabama",
            "away_team": "Florida State",
        },
    )
    out = assign_board_week(frame)
    bowls = out[out["season_type"] == "postseason"].iloc[0]
    # Bowls come after the regular season, not mixed into regular-season week 1.
    assert bowls["board_week"] == 2
    assert out.loc[out["home_team"] == "Kansas State", "board_week"].iloc[0] == 0
    assert out.loc[out["home_team"] == "Alabama", "board_week"].iloc[0] == 1


def test_no_gap_leaves_week_1_alone():
    frame = _rows(
        {
            "season": 2023,
            "week": 1,
            "start_date": "2023-09-02T16:00:00Z",
            "home_team": "A",
            "away_team": "B",
        },
        {
            "season": 2023,
            "week": 1,
            "start_date": "2023-09-03T19:00:00Z",
            "home_team": "C",
            "away_team": "D",
        },
    )
    out = assign_board_week(frame)
    assert list(out["board_week"]) == [1, 1]


def test_slate_for_week_uses_board_week_without_rewriting_week():
    frame = assign_board_week(
        _rows(
            {
                "season": 2026,
                "week": 1,
                "start_date": "2026-08-30T04:00:00Z",
                "home_team": "Stanford",
                "away_team": "Hawai'i",
                "fbs_vs_fbs": True,
            },
            {
                "season": 2026,
                "week": 1,
                "start_date": "2026-09-05T01:00:00Z",
                "home_team": "Stanford",
                "away_team": "Miami",
                "fbs_vs_fbs": True,
            },
        )
    )
    week0 = slate_for_week(frame, 2026, 0)
    week1 = slate_for_week(frame, 2026, 1)
    assert list(week0["away_team"]) == ["Hawai'i"]
    assert list(week1["away_team"]) == ["Miami"]
    assert (week0["week"] == 1).all()
    assert (week1["week"] == 1).all()


def test_payload_week_uses_board_week_zero():
    from src.simulate import prediction_payload

    row = pd.Series(
        {
            "week": 1,
            "board_week": 0,
            "home_team": "Stanford",
            "away_team": "Hawai'i",
            "completed": True,
            "home_points": 37,
            "away_points": 27,
        }
    )
    payload = prediction_payload(row)
    assert payload["week"] == 0


def _g(season, week, start, season_type="regular"):
    return {
        "season": season,
        "week": week,
        "season_type": season_type,
        "start_date": start,
        "home_team": "H",
        "away_team": "A",
    }


def test_postseason_games_do_not_land_in_week_1():
    frame = _rows(
        _g(2025, 1, "2025-08-30T16:00:00Z"),
        _g(2025, 14, "2025-12-06T20:00:00Z"),
        _g(2025, 1, "2025-12-20T20:00:00Z", "postseason"),
    )
    board = assign_board_week(frame)
    assert slate_for_week(board, 2025, 1)["season_type"].tolist() == ["regular"]
    assert board.loc[2, "board_week"] > 14


def test_labor_day_gap_is_not_a_week_zero_cut():
    # Thu-Sun slate plus a Tuesday (UTC) game: only a 2 day gap, so nothing moves to week 0.
    frame = _rows(
        _g(2013, 1, "2013-08-29T23:00:00Z"),
        _g(2013, 1, "2013-08-31T16:00:00Z"),
        _g(2013, 1, "2013-09-01T20:00:00Z"),
        _g(2013, 1, "2013-09-03T01:00:00Z"),
    )
    assert (assign_board_week(frame)["board_week"] == 1).all()
