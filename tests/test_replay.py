from __future__ import annotations

import pandas as pd
import pytest

from src.replay import replay_window, replay_week_catalog
from src.simulate import last_completed_season


def _games(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def test_replay_window_holds_out_last_quarter_of_completed_season():
    rows = []
    for i in range(8):
        rows.append(
            {
                "game_id": 100 + i,
                "season": 2024,
                "week": 1,
                "completed": True,
                "fbs_vs_fbs": True,
                "start_date": pd.Timestamp(f"2024-09-{i + 1:02d}T16:00:00Z"),
            }
        )
    for i in range(8):
        rows.append(
            {
                "game_id": 200 + i,
                "season": 2025,
                "week": i + 1,
                "completed": True,
                "fbs_vs_fbs": True,
                "start_date": pd.Timestamp(f"2025-09-{i + 1:02d}T16:00:00Z"),
            }
        )
    rows.append(
        {
            "game_id": 300,
            "season": 2026,
            "week": 1,
            "completed": False,
            "fbs_vs_fbs": True,
            "start_date": pd.Timestamp("2026-08-30T16:00:00Z"),
        }
    )
    window = replay_window(_games(*rows), train_fraction=0.75)
    assert window["season"] == 2025
    assert window["train_count"] == 6
    assert [int(gid) for gid in window["score"]["game_id"]] == [206, 207]
    assert window["cutoff"] == pd.Timestamp("2025-09-07T16:00:00Z")
    assert 2026 not in set(window["score"]["season"])


def test_replay_window_skips_in_progress_season():
    frame = _games(
        {
            "game_id": 1,
            "season": 2025,
            "week": 1,
            "completed": True,
            "fbs_vs_fbs": True,
            "start_date": pd.Timestamp("2025-08-30T16:00:00Z"),
        },
        {
            "game_id": 2,
            "season": 2026,
            "week": 1,
            "completed": True,
            "fbs_vs_fbs": True,
            "start_date": pd.Timestamp("2026-08-30T16:00:00Z"),
        },
        {
            "game_id": 3,
            "season": 2026,
            "week": 1,
            "completed": False,
            "fbs_vs_fbs": True,
            "start_date": pd.Timestamp("2026-08-31T16:00:00Z"),
        },
    )
    assert last_completed_season(frame) == 2025
    window = replay_window(frame)
    assert window["season"] == 2025
    assert list(window["score"]["game_id"]) == [1]


def test_replay_week_catalog_follows_kickoff_and_labels_bowls():
    slate = pd.DataFrame(
        [
            {
                "week": 1,
                "season_type": "postseason",
                "start_date": pd.Timestamp("2025-12-20T00:00:00Z"),
                "game_id": 2,
            },
            {
                "week": 14,
                "season_type": "regular",
                "start_date": pd.Timestamp("2025-11-29T00:00:00Z"),
                "game_id": 1,
            },
        ]
    )
    options = replay_week_catalog(slate)
    assert [row["week"] for row in options] == [14, 1]
    assert options[0]["label"] == "Week 14"
    assert options[1]["label"] == "Bowls / CFP"
