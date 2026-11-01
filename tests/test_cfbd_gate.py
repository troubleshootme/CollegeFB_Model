from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import CfbdNotAllowed, CfbdQuotaExhausted


@pytest.fixture(autouse=True)
def _reset_cfbd_remaining():
    from src.cfbd_client import set_remaining_calls

    set_remaining_calls(None)
    yield
    set_remaining_calls(None)


def test_get_json_refuses_without_train_flag(monkeypatch):
    monkeypatch.delenv("CFBD_ALLOW_CALLS", raising=False)
    from src.cfbd_client import get_json, set_remaining_calls

    set_remaining_calls(None)

    def boom(*args, **kwargs):
        raise AssertionError("CFBD HTTP must not run on the serve path")

    monkeypatch.setattr("src.cfbd_client._session", boom)
    with pytest.raises(CfbdNotAllowed):
        get_json("/teams/fbs", {"year": 2026})


def test_get_json_circuit_breaks_when_remaining_calls_is_zero(monkeypatch):
    monkeypatch.setenv("CFBD_ALLOW_CALLS", "1")
    from src.cfbd_client import get_json, set_remaining_calls

    set_remaining_calls(0)

    def boom(*args, **kwargs):
        raise AssertionError("quota 0 must not hit the network")

    monkeypatch.setattr("src.cfbd_client._session", boom)
    with pytest.raises(CfbdQuotaExhausted):
        get_json("/games", {"year": 2026})
    set_remaining_calls(None)


def test_load_team_directory_does_not_call_cfbd_when_cache_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("CFBD_ALLOW_CALLS", raising=False)
    monkeypatch.setattr("src.teams.DB_PATH", tmp_path / "missing.db")
    monkeypatch.setattr("src.teams.TEAMS_CACHE_PATH", tmp_path / "missing.json")

    def boom(*args, **kwargs):
        raise AssertionError("empty team cache must not refresh from CFBD")

    monkeypatch.setattr("src.collect.get_json", boom)
    monkeypatch.setattr("src.teams.refresh_team_branding", boom)
    from src.teams import load_team_directory

    assert load_team_directory(refresh_if_empty=True) == {}


def test_load_team_directory_does_not_refresh_missing_logos_without_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("CFBD_ALLOW_CALLS", raising=False)
    db = tmp_path / "collegeFootball.db"
    import sqlite3

    with sqlite3.connect(db) as con:
        pd.DataFrame(
            {"year": [2026], "team_id": [333], "school": ["Alabama"], "conference": ["SEC"], "logo": [None]}
        ).to_sql("fbs_teams", con, index=False)
    monkeypatch.setattr("src.teams.DB_PATH", db)
    monkeypatch.setattr("src.teams.TEAMS_CACHE_PATH", tmp_path / "missing.json")

    def boom(*args, **kwargs):
        raise AssertionError("serve path must not call CFBD /teams for logos")

    monkeypatch.setattr("src.collect.fetch_fbs_teams", boom)
    from src.teams import load_team_directory

    directory = load_team_directory(refresh_if_empty=True)
    assert "Alabama" in directory
    assert directory["Alabama"]["logo_url"] is None


def test_load_team_directory_reads_json_sidecar(tmp_path, monkeypatch):
    cache = tmp_path / "fbs_teams_cache.json"
    cache.write_text(
        json.dumps(
            [
                {
                    "id": 333,
                    "school": "Alabama",
                    "abbreviation": "ALA",
                    "conference": "SEC",
                    "logo": "https://cdn.example/ala.png",
                    "color": "#9E1B32",
                }
            ]
        )
    )
    monkeypatch.setattr("src.teams.DB_PATH", tmp_path / "missing.db")
    monkeypatch.setattr("src.teams.TEAMS_CACHE_PATH", cache)
    from src.teams import load_team_directory

    directory = load_team_directory()
    assert directory["Alabama"]["logo_url"].endswith("ala.png")
    assert directory["Alabama"]["abbreviation"] == "ALA"


def test_api_teams_empty_cache_does_not_hit_cfbd(tmp_path, monkeypatch):
    monkeypatch.delenv("CFBD_ALLOW_CALLS", raising=False)
    monkeypatch.setenv("MODEL_API_KEY", "")
    monkeypatch.setattr("src.config.MODELS_DIR", tmp_path)
    monkeypatch.setattr("src.teams.DB_PATH", tmp_path / "missing.db")
    monkeypatch.setattr("src.teams.TEAMS_CACHE_PATH", tmp_path / "missing.json")

    called = []

    def boom(*args, **kwargs):
        called.append(1)
        raise AssertionError("CFBD")

    monkeypatch.setattr("src.collect.get_json", boom)
    from fastapi.testclient import TestClient
    from importlib import reload
    import app.main as main

    reload(main)
    main.reset_state()
    client = TestClient(main.app)
    response = client.get("/api/teams")
    assert response.status_code == 200
    assert response.json()["teams"] == []
    assert called == []


def test_numeric_game_ids_drop_cfbschedule_uuids():
    from src.collect import attach_numeric_game_ids, keep_numeric_game_ids, numeric_game_id

    assert numeric_game_id(401520182) == 401520182
    assert numeric_game_id("401520182") == 401520182
    assert numeric_game_id("a1b2c3d4-e5f6-7890-abcd-ef1234567890") is None
    lines = pd.DataFrame(
        [
            {"game_id": "uuid-1", "season": 2026, "week": 1, "home_team": "Alabama", "away_team": "Georgia", "spread": -7.5},
            {"game_id": 401520182, "season": 2026, "week": 1, "home_team": "Ohio State", "away_team": "Texas", "spread": -3.5},
        ]
    )
    kept = keep_numeric_game_ids(lines)
    assert kept["game_id"].tolist() == [401520182]
    games = pd.DataFrame(
        [{"game_id": 99, "season": 2026, "week": 1, "home_team": "Alabama", "away_team": "Georgia"}]
    )
    remapped = attach_numeric_game_ids(lines, games)
    assert sorted(remapped["game_id"].tolist()) == [99, 401520182]


def test_annotate_board_market_picks_without_over_under_is_json_safe():
    from src.train import annotate_board, combine_market_picks

    assert combine_market_picks("Alabama", np.nan) == "Alabama"
    assert combine_market_picks(pd.NA, np.float64("nan")) is None
    frame = pd.DataFrame(
        [
            {
                "start_date": pd.Timestamp("2026-10-25T19:00:00Z"),
                "home_team": "Alabama",
                "away_team": "Georgia",
                "pred_margin": 6.2,
                "pred_home_win_prob": 0.64,
                "pred_home_points": 31.0,
                "pred_away_points": 24.0,
                "edge_vs_spread": 2.7,
                "over_under": np.nan,
            }
        ]
    )
    out = annotate_board(frame)
    assert out.iloc[0]["cover_side"] == "Alabama"
    assert pd.isna(out.iloc[0]["ou_side"])
    assert out.iloc[0]["market_picks"] == "Alabama"


def test_prediction_payload_json_dumps_when_over_under_missing():
    from src.simulate import prediction_payload

    row = pd.Series(
        {
            "game_id": np.int64(401),
            "season": np.int64(2026),
            "week": np.int64(9),
            "completed": np.bool_(False),
            "home_team": np.str_("Alabama"),
            "away_team": np.str_("Georgia"),
            "neutral_site": np.int64(0),
            "conference_game": np.int64(1),
            "pred_margin": np.float64(6.2),
            "pred_home_win_prob": np.float64(0.64),
            "pred_home_points": np.float64(31.1),
            "pred_away_points": np.float64(24.9),
            "pick": np.str_("Alabama"),
            "pick_prob": np.float64(0.64),
            "conf": np.str_("likely"),
            "cover_side": pd.NA,
            "ou_side": np.nan,
            "market_picks": np.nan,
            "ats_edge": np.float64("nan"),
            "spread": np.float64(-3.5),
            "over_under": np.nan,
            "kick_label": np.str_("Sat Oct 25, 2:00 PM CT"),
        }
    )
    payload = prediction_payload(row)
    encoded = json.dumps(payload)
    assert "Alabama" in encoded
    assert payload["over_under"] is None
    assert payload["ou_side"] is None
    assert payload["market_picks"] is None
    assert payload["cover_side"] is None
    assert payload["pick"] == "Alabama"


def test_crontab_has_trailing_newline_and_cfbd_flag():
    text = Path("ops/crontab").read_bytes()
    assert text.endswith(b"\n"), "cron ignores a last line with no trailing newline"
    assert b"CFBD_ALLOW_CALLS=1" in text
    assert b"src.schedule_sync" in text
    assert not text.endswith(b"\n\n\n")
