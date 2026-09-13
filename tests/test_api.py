from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "")
    monkeypatch.setattr("src.config.MODELS_DIR", tmp_path)
    monkeypatch.setattr("src.simulate.MODELS_DIR", tmp_path)
    from app.main import app, reset_state

    reset_state()
    return TestClient(app)


def _team(name, color, logo):
    return {
        "id": hash(name) % 1000,
        "name": name,
        "school": name,
        "abbreviation": name[:3].upper(),
        "conference": "SEC",
        "logo_url": logo,
        "primary_color": color,
        "secondary_color": "#FFFFFF",
    }


def test_create_train_job_does_not_pass_holdout(client, monkeypatch):
    seen = {}

    def fake_start(kind, **kwargs):
        seen["kind"] = kind
        seen["kwargs"] = kwargs
        return {"id": "abc", "kind": kind, "status": "queued", "log": ""}

    monkeypatch.setattr("app.main.start_job", fake_start)
    response = client.post("/api/jobs/train", json={})
    assert response.status_code == 200
    assert seen["kind"] == "train"
    assert "holdout_season" not in seen["kwargs"]


def test_health_ok(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "models_ready" in body


def test_predictions_without_models_returns_503(client, monkeypatch):
    monkeypatch.setattr("app.main.models_ready", lambda: False)
    response = client.get("/api/predictions", params={"season": 2026, "week": 1})
    assert response.status_code == 503
    assert "train" in response.json()["detail"].lower()


def test_weekly_predictions_shape(client, monkeypatch):
    teams = {
        "Alabama": _team("Alabama", "#9E1B32", "https://cdn.example/ala.png"),
        "Georgia": _team("Georgia", "#BA0C2F", "https://cdn.example/uga.png"),
    }
    row = pd.Series(
        {
            "game_id": 401,
            "season": 2026,
            "week": 1,
            "completed": False,
            "status": "scheduled",
            "start_date": pd.Timestamp("2026-08-30T16:00:00Z"),
            "kick_label": "Sat Aug 30, 11:00 AM CT",
            "home_team": "Alabama",
            "away_team": "Georgia",
            "venue": "Bryant-Denny Stadium",
            "neutral_site": 0,
            "conference_game": 1,
            "pred_margin": 6.2,
            "pred_home_win_prob": 0.64,
            "pred_home_points": 31.1,
            "pred_away_points": 24.9,
            "pick": "Alabama",
            "pick_prob": 0.64,
            "conf": "likely",
            "cover_side": "Alabama",
            "ats_edge": 2.7,
            "edge_vs_spread": 2.7,
            "spread": -3.5,
            "over_under": 56.0,
            "wx_temp_max": 31.0,
            "wx_temp_min": 21.0,
        }
    )
    monkeypatch.setattr("app.main.models_ready", lambda: True)
    monkeypatch.setattr("app.main.load_team_directory", lambda: teams)
    monkeypatch.setattr("app.main.score_week", lambda season, week: pd.DataFrame([row]))
    response = client.get("/api/predictions", params={"season": 2026, "week": 1})
    assert response.status_code == 200
    games = response.json()["games"]
    assert len(games) == 1
    game = games[0]
    assert game["home_win_prob"] == pytest.approx(0.64)
    assert game["away_win_prob"] == pytest.approx(0.36)
    assert game["projected_home_score"] == pytest.approx(31.1)
    assert game["home_team"]["logo_url"].endswith("ala.png")
    assert game["pick"] == "Alabama"


def test_post_matchup_returns_probs_that_sum_to_one(client, monkeypatch):
    teams = {
        "Alabama": _team("Alabama", "#9E1B32", "https://cdn.example/ala.png"),
        "Georgia": _team("Georgia", "#BA0C2F", "https://cdn.example/uga.png"),
    }
    scored = pd.DataFrame(
        [
            {
                "game_id": None,
                "season": 2026,
                "week": 4,
                "completed": False,
                "status": "hypothetical",
                "start_date": pd.NaT,
                "kick_label": "",
                "home_team": "Alabama",
                "away_team": "Georgia",
                "venue": "Bryant-Denny Stadium",
                "neutral_site": 0,
                "conference_game": 1,
                "pred_margin": 7.5,
                "pred_home_win_prob": 0.72,
                "pred_home_points": 31.25,
                "pred_away_points": 23.75,
                "pick": "Alabama",
                "pick_prob": 0.72,
                "conf": "likely",
                "cover_side": "Alabama",
                "ats_edge": 4.0,
                "edge_vs_spread": 4.0,
                "spread": -3.5,
                "over_under": 55.0,
                "wx_temp_max": 31.0,
                "wx_temp_min": 21.0,
            }
        ]
    )
    monkeypatch.setattr("app.main.models_ready", lambda: True)
    monkeypatch.setattr("app.main.load_team_directory", lambda: teams)

    def _fake_matchup(home, away, **kwargs):
        assert home == "Alabama"
        assert away == "Georgia"
        return scored

    monkeypatch.setattr("app.main.simulate_matchup", _fake_matchup)
    response = client.post(
        "/api/matchups",
        json={"home_team": "Alabama", "away_team": "Georgia", "spread": -3.5, "over_under": 55},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "hypothetical"
    assert 0 <= body["home_win_prob"] <= 1
    assert 0 <= body["away_win_prob"] <= 1
    assert body["home_win_prob"] + body["away_win_prob"] == pytest.approx(1.0)


def test_teams_include_logo_and_color(client, monkeypatch):
    monkeypatch.setattr(
        "app.main.load_team_directory",
        lambda: {
            "Alabama": _team("Alabama", "#9E1B32", "https://cdn.example/ala.png"),
        },
    )
    response = client.get("/api/teams")
    assert response.status_code == 200
    teams = response.json()["teams"]
    assert teams[0]["logo_url"].endswith("ala.png")
    assert teams[0]["primary_color"] == "#9E1B32"


def test_api_key_rejected_when_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "secret-key")
    monkeypatch.setattr("src.config.MODELS_DIR", tmp_path)
    from importlib import reload
    import app.main as main

    reload(main)
    main.reset_state()
    locked = TestClient(main.app)
    denied = locked.get("/api/teams")
    assert denied.status_code == 401
    ok = locked.get("/api/teams", headers={"X-API-Key": "secret-key"})
    assert ok.status_code == 200
    health = locked.get("/api/health")
    assert health.status_code == 200
