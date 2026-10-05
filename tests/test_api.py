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


def test_health_ok(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "models_ready" in body


def test_health_does_not_build_feature_frame(client, monkeypatch):
    monkeypatch.setattr("app.main.get_feature_frame", lambda: (_ for _ in ()).throw(AssertionError("built frame")))
    monkeypatch.setattr("app.main.cached_feature_frame", lambda: None)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


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
    assert game["projected_home_score"] == 31
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


def test_weeks_keeps_live_season_and_names_replay_season(client, monkeypatch):
    frame = pd.DataFrame(
        [
            {"season": 2025, "week": 14, "game_id": 1, "completed": True, "fbs_vs_fbs": True},
            {"season": 2026, "week": 1, "game_id": 2, "completed": False, "fbs_vs_fbs": True},
        ]
    )
    monkeypatch.setattr("app.main.get_feature_frame", lambda: frame)
    response = client.get("/api/weeks")
    assert response.status_code == 200
    body = response.json()
    assert body["season"] == 2026
    assert body["replay_season"] == 2025


def test_weeks_exposes_week_0_from_board_week(client, monkeypatch):
    frame = pd.DataFrame(
        [
            {"season": 2026, "week": 1, "board_week": 0, "game_id": 1, "completed": True, "fbs_vs_fbs": True},
            {"season": 2026, "week": 1, "board_week": 1, "game_id": 2, "completed": False, "fbs_vs_fbs": True},
        ]
    )
    monkeypatch.setattr("app.main.get_feature_frame", lambda: frame)
    body = client.get("/api/weeks").json()
    weeks = [row["week"] for row in body["weeks"] if row["season"] == 2026]
    assert weeks == [0, 1]


def test_replay_returns_held_out_finals_without_touching_live_week(client, monkeypatch):
    teams = {
        "Alabama": _team("Alabama", "#9E1B32", "https://cdn.example/ala.png"),
        "Georgia": _team("Georgia", "#BA0C2F", "https://cdn.example/uga.png"),
    }
    row = pd.Series(
        {
            "game_id": 401752,
            "season": 2025,
            "week": 14,
            "completed": True,
            "status": "completed",
            "start_date": pd.Timestamp("2025-11-29T17:00:00Z"),
            "kick_label": "Sat Nov 29, 11:00 AM CT",
            "home_team": "Alabama",
            "away_team": "Georgia",
            "venue": "Bryant-Denny Stadium",
            "neutral_site": 0,
            "conference_game": 1,
            "home_points": 27.0,
            "away_points": 24.0,
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
        }
    )

    def _board(week=None):
        return {
            "season": 2025,
            "train_fraction": 0.75,
            "train_count": 6,
            "score_count": 2,
            "cutoff": "2025-11-22T17:00:00+00:00",
            "week": week,
            "weeks": [14],
            "games": pd.DataFrame([row]),
            "source": "local",
        }

    monkeypatch.setattr("app.main.models_ready", lambda: True)
    monkeypatch.setattr("app.main.load_team_directory", lambda: teams)
    monkeypatch.setattr("app.main.replay_board", _board)
    response = client.get("/api/replay", params={"week": 14})
    assert response.status_code == 200
    body = response.json()
    assert body["season"] == 2025
    assert body["week"] == 14
    game = body["games"][0]
    assert game["completed"] is True
    assert game["home_points"] == 27
    assert game["away_points"] == 24
    assert game["winner_hit"] is True


def test_job_post_requires_json_content_type_and_rejects_foreign_origin(monkeypatch):
    from app import main as app_main

    started = []
    monkeypatch.setattr(app_main, "start_job", lambda kind, holdout_season=None: started.append(kind) or {"id": "x"})
    client = TestClient(app_main.app)
    # form-style "simple" cross-site request: no preflight, no JSON content type
    assert client.post("/api/jobs/pipeline", headers={"Origin": "https://evil.example"}).status_code == 415
    assert client.post(
        "/api/jobs/pipeline", json={}, headers={"Origin": "https://evil.example"}
    ).status_code == 403
    assert started == []
    assert client.post("/api/jobs/pipeline", json={}, headers={"Origin": "http://localhost:38417"}).status_code == 200
    assert client.post("/api/jobs/pipeline", json={}).status_code == 200  # non-browser client
    assert started == ["pipeline", "pipeline"]
