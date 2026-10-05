"""Score-by-quarter model: storage, scoring-event realism, and the displayed line score."""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from src import quarters as Q
from src.linescores import COLUMNS, parse_line_scores, rows_from_games, sync_line_scores, stored_years


def _game(gid, home, away, completed=True, season=2025):
    return {"id": gid, "season": season, "completed": completed, "homeLineScores": home, "awayLineScores": away}


def test_parse_line_scores_splits_regulation_and_overtime():
    assert parse_line_scores([7, 3, 0, 14]) == ([7.0, 3.0, 0.0, 14.0], 0)
    quarters, ot = parse_line_scores([7, 3, 0, 14, 6, 3])
    assert quarters == [7.0, 3.0, 0.0, 14.0] and ot == 9
    quarters, _ = parse_line_scores(None)
    assert all(q != q for q in quarters)


def test_rows_from_games_skips_unplayed_and_incomplete_line_scores():
    rows = rows_from_games(
        [
            _game(1, [7, 0, 7, 7], [3, 3, 0, 7]),
            _game(2, [7, 0, 7, 7], [3, 3, 0, 7], completed=False),
            _game(3, [7, 0], [3, 3, 0, 7]),
            _game(4, [7, 0, 7, 7, 6], [3, 3, 0, 7, 0]),
        ]
    )
    assert [r["game_id"] for r in rows] == [1, 4]
    assert rows[1]["went_to_ot"] == 1 and rows[1]["home_ot"] == 6 and rows[0]["went_to_ot"] == 0


def test_sync_line_scores_replaces_only_the_requested_season(tmp_path):
    db = tmp_path / "t.db"

    def fetch(year):
        return pd.DataFrame(
            [{**{c: 0 for c in COLUMNS}, "game_id": year * 10 + 1, "season": year, "completed": 1}], columns=COLUMNS
        )

    assert sync_line_scores([2024, 2025], db_path=db, fetch=fetch) == 2
    assert sync_line_scores([2025], db_path=db, fetch=fetch) == 1  # re-pull does not duplicate
    with sqlite3.connect(db) as con:
        assert stored_years(con) == {2024, 2025}
        assert con.execute("select count(*) from game_line_scores").fetchone()[0] == 2


@pytest.mark.parametrize("mu", [1.5, 4.0, 6.9, 10.0, 14.0])
def test_quarter_pmf_is_a_distribution_with_the_requested_mean(mu):
    model = Q.QuarterModel()
    for q in range(4):
        pmf = Q.quarter_pmf(model, q, mu)
        assert pmf.sum() == pytest.approx(1.0)
        assert (pmf * np.arange(len(pmf))).sum() == pytest.approx(mu, rel=0.03)
        assert pmf[1] == 0.0  # a team cannot score exactly 1 point in a quarter


def test_shares_sum_to_one_and_favourites_front_load_blowouts():
    model = Q.fit_shares(
        Q.QuarterModel(),
        pd.DataFrame(
            {
                **{f"home_q{q}": np.random.default_rng(q).poisson(7, 400) for q in Q.QUARTERS},
                **{f"away_q{q}": np.random.default_rng(q + 9).poisson(7, 400) for q in Q.QUARTERS},
                "exp_margin": np.random.default_rng(1).normal(0, 12, 400),
            }
        ),
    )
    for fav in (True, False):
        assert Q.quarter_shares(model, np.array([0.0, 14.0, 35.0]), fav).sum(axis=-1) == pytest.approx(1.0)


def test_simulated_scores_are_reachable_football_scores():
    model = Q.QuarterModel()
    sim = Q.simulate_regulation(model, np.full(200, 52.0), np.linspace(-20, 20, 200), n_sims=40, seed=5)
    quarter_scores = np.r_[sim["home"].ravel(), sim["away"].ravel()]
    assert not (quarter_scores == 1).any()
    finals = np.r_[sim["home"].sum(2).ravel(), sim["away"].sum(2).ravel()]
    assert not (finals == 1).any()
    assert not np.isin(finals, [4]).mean() > 0.001  # essentially never
    assert (quarter_scores == 7).mean() > (quarter_scores == 6).mean()  # PAT is usually good


@pytest.mark.parametrize("margin", [0.0, 0.1, -0.2, 3.0, -7.5, 14.0, -28.0])
def test_typical_line_score_is_legal_ahead_with_the_predicted_winner_and_stable(margin):
    model = Q.QuarterModel(quarter_freq={"0": 0.27, "3": 0.11, "7": 0.26, "10": 0.08, "2": 0.0019, "4": 0.0}, final_freq={})
    game = Q.typical_line_score(model, "Home U", "Away U", 51.0, margin)
    home, away = sum(game["home"]), sum(game["away"])
    assert home != away
    assert (home > away) == (margin >= 0)
    assert home not in (1, 2, 4, 5) and away not in (1, 2, 4, 5)
    assert all(q != 1 for q in game["home"] + game["away"])
    assert Q.typical_line_score(model, "Home U", "Away U", 51.0, margin) == game


def test_typical_line_score_tracks_expected_points():
    model = Q.QuarterModel()
    homes, aways = [], []
    for i in range(60):
        game = Q.typical_line_score(model, f"h{i}", f"a{i}", 56.0, 10.0)
        homes.append(sum(game["home"]))
        aways.append(sum(game["away"]))
    assert abs(np.mean(homes) - 33.0) < 2.5 and abs(np.mean(aways) - 23.0) < 2.5


def test_model_round_trips_through_disk(tmp_path):
    model = Q.QuarterModel(sigma_margin=3.2, fg_gamma=0.4)
    model.save(tmp_path)
    loaded = Q.QuarterModel.load(tmp_path)
    assert loaded.sigma_margin == 3.2 and loaded.fg_gamma == 0.4
    assert Q.QuarterModel.load(tmp_path / "missing") is None


def test_fit_recovers_a_model_from_games_it_generated():
    truth = Q.QuarterModel()
    rng = np.random.default_rng(0)
    margin = rng.normal(0, 10, 500)
    total = rng.normal(52, 5, 500)
    sim = Q.simulate_regulation(truth, total, margin, n_sims=1, seed=1)
    scores = pd.DataFrame(
        {
            **{f"home_q{q}": sim["home"][:, 0, q - 1] for q in Q.QUARTERS},
            **{f"away_q{q}": sim["away"][:, 0, q - 1] for q in Q.QUARTERS},
            "exp_margin": margin,
            "exp_total": total,
        }
    )
    fitted = Q.fit(scores, max_iter=4)
    assert fitted.quarter_freq and fitted.final_freq
    assert fitted.final_freq.get("1", 0.0) == 0.0
    assert np.isfinite(fitted.sigma_margin) and np.isfinite(fitted.sigma_total)


def test_score_slate_uses_quarters_when_a_model_exists(monkeypatch):
    import src.simulate as S
    from tests.test_simulate import _game, _models  # reuse the stub models

    monkeypatch.setattr(S, "load_quarter_model", lambda: Q.QuarterModel())
    slate = pd.DataFrame([_game(spread=-3.0, over_under=51.0)])
    scored = S.score_slate(slate, models=_models())
    row = scored.iloc[0]
    assert row["pred_home_points"] == sum(row[f"pred_home_q{q}"] for q in (1, 2, 3, 4))
    assert row["pred_away_points"] == sum(row[f"pred_away_q{q}"] for q in (1, 2, 3, 4))
    assert row["pred_home_points"] != row["pred_away_points"]
    payload = S.prediction_payload(row)
    assert len(payload["pred_home_quarters"]) == 4 and sum(payload["pred_home_quarters"]) == payload["projected_home_score"]


def test_score_slate_falls_back_without_a_quarter_model(monkeypatch):
    import src.simulate as S
    from tests.test_simulate import _game, _models

    monkeypatch.setattr(S, "load_quarter_model", lambda: None)
    scored = S.score_slate(pd.DataFrame([_game(spread=-3.0, over_under=51.0)]), models=_models())
    assert "pred_home_q1" not in scored.columns
    assert S.prediction_payload(scored.iloc[0])["pred_home_quarters"] is None
