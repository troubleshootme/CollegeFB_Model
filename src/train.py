from __future__ import annotations

import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)

from src.config import MODELS_DIR
from src.ensemble import BlendRegressor, MarginWinClassifier, make_hgb, make_ridge
from src.features import FEATURE_COLS, MARKET_COLS, build_feature_frame, model_matrix


ATS_THRESHOLDS = (0, 3, 5, 7)
WALK_FORWARD_START = 2022  # first season scored out-of-sample in reports
CALIBRATION_START = 2016  # first season whose out-of-fold margins feed the win calibrator


def _metrics(y_true: pd.Series, y_pred: np.ndarray, spreads: pd.Series | None = None) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    winner_acc = float(accuracy_score((y_true > 0).astype(int), (y_pred > 0).astype(int)))
    out = {"mae": round(mae, 3), "rmse": round(rmse, 3), "winner_accuracy": round(winner_acc, 4), "n": int(len(y_true))}
    if spreads is not None:
        residual = y_true.to_numpy(dtype=float) + pd.to_numeric(spreads, errors="coerce").to_numpy(dtype=float)
        pred_edge = np.asarray(y_pred, dtype=float) + pd.to_numeric(spreads, errors="coerce").to_numpy(dtype=float)
        ats = _ats_threshold_metrics(residual, pred_edge)
        out["ats_accuracy"] = ats.get("ats_ge_0", {}).get("accuracy")
        out["ats_n"] = ats.get("ats_ge_0", {}).get("n")
        mask = np.isfinite(residual) & np.isfinite(pred_edge)
        if mask.any():
            market_margin = -pd.to_numeric(spreads, errors="coerce").to_numpy(dtype=float)[mask]
            out["market_mae"] = round(float(mean_absolute_error(y_true.to_numpy()[mask], market_margin)), 3)
    return out


def _ats_threshold_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[valid]
    pred = pred[valid]
    out: dict = {"mae": round(float(mean_absolute_error(actual, pred)), 3) if len(actual) else None, "n": int(len(actual))}
    pushes = actual == 0
    for thresh in ATS_THRESHOLDS:
        mask = (np.abs(pred) >= thresh) & ~pushes
        n = int(mask.sum())
        if n == 0:
            out[f"ats_ge_{thresh}"] = {"n": 0, "accuracy": None}
            continue
        acc = float(((actual[mask] > 0) == (pred[mask] > 0)).mean())
        out[f"ats_ge_{thresh}"] = {"n": n, "accuracy": round(acc, 4)}
    return out


def _kick_label(value) -> str:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return ""
    local = ts.tz_convert("America/Chicago")
    hour = local.strftime("%I").lstrip("0") or "0"
    return f"{local.strftime('%a %b')} {local.day}, {hour}:{local.strftime('%M %p')} CT"


def _win_conf(home_wp: float) -> str:
    p = max(float(home_wp), 1.0 - float(home_wp))
    if p >= 0.90:
        return "lock"
    if p >= 0.75:
        return "strong"
    if p >= 0.60:
        return "likely"
    return "lean"


def annotate_board(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["kick_label"] = out["start_date"].map(_kick_label)
    margin = pd.to_numeric(out["pred_margin"], errors="coerce")
    wp = pd.to_numeric(out["pred_home_win_prob"], errors="coerce")
    out["pick"] = np.where(margin >= 0, out["home_team"], out["away_team"])
    out["pick_prob"] = np.where(out["pick"].eq(out["home_team"]), wp, 1.0 - wp)
    out["conf"] = wp.map(_win_conf)
    if "edge_vs_spread" in out.columns:
        edge = pd.to_numeric(out["edge_vs_spread"], errors="coerce")
    else:
        edge = pd.Series(np.nan, index=out.index)
    cover = pd.Series(pd.NA, index=out.index, dtype="object")
    cover.loc[edge > 0.25] = out.loc[edge > 0.25, "home_team"]
    cover.loc[edge < -0.25] = out.loc[edge < -0.25, "away_team"]
    out["cover_side"] = cover
    return out


def _default_holdout(frame: pd.DataFrame) -> int:
    """Latest season that is over (a season in progress is not a holdout).

    A season counts as over at >=99% of FBS games played, so one cancelled game
    does not pin the holdout to an older year.
    """
    fbs = frame[frame["fbs_vs_fbs"].astype(bool)]
    played = fbs.groupby("season")["completed"].agg(lambda c: float(c.astype(bool).mean()))
    done = played[played >= 0.99].index
    if len(done) == 0:
        raise RuntimeError("No fully completed season found. Run data collection first.")
    return int(done.max())


def _atomic_dump(payload: dict, path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(payload, tmp)
    os.replace(tmp, path)


def _pack(model, features: list[str], kind: str) -> dict:
    return {"model": model, "features": features, "kind": kind, "sklearn_version": sklearn.__version__}


def _out_of_fold_margins(completed: pd.DataFrame, feature_cols: list[str], first_test: int, last_test: int) -> pd.DataFrame:
    """Expanding-window margin predictions: season s is scored by a model fit on seasons < s."""
    parts = []
    for season in range(first_test, last_test + 1):
        tr = completed[completed["season"] < season]
        te = completed[completed["season"] == season]
        if len(tr) < 500 or te.empty:
            continue
        model = BlendRegressor().fit(model_matrix(tr, feature_cols), tr["margin"].astype(float))
        parts.append(
            pd.DataFrame(
                {
                    "season": season,
                    "oof_margin": model.predict(model_matrix(te, feature_cols)),
                    "home_win": (te["home_points"] > te["away_points"]).astype(int).to_numpy(),
                },
                index=te.index,
            )
        )
    if not parts:
        return pd.DataFrame(columns=["season", "oof_margin", "home_win"])
    return pd.concat(parts)


def _fit_win_model(margin_model, oof: pd.DataFrame, upto_season: int) -> MarginWinClassifier:
    """Logistic calibration of margin -> P(home win), using only out-of-fold seasons <= upto_season."""
    use = oof[oof["season"] <= upto_season]
    if len(use) < 200:
        raise RuntimeError("Not enough out-of-fold games to calibrate win probabilities.")
    calibrator = MarginWinClassifier.fit_calibrator(use["oof_margin"].to_numpy(), use["home_win"].to_numpy())
    return MarginWinClassifier(margin_model, calibrator)


def _fit_all(df: pd.DataFrame, feature_cols: list[str], market_cols: list[str], ats_cols: list[str], oof: pd.DataFrame, upto: int):
    """Fit every shipped model on ``df``. Win calibration uses OOF seasons <= ``upto``."""
    y = df["margin"].astype(float)
    margin = BlendRegressor().fit(model_matrix(df, feature_cols), y)
    market = BlendRegressor().fit(model_matrix(df, market_cols), y)
    win = _fit_win_model(margin, oof, upto)
    lined = df[df["spread"].notna()]
    ats = make_hgb().fit(model_matrix(lined, ats_cols), lined["margin"].astype(float) + lined["spread"].astype(float))
    return margin, market, win, ats


def train(holdout_season: int | None = None) -> dict:
    MODELS_DIR.mkdir(exist_ok=True)
    print("Building features from SQLite...")
    frame = build_feature_frame()
    print(f"Feature rows: {len(frame):,}")
    if holdout_season is None:
        holdout_season = _default_holdout(frame)
    completed = frame[
        frame["completed"].astype(bool)
        & frame["fbs_vs_fbs"].astype(bool)
        & frame["margin"].notna()
        & frame["home_points"].notna()
        & frame["away_points"].notna()
    ].copy()
    feature_cols = [col for col in FEATURE_COLS if col in completed.columns]
    market_cols = [col for col in feature_cols + MARKET_COLS if col in completed.columns]
    ats_cols = [col for col in feature_cols + ["spread"] if col in completed.columns]
    train_df = completed[completed["season"] < holdout_season]
    test_df = completed[completed["season"] == holdout_season]
    if train_df.empty:
        raise RuntimeError("No completed training games found. Run data collection first.")
    if test_df.empty:
        holdout_season = int(completed["season"].max())
        train_df = completed[completed["season"] < holdout_season]
        test_df = completed[completed["season"] == holdout_season]
    upcoming = frame[
        ~frame["completed"].astype(bool)
        & frame["fbs_vs_fbs"].astype(bool)
        & (frame["season"] >= holdout_season)
    ].copy()
    y_test = test_df["margin"].astype(float)
    y_win_test = (test_df["home_points"] > test_df["away_points"]).astype(int)

    print("Out-of-fold margins for win-probability calibration...")
    last_season = int(completed["season"].max())
    oof = _out_of_fold_margins(completed, feature_cols, CALIBRATION_START, last_season)

    # --- Honest evaluation: everything below is fit on seasons < holdout only. ---
    print(f"Evaluating on holdout {holdout_season} (train {len(train_df):,} games, test {len(test_df):,})")
    ev_margin, ev_market, ev_win, ev_ats = _fit_all(
        train_df, feature_cols, market_cols, ats_cols, oof, holdout_season - 1
    )
    hgb_only = make_hgb().fit(model_matrix(train_df, feature_cols), train_df["margin"].astype(float))
    ridge_only = make_ridge().fit(model_matrix(train_df, feature_cols), train_df["margin"].astype(float))
    x_test = model_matrix(test_df, feature_cols)
    blend_pred = ev_margin.predict(x_test)
    market_pred = ev_market.predict(model_matrix(test_df, market_cols))
    win_proba = ev_win.predict_proba(x_test)[:, 1]

    ats_test = test_df[test_df["spread"].notna()]
    y_ats_test = ats_test["margin"].astype(float) + ats_test["spread"].astype(float)
    ats_pred = ev_ats.predict(model_matrix(ats_test, ats_cols))
    ats_holdout = _ats_threshold_metrics(y_ats_test.to_numpy(), ats_pred)
    ats_from_margin = _ats_threshold_metrics(
        (test_df["margin"].astype(float) + test_df["spread"].astype(float)).to_numpy(),
        np.asarray(blend_pred + test_df["spread"].to_numpy(), dtype=float),
    )

    results = {
        "holdout_season": holdout_season,
        "train_seasons": f"{int(train_df['season'].min())}-{int(train_df['season'].max())}",
        "train_games": int(len(train_df)),
        "test_games": int(len(test_df)),
        "deployed_on": f"{int(completed['season'].min())}-{last_season} ({len(completed):,} games)",
        "margin_model": "ridge+shallow-HGB blend",
        "win_model": "logistic on out-of-fold blend margin",
        "features": feature_cols,
        "blend_margin": _metrics(y_test, blend_pred, test_df["spread"]),
        "hgb_margin": _metrics(y_test, hgb_only.predict(x_test), test_df["spread"]),
        "ridge_margin": _metrics(y_test, ridge_only.predict(x_test), test_df["spread"]),
        "blend_with_market": _metrics(y_test, market_pred, test_df["spread"]),
        "hgb_win": {
            "accuracy": round(float(accuracy_score(y_win_test, (win_proba >= 0.5).astype(int))), 4),
            "brier": round(float(brier_score_loss(y_win_test, win_proba)), 4),
            "log_loss": round(float(log_loss(y_win_test, np.clip(win_proba, 1e-6, 1 - 1e-6))), 4),
        },
        "ats_residual": ats_holdout,
        "ats_from_margin_model": ats_from_margin,
        "weather_coverage": {
            "train": round(float(train_df["wx_temp_max"].notna().mean()), 3) if "wx_temp_max" in train_df.columns else 0,
            "test": round(float(test_df["wx_temp_max"].notna().mean()), 3) if "wx_temp_max" in test_df.columns else 0,
        },
    }

    print("Walk-forward evaluation (same models and hyperparameters as shipped)...")
    walk: dict = {}
    ats_walk: dict = {}
    for season in sorted(int(s) for s in completed["season"].unique()):
        if season < WALK_FORWARD_START or season > holdout_season:
            continue
        tr = completed[completed["season"] < season]
        te = completed[completed["season"] == season]
        if len(tr) < 500 or te.empty:
            continue
        model = BlendRegressor().fit(model_matrix(tr, feature_cols), tr["margin"].astype(float))
        walk[str(season)] = _metrics(te["margin"].astype(float), model.predict(model_matrix(te, feature_cols)), te["spread"])
        tr_l = tr[tr["spread"].notna()]
        te_l = te[te["spread"].notna()]
        if len(tr_l) >= 500 and not te_l.empty:
            ats_model = make_hgb().fit(
                model_matrix(tr_l, ats_cols), tr_l["margin"].astype(float) + tr_l["spread"].astype(float)
            )
            ats_walk[str(season)] = _ats_threshold_metrics(
                (te_l["margin"].astype(float) + te_l["spread"].astype(float)).to_numpy(),
                ats_model.predict(model_matrix(te_l, ats_cols)),
            )
    results["walk_forward"] = walk
    results["walk_forward_ats"] = ats_walk

    print("Computing permutation importance...")
    perm = permutation_importance(
        ev_margin, x_test, y_test, n_repeats=10, random_state=42, scoring="neg_mean_absolute_error"
    )
    importance = pd.DataFrame(
        {"feature": feature_cols, "importance": perm.importances_mean, "std": perm.importances_std}
    ).sort_values("importance", ascending=False)

    test_out = test_df[
        ["season", "week", "start_date", "home_team", "away_team", "home_points", "away_points", "margin", "spread", "over_under"]
    ].copy()
    test_out["pred_margin"] = blend_pred
    test_out["pred_margin_with_market"] = market_pred
    test_out["pred_home_win_prob"] = win_proba
    test_out["ridge_margin"] = ridge_only.predict(x_test)
    test_out["edge_vs_spread"] = test_out["pred_margin"] + test_out["spread"]
    test_out["ats_edge"] = np.nan
    if not ats_test.empty:
        test_out.loc[ats_test.index, "ats_edge"] = ats_pred
    test_out.to_csv(MODELS_DIR / f"holdout_{holdout_season}_predictions.csv", index=False)

    # --- Deployment: refit on every completed game (holdout and in-progress season included). ---
    print(f"Refitting shipped models on all {len(completed):,} completed games...")
    margin_model, market_model, win_model, ats_model = _fit_all(
        completed, feature_cols, market_cols, ats_cols, oof, last_season
    )

    if not upcoming.empty:
        upcoming_x = model_matrix(upcoming, feature_cols)
        keep_cols = [
            col
            for col in [
                "season", "week", "start_date", "home_team", "away_team", "spread", "over_under",
                "home_pregame_elo", "away_pregame_elo", "venue", "home_conference", "away_conference",
                "home_classification", "away_classification", "conference_game", "neutral_site",
                "fbs_vs_fbs", "wx_temp_max", "wx_precip", "wx_wind", "wx_precip_exposed",
                "home_points_l4", "away_points_l4", "home_points_prior", "away_points_prior",
                "home_rest_days", "away_rest_days", "home_talent", "away_talent", "elo_diff", "distance_miles",
            ]
            if col in upcoming.columns
        ]
        upcoming_out = upcoming[keep_cols].copy()
        upcoming_out["pred_margin"] = margin_model.predict(upcoming_x)
        upcoming_out["pred_home_win_prob"] = win_model.predict_proba(upcoming_x)[:, 1]
        from src.simulate import predicted_totals

        totals = predicted_totals(upcoming)
        upcoming_out["pred_home_points"] = totals / 2 + upcoming_out["pred_margin"] / 2
        upcoming_out["pred_away_points"] = totals / 2 - upcoming_out["pred_margin"] / 2
        upcoming_out["edge_vs_spread"] = upcoming_out["pred_margin"] + upcoming_out["spread"]
        lined = upcoming[upcoming["spread"].notna()]
        upcoming_out["ats_edge"] = np.nan
        if not lined.empty:
            upcoming_out.loc[lined.index, "ats_edge"] = ats_model.predict(model_matrix(lined, ats_cols))
        upcoming_out = annotate_board(upcoming_out.sort_values(["season", "week", "start_date"]))
        upcoming_out.to_csv(MODELS_DIR / "upcoming_predictions.csv", index=False)
        results["upcoming_games"] = int(len(upcoming_out))
        week1 = upcoming_out[
            (upcoming_out["season"] == holdout_season + 1) & (upcoming_out["week"] == 1) & upcoming_out["spread"].notna()
        ]
        if week1.empty:
            week1 = upcoming_out[(upcoming_out["week"] == 1) & upcoming_out["spread"].notna()]
        if not week1.empty:
            season = int(week1["season"].iloc[0])
            week1.to_csv(MODELS_DIR / f"week1_{season}.csv", index=False)
            results["week1_games"] = int(len(week1))
    else:
        results["upcoming_games"] = 0

    # Pre-holdout models: the only ones that have NOT seen the holdout season. Replay scores with these.
    for key, model, cols, kind in (
        ("margin", ev_margin, feature_cols, "margin_blend"),
        ("win", ev_win, feature_cols, "win_margin_logistic"),
        ("ats", ev_ats, ats_cols, "ats_residual"),
    ):
        payload = _pack(model, cols, kind)
        payload["holdout_season"] = holdout_season
        _atomic_dump(payload, MODELS_DIR / f"holdout_{key}.joblib")
    _atomic_dump(_pack(margin_model, feature_cols, "margin_blend"), MODELS_DIR / "margin_hgb.joblib")
    _atomic_dump(_pack(market_model, market_cols, "margin_blend_market"), MODELS_DIR / "margin_hgb_market.joblib")
    _atomic_dump(_pack(make_ridge().fit(model_matrix(completed, feature_cols), completed["margin"].astype(float)), feature_cols, "margin_ridge"), MODELS_DIR / "margin_ridge.joblib")
    _atomic_dump(_pack(win_model, feature_cols, "win_margin_logistic"), MODELS_DIR / "win_hgb.joblib")
    _atomic_dump(_pack(ats_model, ats_cols, "ats_residual"), MODELS_DIR / "ats_residual.joblib")
    importance.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
    (MODELS_DIR / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps({k: v for k, v in results.items() if k != "features"}, indent=2))
    print(f"Saved artifacts in {MODELS_DIR}")
    return results


if __name__ == "__main__":
    train()
