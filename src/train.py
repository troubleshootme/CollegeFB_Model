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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR
from src.features import FEATURE_COLS, MARKET_COLS, build_feature_frame, model_matrix


ATS_THRESHOLDS = (0, 3, 5, 7)


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


def _ridge() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=5.0)),
        ]
    )


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


def _hgb_regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.06,
        max_iter=400,
        l2_regularization=0.1,
        min_samples_leaf=25,
        random_state=42,
    )


def train() -> dict:
    MODELS_DIR.mkdir(exist_ok=True)
    print("Building features from SQLite...")
    frame = build_feature_frame()
    print(f"Feature rows: {len(frame):,}")
    completed = frame[
        frame["completed"].astype(bool)
        & frame["fbs_vs_fbs"].astype(bool)
        & frame["margin"].notna()
        & frame["home_points"].notna()
        & frame["away_points"].notna()
    ].copy()
    upcoming = frame[
        ~frame["completed"].astype(bool)
        & frame["fbs_vs_fbs"].astype(bool)
    ].copy()

    feature_cols = [col for col in FEATURE_COLS if col in completed.columns]
    market_cols = [col for col in feature_cols + MARKET_COLS if col in completed.columns]
    train_df = completed
    if train_df.empty:
        raise RuntimeError("No completed training games found. Run data collection first.")
    from src.weekly import completed_weeks, record_base_catchup

    finished = completed_weeks(completed)
    if finished:
        eval_season, eval_week = finished[-1]
        eval_df = completed[
            (completed["season"] == eval_season) & (completed["week"] == eval_week)
        ].copy()
    else:
        eval_season = int(train_df["season"].max())
        eval_week = int(train_df["week"].max()) if "week" in train_df.columns else 1
        eval_df = train_df.tail(min(800, len(train_df))).copy()
    if eval_df.empty:
        eval_df = train_df.tail(min(800, len(train_df))).copy()

    x_train = model_matrix(train_df, feature_cols)
    x_eval = model_matrix(eval_df, feature_cols)
    x_train_mkt = model_matrix(train_df, market_cols)
    x_eval_mkt = model_matrix(eval_df, market_cols)
    y_train = train_df["margin"].astype(float)
    y_eval = eval_df["margin"].astype(float)

    print(f"Training HGB on {len(train_df):,} completed games (eval week {eval_season} w{eval_week} n={len(eval_df):,})")
    hgb = _hgb_regressor()
    hgb.fit(x_train, y_train)
    hgb_mkt = _hgb_regressor()
    hgb_mkt.fit(x_train_mkt, y_train)
    print("Fitting ridge and win classifier...")
    ridge = _ridge()
    ridge.fit(x_train, y_train)

    hgb_pred = hgb.predict(x_eval)
    hgb_mkt_pred = hgb_mkt.predict(x_eval_mkt)
    ridge_pred = ridge.predict(x_eval)

    win_model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.06,
        max_iter=300,
        l2_regularization=0.1,
        min_samples_leaf=25,
        random_state=42,
    )
    y_win_train = (train_df["home_points"] > train_df["away_points"]).astype(int)
    y_win_eval = (eval_df["home_points"] > eval_df["away_points"]).astype(int)
    win_model.fit(x_train, y_win_train)
    win_proba = win_model.predict_proba(x_eval)[:, 1]

    print("Fitting ATS residual model...")
    ats_cols = [col for col in feature_cols + ["spread"] if col in train_df.columns]
    ats_train = train_df[train_df["spread"].notna() & train_df["margin"].notna()].copy()
    ats_eval = eval_df[eval_df["spread"].notna() & eval_df["margin"].notna()].copy()
    y_ats_train = ats_train["margin"].astype(float) + ats_train["spread"].astype(float)
    y_ats_eval = ats_eval["margin"].astype(float) + ats_eval["spread"].astype(float)
    ats_model = _hgb_regressor()
    ats_model.fit(model_matrix(ats_train, ats_cols), y_ats_train)
    ats_pred = ats_model.predict(model_matrix(ats_eval, ats_cols)) if not ats_eval.empty else np.array([])
    ats_eval_metrics = _ats_threshold_metrics(y_ats_eval.to_numpy(), ats_pred) if not ats_eval.empty else {"n": 0}
    margin_edge = hgb_pred + eval_df["spread"].to_numpy()
    ats_from_margin = _ats_threshold_metrics(
        (eval_df["margin"].astype(float) + eval_df["spread"].astype(float)).to_numpy(),
        np.asarray(margin_edge, dtype=float),
    )

    hgb_metrics = _metrics(y_eval, hgb_pred, eval_df["spread"])
    results = {
        "train_seasons": f"{int(train_df['season'].min())}-{int(train_df['season'].max())}",
        "train_games": int(len(train_df)),
        "eval_season": int(eval_season),
        "eval_week": int(eval_week),
        "eval_games": int(len(eval_df)),
        "features": feature_cols,
        "hgb_margin": hgb_metrics,
        "hgb_with_market": _metrics(y_eval, hgb_mkt_pred, eval_df["spread"]),
        "ridge_margin": _metrics(y_eval, ridge_pred, eval_df["spread"]),
        "hgb_win": {
            "accuracy": round(float(accuracy_score(y_win_eval, (win_proba >= 0.5).astype(int))), 4),
            "brier": round(float(brier_score_loss(y_win_eval, np.clip(win_proba, 1e-6, 1 - 1e-6))), 4),
            "log_loss": round(
                float(log_loss(y_win_eval, np.clip(win_proba, 1e-6, 1 - 1e-6), labels=[0, 1])),
                4,
            ),
        },
        "ats_residual": ats_eval_metrics,
        "ats_from_margin_model": ats_from_margin,
        "latest_week": {
            "season": int(eval_season),
            "week": int(eval_week),
            "n": int(len(eval_df)),
            "mae_base": hgb_metrics.get("mae"),
            "winner_accuracy": hgb_metrics.get("winner_accuracy"),
        },
        "weather_coverage": {
            "train": round(float(train_df["wx_temp_max"].notna().mean()), 3) if "wx_temp_max" in train_df.columns else 0,
            "eval": round(float(eval_df["wx_temp_max"].notna().mean()), 3) if "wx_temp_max" in eval_df.columns else 0,
        },
    }

    print("Walk-forward evaluation...")
    walk = {}
    for season in sorted(completed["season"].unique()):
        if season < 2022:
            continue
        tr = completed[completed["season"] < season]
        te = completed[completed["season"] == season]
        if len(tr) < 500 or te.empty:
            continue
        model = _hgb_regressor()
        model.fit(model_matrix(tr, feature_cols), tr["margin"].astype(float))
        pred = model.predict(model_matrix(te, feature_cols))
        walk[str(int(season))] = _metrics(te["margin"].astype(float), pred, te["spread"])
    results["walk_forward"] = walk

    print("Walk-forward ATS residual...")
    ats_walk = {}
    for season in sorted(completed["season"].unique()):
        if season < 2022:
            continue
        tr = completed[(completed["season"] < season) & completed["spread"].notna()]
        te = completed[(completed["season"] == season) & completed["spread"].notna()]
        if len(tr) < 500 or te.empty:
            continue
        model = _hgb_regressor()
        y_tr = tr["margin"].astype(float) + tr["spread"].astype(float)
        y_te = te["margin"].astype(float) + te["spread"].astype(float)
        model.fit(model_matrix(tr, ats_cols), y_tr)
        pred = model.predict(model_matrix(te, ats_cols))
        ats_walk[str(int(season))] = _ats_threshold_metrics(y_te.to_numpy(), pred)
    results["walk_forward_ats"] = ats_walk

    print("Computing permutation importance...")
    perm = permutation_importance(
        hgb,
        x_eval,
        y_eval,
        n_repeats=4,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )
    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": perm.importances_mean,
        }
    ).sort_values("importance", ascending=False)

    test_out = eval_df[
        ["season", "week", "start_date", "home_team", "away_team", "home_points", "away_points", "margin", "spread", "over_under"]
    ].copy()
    test_out["pred_margin"] = hgb_pred
    test_out["pred_margin_with_market"] = hgb_mkt_pred
    test_out["pred_home_win_prob"] = win_proba
    test_out["ridge_margin"] = ridge_pred
    test_out["edge_vs_spread"] = test_out["pred_margin"] + test_out["spread"]
    test_out["ats_edge"] = np.nan
    if not ats_eval.empty:
        test_out.loc[ats_eval.index, "ats_edge"] = ats_pred
    test_out.to_csv(MODELS_DIR / f"eval_{eval_season}_w{eval_week}_predictions.csv", index=False)

    if not upcoming.empty:
        upcoming_x = model_matrix(upcoming, feature_cols)
        keep_cols = [
            col
            for col in [
                "season",
                "week",
                "start_date",
                "home_team",
                "away_team",
                "spread",
                "over_under",
                "home_pregame_elo",
                "away_pregame_elo",
                "venue",
                "home_conference",
                "away_conference",
                "home_classification",
                "away_classification",
                "conference_game",
                "neutral_site",
                "fbs_vs_fbs",
                "wx_temp_max",
                "wx_precip",
                "wx_wind",
                "wx_precip_exposed",
                "home_points_l4",
                "away_points_l4",
                "home_points_prior",
                "away_points_prior",
                "home_rest_days",
                "away_rest_days",
                "home_talent",
                "away_talent",
                "elo_diff",
                "distance_miles",
            ]
            if col in upcoming.columns
        ]
        upcoming_out = upcoming[keep_cols].copy()
        upcoming_out["pred_margin"] = hgb.predict(upcoming_x)
        upcoming_out["pred_home_win_prob"] = win_model.predict_proba(upcoming_x)[:, 1]
        totals = pd.to_numeric(upcoming["over_under"], errors="coerce")
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
        week1 = upcoming_out[(upcoming_out["week"] == 1) & upcoming_out["spread"].notna()]
        if not week1.empty:
            season = int(week1["season"].iloc[0])
            week1.to_csv(MODELS_DIR / f"week1_{season}.csv", index=False)
            results["week1_games"] = int(len(week1))
    else:
        results["upcoming_games"] = 0

    joblib.dump({"model": hgb, "features": feature_cols, "kind": "margin_hgb"}, MODELS_DIR / "margin_hgb.joblib")
    joblib.dump({"model": hgb_mkt, "features": market_cols, "kind": "margin_hgb_market"}, MODELS_DIR / "margin_hgb_market.joblib")
    joblib.dump({"model": ridge, "features": feature_cols, "kind": "margin_ridge"}, MODELS_DIR / "margin_ridge.joblib")
    joblib.dump({"model": win_model, "features": feature_cols, "kind": "win_hgb"}, MODELS_DIR / "win_hgb.joblib")
    joblib.dump({"model": ats_model, "features": ats_cols, "kind": "ats_residual"}, MODELS_DIR / "ats_residual.joblib")
    importance.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
    (MODELS_DIR / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    record_base_catchup(completed)
    print(json.dumps({k: v for k, v in results.items() if k != "features"}, indent=2))
    print(f"Saved artifacts in {MODELS_DIR}")
    return results


if __name__ == "__main__":
    train()
