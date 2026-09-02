"""Walk-forward evaluation versus Elo, CFBD pregame WP, and closing spread."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error

from cfb_model import config
from cfb_model.features import TARGET_COL, available_features, build_features
from cfb_model.models import completed_mask, fit_models, matrix, predict_frame, save_models


def walk_forward(
    frame: pd.DataFrame | None = None,
    *,
    min_test_year: int = 2021,
    min_train: int = 50,
    blend: bool = False,
) -> tuple[pd.DataFrame, dict, dict]:
    if frame is None:
        frame = build_features()
    done = frame.loc[completed_mask(frame)].copy()
    years = sorted(int(y) for y in done["season"].dropna().unique())
    folds = []
    scored_parts = []
    last_models = None
    for test_year in years:
        if test_year < min_test_year:
            continue
        train = done[done["season"] < test_year]
        test = done[done["season"] == test_year]
        if len(train) < min_train or test.empty:
            continue
        cols = available_features(train, blend=blend)
        models = fit_models(train, cols)
        cols = models["columns"]
        X_test, _ = matrix(test, cols)
        X_test = X_test.reindex(columns=cols)
        ridge_mae = mean_absolute_error(test[TARGET_COL], models["ridge"].predict(X_test))
        xgb_mae = mean_absolute_error(test[TARGET_COL], models["xgb"].predict(models["imputer"].transform(X_test)))
        models["champion"] = "xgb" if xgb_mae <= ridge_mae else "ridge"
        pred = predict_frame(models, test, apply_learning=False)
        pred["fold_year"] = test_year
        scored_parts.append(pred)
        folds.append(
            {
                "year": test_year,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "ridge_mae": float(ridge_mae),
                "xgb_mae": float(xgb_mae),
                "champion": models["champion"],
                **_metrics_dict(pred, prefix=""),
            }
        )
        last_models = models
    scored = pd.concat(scored_parts, ignore_index=True) if scored_parts else done.iloc[0:0]
    summary = {
        "folds": folds,
        "overall": _metrics_dict(scored) if not scored.empty else {},
        "by_week_bucket": _group_metrics(scored, "week_bucket") if not scored.empty else {},
        "champion_votes": pd.Series([f["champion"] for f in folds]).value_counts().to_dict() if folds else {},
    }
    if last_models is None and not done.empty:
        last_models = fit_models(done, available_features(done, blend=blend))
    if last_models is not None and not done.empty:
        last_models["n_train"] = int(len(done))
        overall_ridge = summary.get("overall", {}).get("ridge_mae")
        overall_xgb = summary.get("overall", {}).get("xgb_mae")
        mature = [f for f in folds if f.get("n_train", 0) >= 1000]
        if mature:
            overall_ridge = float(sum(f["ridge_mae"] for f in mature) / len(mature))
            overall_xgb = float(sum(f["xgb_mae"] for f in mature) / len(mature))
        if overall_xgb is not None and overall_ridge is not None:
            champ_name = "xgb" if overall_xgb <= overall_ridge else "ridge"
        else:
            votes = summary.get("champion_votes") or {"xgb": 1}
            champ_name = max(votes, key=votes.get)
        last_models = fit_models(done, last_models["columns"])
        last_models["champion"] = champ_name
        last_models["n_train"] = int(len(done))
        X_all, _ = matrix(done, last_models["columns"])
        champ = last_models[champ_name]
        if champ_name == "xgb":
            fitted = champ.predict(last_models["imputer"].transform(X_all))
        else:
            fitted = champ.predict(X_all)
        last_models["sigma"] = float(np.std(done[TARGET_COL] - fitted, ddof=1) or config.DEFAULT_MARGIN_SIGMA)
        summary["final_champion"] = champ_name
    return scored, summary, last_models or {}


def _metrics_dict(pred: pd.DataFrame, prefix: str = "") -> dict:
    if pred.empty or TARGET_COL not in pred:
        return {}
    y = pred[TARGET_COL].astype(float)
    metrics: dict[str, float] = {}
    for name, col in [
        ("model", "pred_margin"),
        ("ridge", "pred_margin_ridge"),
        ("xgb", "pred_margin_xgb"),
        ("elo", "baseline_elo_margin"),
        ("close", "implied_home_margin"),
    ]:
        if col not in pred:
            continue
        hat = pd.to_numeric(pred[col], errors="coerce")
        mask = y.notna() & hat.notna()
        if mask.sum() == 0:
            continue
        metrics[f"{prefix}{name}_mae"] = float(mean_absolute_error(y[mask], hat[mask]))
        metrics[f"{prefix}{name}_rmse"] = float(np.sqrt(mean_squared_error(y[mask], hat[mask])))
    actual_win = (y > 0).astype(int)
    if "pred_home_wp" in pred:
        wp = pred["pred_home_wp"].clip(1e-6, 1 - 1e-6)
        mask = y.notna() & wp.notna()
        if mask.sum():
            metrics[f"{prefix}wp_brier"] = float(brier_score_loss(actual_win[mask], wp[mask]))
            metrics[f"{prefix}wp_log_loss"] = float(log_loss(actual_win[mask], wp[mask]))
            metrics[f"{prefix}wp_accuracy"] = float(((wp[mask] >= 0.5).astype(int) == actual_win[mask]).mean())
    if "cfbd_home_wp" in pred:
        wp = pd.to_numeric(pred["cfbd_home_wp"], errors="coerce").clip(1e-6, 1 - 1e-6)
        mask = y.notna() & wp.notna()
        if mask.sum():
            metrics[f"{prefix}cfbd_wp_brier"] = float(brier_score_loss(actual_win[mask], wp[mask]))
    ats = ats_record(pred)
    metrics.update({f"{prefix}{k}": v for k, v in ats.items()})
    early = pred[pred.get("week_bucket", "midlate") == "early"]
    mid = pred[pred.get("week_bucket", "midlate") == "midlate"]
    if not early.empty and "pred_margin" in early:
        mask = early[TARGET_COL].notna() & early["pred_margin"].notna()
        if mask.sum():
            metrics[f"{prefix}early_mae"] = float(mean_absolute_error(early.loc[mask, TARGET_COL], early.loc[mask, "pred_margin"]))
    if not mid.empty and "pred_margin" in mid:
        mask = mid[TARGET_COL].notna() & mid["pred_margin"].notna()
        if mask.sum():
            metrics[f"{prefix}midlate_mae"] = float(mean_absolute_error(mid.loc[mask, TARGET_COL], mid.loc[mask, "pred_margin"]))
    return metrics


def ats_record(pred: pd.DataFrame) -> dict[str, float]:
    """Home spread is CFBD convention: negative if home is favored.

    Home covers when home_margin + close_spread > 0.
    """
    if pred.empty or "close_spread" not in pred:
        return {"ats_n": 0}
    margin = pd.to_numeric(pred[TARGET_COL], errors="coerce")
    spread = pd.to_numeric(pred["close_spread"], errors="coerce")
    pick_home = pd.to_numeric(pred.get("pred_margin"), errors="coerce") + spread
    cover = margin + spread
    mask = margin.notna() & spread.notna() & pick_home.notna() & cover.ne(0)
    if mask.sum() == 0:
        return {"ats_n": 0}
    model_home = pick_home[mask] > 0
    home_covers = cover[mask] > 0
    hits = (model_home & home_covers) | (~model_home & ~home_covers)
    # Closing line as a baseline always "picks" the favorite? Use spread as margin forecast.
    close_pick_home = (-spread[mask]) + spread[mask]  # 0 — skip
    # Market ATS baseline: always take home? Not meaningful.
    # Instead: model vs market by whether model side covers.
    return {
        "ats_n": float(mask.sum()),
        "ats_hit_rate": float(hits.mean()),
        "ats_edge_mean": float(pd.to_numeric(pred.loc[mask, "edge"], errors="coerce").mean()),
    }


def _group_metrics(pred: pd.DataFrame, col: str) -> dict:
    out = {}
    if col not in pred:
        return out
    for key, part in pred.groupby(col):
        out[str(key)] = _metrics_dict(part)
    return out


def write_metrics(summary: dict, path: Path | None = None) -> Path:
    config.ensure_dirs()
    path = Path(path or (config.MODELS_DIR / "metrics.json"))
    path.write_text(json.dumps(summary, indent=2, default=_json_default))
    return path


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def run_evaluate(save: bool = True) -> dict:
    scored, summary, models = walk_forward()
    if save and models:
        save_models(models)
        write_metrics(summary)
        config.ensure_dirs()
        scored_path = config.DATA_DIR / "walkforward_predictions.csv"
        if not scored.empty:
            scored.to_csv(scored_path, index=False)
        try:
            from cfb_model.learn import run_learn

            run_learn(scored=scored, db=False, refresh_identity=False)
        except Exception as exc:
            print(f"learn skipped: {exc}")
    return summary
