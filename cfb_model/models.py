"""Ridge baseline + XGBoost margin models, with normal-distribution win probabilities."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from cfb_model import config
from cfb_model.features import FEATURE_COLS, TARGET_COL, available_features, build_features

RIDGE_ALPHAS = np.logspace(-2, 3, 16)


def completed_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        frame.get("completed", 1).fillna(0).astype(int).eq(1)
        & frame[TARGET_COL].notna()
        & frame["home_points"].notna()
        & frame["away_points"].notna()
    )


def matrix(frame: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    cols = columns or available_features(frame)
    usable = [c for c in cols if c in frame.columns]
    return frame[usable].apply(pd.to_numeric, errors="coerce"), usable


def make_ridge() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=RIDGE_ALPHAS, cv=5)),
        ]
    )


def make_xgb() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=4,
        reg_lambda=2.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
        tree_method="hist",
    )


def fit_models(train: pd.DataFrame, columns: list[str] | None = None) -> dict:
    X, cols = matrix(train, columns)
    cols = [c for c in cols if X[c].notna().any()]
    X = X[cols]
    y = train[TARGET_COL].astype(float)
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=cols, index=X.index)
    ridge = make_ridge()
    ridge.fit(X, y)  # ridge pipeline already imputes
    xgb = make_xgb()
    xgb.fit(X_imp, y)
    ridge_pred = ridge.predict(X)
    sigma = float(np.std(y - ridge_pred, ddof=1) or config.DEFAULT_MARGIN_SIGMA)
    return {"ridge": ridge, "xgb": xgb, "columns": cols, "sigma": sigma, "imputer": imputer}


def predict_frame(models: dict, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    X, _ = matrix(out, models["columns"])
    X = X.reindex(columns=models["columns"])
    out["pred_margin_ridge"] = models["ridge"].predict(X)
    X_imp = models["imputer"].transform(X)
    out["pred_margin_xgb"] = models["xgb"].predict(X_imp)
    champion = models.get("champion", "xgb")
    out["pred_margin"] = out[f"pred_margin_{champion}"]
    sigma = float(models.get("sigma") or config.DEFAULT_MARGIN_SIGMA)
    out["pred_home_wp"] = norm.cdf(out["pred_margin"] / sigma)
    out["pred_spread"] = -out["pred_margin"]
    close = pd.to_numeric(out.get("close_spread"), errors="coerce")
    implied = -close
    out["implied_home_margin"] = implied
    out["edge"] = out["pred_margin"] - implied
    out["confidence"] = (out["edge"].abs() / sigma).replace([np.inf, -np.inf], np.nan)
    out["baseline_elo_margin"] = pd.to_numeric(out.get("elo_diff"), errors="coerce") / config.ELO_MARGIN_SCALE
    return out


def ridge_coefficients(models: dict) -> pd.DataFrame:
    ridge: Pipeline = models["ridge"]
    coef = ridge.named_steps["model"].coef_
    return pd.DataFrame({"feature": models["columns"], "ridge_coef": coef}).sort_values(
        "ridge_coef", key=np.abs, ascending=False
    )


def xgb_importances(models: dict) -> pd.DataFrame:
    booster: XGBRegressor = models["xgb"]
    return pd.DataFrame(
        {"feature": models["columns"], "xgb_gain": booster.feature_importances_}
    ).sort_values("xgb_gain", ascending=False)


def save_models(models: dict, directory: Path | None = None) -> Path:
    config.ensure_dirs()
    directory = Path(directory or config.MODELS_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "ridge": models["ridge"],
            "imputer": models["imputer"],
            "columns": models["columns"],
            "sigma": models["sigma"],
            "champion": models.get("champion", "xgb"),
            "n_train": models.get("n_train"),
        },
        directory / "ridge.joblib",
    )
    models["xgb"].save_model(directory / "xgb.json")
    meta = {
        "columns": models["columns"],
        "sigma": models["sigma"],
        "champion": models.get("champion", "xgb"),
        "n_train": models.get("n_train"),
    }
    (directory / "model_meta.json").write_text(json.dumps(meta, indent=2))
    return directory


def load_models(directory: Path | None = None) -> dict:
    directory = Path(directory or config.MODELS_DIR)
    blob = joblib.load(directory / "ridge.joblib")
    if isinstance(blob, dict) and "ridge" in blob:
        ridge = blob["ridge"]
        imputer = blob["imputer"]
        meta_columns = blob.get("columns")
        sigma = blob.get("sigma", config.DEFAULT_MARGIN_SIGMA)
        champion = blob.get("champion", "xgb")
    else:
        ridge = blob
        imputer = SimpleImputer(strategy="median")
        meta_columns = None
        sigma = config.DEFAULT_MARGIN_SIGMA
        champion = "xgb"
    meta = json.loads((directory / "model_meta.json").read_text())
    xgb = make_xgb()
    xgb.load_model(directory / "xgb.json")
    columns = meta_columns or meta["columns"]
    if imputer is not None and not hasattr(imputer, "statistics_"):
        # fitted imputer is required; reconstruct from zeros if needed
        imputer.fit(pd.DataFrame(np.zeros((2, len(columns))), columns=columns))
    return {
        "ridge": ridge,
        "xgb": xgb,
        "imputer": imputer,
        "columns": columns,
        "sigma": meta.get("sigma", sigma),
        "champion": meta.get("champion", champion),
    }


def train_from_db(db_path: Path | None = None, blend: bool = False) -> tuple[dict, pd.DataFrame]:
    frame = build_features(db_path=db_path)
    train = frame.loc[completed_mask(frame)].copy()
    if train.empty:
        raise ValueError("No completed games available to train on")
    cols = available_features(train, blend=blend)
    models = fit_models(train, cols)
    models["n_train"] = int(len(train))
    return models, frame
