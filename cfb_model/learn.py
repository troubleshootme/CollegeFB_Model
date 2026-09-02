"""Closed loop: log predictions, score finals, feed mistakes back into the next card.

Nothing here peeks at a game that has not been played. Residuals are computed
only on completed games, then stored as EWMA corrections (recent weeks count
more than 2020). The next `predict` call applies those corrections *before*
kickoff.

What gets learned
- Overall / early-season / mid-late bias (we keep missing homes by 2 → stop)
- Residual scale (sigma) so WP is honest
- Rush vs pass opponent elasticity, when new box scores exist
- Short 'lessons' so a human can see *why* the cards moved
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cfb_model import config
from cfb_model.features import TARGET_COL
from cfb_model.identity import (
    PASS_YARDS_OPPONENT_WEIGHT,
    RUSH_SHARE_OPPONENT_WEIGHT,
    RUSH_YARDS_OPPONENT_WEIGHT,
)

STATE_VERSION = 1
EWMA_ALPHA = 0.18  # ~5-week half-life if you learn every week
MIN_BATCH = 8


def state_path() -> Path:
    config.ensure_dirs()
    return config.LEARNING_DIR / "state.json"


def log_path() -> Path:
    config.ensure_dirs()
    return config.LEARNING_DIR / "predictions.jsonl"


def residuals_path() -> Path:
    config.ensure_dirs()
    return config.LEARNING_DIR / "residuals.csv"


def default_state() -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "n_scored": 0,
        "n_early": 0,
        "n_midlate": 0,
        "sigma": config.DEFAULT_MARGIN_SIGMA,
        "ewma_alpha": EWMA_ALPHA,
        "bias": {"overall": 0.0, "early": 0.0, "midlate": 0.0},
        "elasticity": {
            "rush_yards": RUSH_YARDS_OPPONENT_WEIGHT,
            "pass_yards": PASS_YARDS_OPPONENT_WEIGHT,
            "rush_share": RUSH_SHARE_OPPONENT_WEIGHT,
        },
        "injury": {"scale": 1.0, "n": 0},
        "lessons": [],
        "updated_at": None,
    }


def load_state(path: Path | None = None) -> dict[str, Any]:
    dest = Path(path or state_path())
    if not dest.exists():
        return default_state()
    blob = json.loads(dest.read_text())
    base = default_state()
    base.update({k: v for k, v in blob.items() if k in base or k in {"version", "updated_at", "n_scored", "sigma", "n_early", "n_midlate"}})
    if isinstance(blob.get("bias"), dict):
        base["bias"].update(blob["bias"])
    if isinstance(blob.get("elasticity"), dict):
        base["elasticity"].update(blob["elasticity"])
    if isinstance(blob.get("injury"), dict):
        base.setdefault("injury", {"scale": 1.0, "n": 0}).update(blob["injury"])
    if isinstance(blob.get("lessons"), list):
        base["lessons"] = blob["lessons"][-12:]
    return base


def save_state(state: dict[str, Any], path: Path | None = None) -> Path:
    dest = Path(path or state_path())
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(state, indent=2, default=str))
    return dest


def log_predictions(frame: pd.DataFrame) -> Path:
    """Append one JSON object per game. Re-predicting the same id updates nothing (append-only)."""
    dest = log_path()
    keep = [
        c
        for c in [
            "id",
            "season",
            "week",
            "start_date",
            "home_team",
            "away_team",
            "pred_margin",
            "pred_margin_raw",
            "pred_home_wp",
            "close_spread",
            "identity_stage",
            "week_bucket",
            "home_games_played",
            "away_games_played",
            "home_qb",
            "away_qb",
            "qb_out_points_diff",
            "injury_load_diff",
        ]
        if c in frame.columns
    ]
    now = datetime.now(timezone.utc).isoformat()
    with dest.open("a", encoding="utf-8") as handle:
        for row in frame[keep].to_dict(orient="records"):
            row["logged_at"] = now
            handle.write(json.dumps(row, default=str) + "\n")
    return dest


def score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Residuals for completed games only. residual = actual home margin − prediction."""
    out = frame.copy()
    if TARGET_COL not in out.columns:
        if "home_points" in out and "away_points" in out:
            out[TARGET_COL] = pd.to_numeric(out["home_points"], errors="coerce") - pd.to_numeric(
                out["away_points"], errors="coerce"
            )
        else:
            return out.iloc[0:0]
    completed = out.get("completed", 1)
    if not isinstance(completed, pd.Series):
        completed = pd.Series(completed, index=out.index)
    mask = completed.fillna(0).astype(int).eq(1) & out[TARGET_COL].notna() & pd.to_numeric(
        out.get("pred_margin"), errors="coerce"
    ).notna()
    scored = out.loc[mask].copy()
    if scored.empty:
        return scored
    scored["residual"] = pd.to_numeric(scored[TARGET_COL], errors="coerce") - pd.to_numeric(
        scored["pred_margin_raw"] if "pred_margin_raw" in scored.columns else scored["pred_margin"],
        errors="coerce",
    )
    scored["abs_error"] = scored["residual"].abs()
    if "week_bucket" not in scored.columns:
        scored["week_bucket"] = np.where(pd.to_numeric(scored.get("week"), errors="coerce").fillna(99) <= 4, "early", "midlate")
    games = pd.concat(
        [
            pd.to_numeric(scored.get("home_games_played"), errors="coerce"),
            pd.to_numeric(scored.get("away_games_played"), errors="coerce"),
        ],
        axis=1,
    ).min(axis=1)
    scored["identity_stage"] = np.where(games.fillna(0) <= 0, "preseason", np.where(games < config.EARLY_GAMES, "forming", "set"))
    return scored


def _ewma(old: float, new: float, n: int, alpha: float = EWMA_ALPHA) -> float:
    if n < MIN_BATCH or new is None or (isinstance(new, float) and np.isnan(new)):
        return float(old)
    return float((1.0 - alpha) * old + alpha * new)


def _update_injury_scale(state: dict[str, Any], scored: pd.DataFrame) -> dict[str, Any]:
    """If we keep missing games with a QB out, move the injury prior toward reality."""
    if scored is None or scored.empty:
        return state
    if "qb_out_points_diff" not in scored.columns and "injury_load_diff" not in scored.columns:
        return state
    inj = pd.to_numeric(scored.get("qb_out_points_diff"), errors="coerce")
    if inj is None or inj.isna().all():
        inj = config.QB_OUT_POINTS * pd.to_numeric(scored.get("injury_load_diff"), errors="coerce")
    res = pd.to_numeric(scored.get("residual"), errors="coerce")
    mask = inj.notna() & res.notna() & inj.abs().gt(0.5)
    if int(mask.sum()) < MIN_BATCH:
        return state
    x = inj[mask].to_numpy(dtype=float)
    y = res[mask].to_numpy(dtype=float)
    var = float(np.dot(x, x))
    if var < 1e-6:
        return state
    hat = float(np.clip(-float(np.dot(y, x)) / var, 0.25, 1.75))
    old = float((state.get("injury") or {}).get("scale") or 1.0)
    n = int(mask.sum())
    state["injury"] = {
        "scale": _ewma(old, hat, n),
        "n": int((state.get("injury") or {}).get("n") or 0) + n,
    }
    return state


def update_state(state: dict[str, Any], scored: pd.DataFrame) -> dict[str, Any]:
    if scored is None or scored.empty or "residual" not in scored:
        return state
    out = json.loads(json.dumps(state))  # copy
    alpha = float(out.get("ewma_alpha") or EWMA_ALPHA)
    res = pd.to_numeric(scored["residual"], errors="coerce").dropna()
    if res.empty:
        return out
    out["n_scored"] = int(out.get("n_scored") or 0) + int(len(res))
    out["bias"]["overall"] = _ewma(out["bias"]["overall"], float(res.mean()), len(res), alpha)
    sigma_hat = float(res.std(ddof=1)) if len(res) > 2 else config.DEFAULT_MARGIN_SIGMA
    if np.isfinite(sigma_hat) and sigma_hat > 1:
        out["sigma"] = _ewma(float(out.get("sigma") or config.DEFAULT_MARGIN_SIGMA), sigma_hat, len(res), alpha)
    for bucket, key in (("early", "n_early"), ("midlate", "n_midlate")):
        part = scored[scored.get("week_bucket") == bucket]
        if part.empty or "residual" not in part:
            continue
        mu = float(pd.to_numeric(part["residual"], errors="coerce").mean())
        out["n_scored_bucket"] = out.get("n_scored_bucket") or {}
        out[key] = int(out.get(key) or 0) + int(len(part))
        out["bias"][bucket] = _ewma(out["bias"].get(bucket, 0.0), mu, len(part), alpha)
    out = _update_injury_scale(out, scored)
    out["lessons"] = _lessons(scored, out) + list(out.get("lessons") or [])
    out["lessons"] = out["lessons"][:12]
    out["updated_at"] = datetime.now(timezone.utc).isoformat()
    return out


def _lessons(scored: pd.DataFrame, state: dict[str, Any]) -> list[str]:
    notes = []
    mae = float(scored["abs_error"].mean()) if "abs_error" in scored else float("nan")
    bias = float(scored["residual"].mean())
    n = int(len(scored))
    direction = "underrating homes" if bias > 0.4 else "overrating homes" if bias < -0.4 else "roughly unbiased"
    notes.append(f"Last {n} scored games: MAE {mae:.1f}, {direction} by {bias:+.1f} pts.")
    early = scored[scored.get("week_bucket") == "early"]
    if len(early) >= 5:
        e = float(early["residual"].mean())
        notes.append(
            f"Early season (weeks 1–4): residual {e:+.1f}. Priors (FPI/talent/coach) still carry these cards."
        )
    forming = scored[scored.get("identity_stage") == "forming"]
    settled = scored[scored.get("identity_stage") == "set"]
    if len(forming) >= 5 and len(settled) >= 5:
        notes.append(
            f"Identity still forming: MAE {float(forming['abs_error'].mean()):.1f} vs "
            f"set (week 5+): MAE {float(settled['abs_error'].mean()):.1f}."
        )
    elas = state.get("elasticity") or {}
    notes.append(
        f"Rush yards still opponent-elastic (w={elas.get('rush_yards', RUSH_YARDS_OPPONENT_WEIGHT):.2f}); "
        f"pass yards sticky (w={elas.get('pass_yards', PASS_YARDS_OPPONENT_WEIGHT):.2f})."
    )
    inj = state.get("injury") or {}
    notes.append(
        f"Injury prior scale {float(inj.get('scale') or 1.0):.2f} "
        f"(QB out ≈ {config.QB_OUT_POINTS:.1f} pts; dual-threat sits cost more)."
    )
    return notes


def apply_state(frame: pd.DataFrame, state: dict[str, Any] | None = None, *, injury_prior: bool = True) -> pd.DataFrame:
    """Shift the raw model margin by learned bias. Does not look at this game's score."""
    out = frame.copy()
    state = state or load_state()
    raw = pd.to_numeric(out.get("pred_margin"), errors="coerce")
    out["pred_margin_raw"] = raw
    if "week_bucket" not in out.columns:
        out["week_bucket"] = np.where(
            pd.to_numeric(out.get("week"), errors="coerce").fillna(99) <= 4, "early", "midlate"
        )
    overall = float(state.get("bias", {}).get("overall") or 0.0)
    n_early = int(state.get("n_early") or 0)
    n_late = int(state.get("n_midlate") or 0)
    early_bias = float(state.get("bias", {}).get("early") or 0.0)
    late_bias = float(state.get("bias", {}).get("midlate") or 0.0)
    is_early = out["week_bucket"].astype(str).eq("early")
    bucket = np.where(is_early, early_bias if n_early >= MIN_BATCH else overall, late_bias if n_late >= MIN_BATCH else overall)
    blended = _blend_early_prior(out.assign(pred_margin=raw))
    out["pred_margin"] = blended + pd.to_numeric(pd.Series(bucket, index=out.index), errors="coerce")
    if injury_prior:
        from cfb_model.injuries import injury_prior_margin

        scale = float((state.get("injury") or {}).get("scale") or 1.0)
        out["pred_margin"] = pd.to_numeric(out["pred_margin"], errors="coerce") + injury_prior_margin(out, scale=scale)
        out["injury_prior_applied"] = True
    games = pd.concat(
        [
            pd.to_numeric(out.get("home_games_played"), errors="coerce"),
            pd.to_numeric(out.get("away_games_played"), errors="coerce"),
        ],
        axis=1,
    ).min(axis=1)
    out["identity_stage"] = np.where(
        games.fillna(0) <= 0, "preseason", np.where(games < config.EARLY_GAMES, "forming", "set")
    )
    sigma = float(state.get("sigma") or config.DEFAULT_MARGIN_SIGMA)
    from scipy.stats import norm

    out["pred_home_wp"] = norm.cdf(pd.to_numeric(out["pred_margin"], errors="coerce") / sigma)
    out["learning_applied"] = True
    return out


def _blend_early_prior(frame: pd.DataFrame) -> pd.Series:
    """Week 1 we barely have in-season form — lean on FPI / SP+ / Elo until we do."""
    pred = pd.to_numeric(frame["pred_margin"], errors="coerce")

    def _col(name: str) -> pd.Series:
        if name not in frame.columns:
            return pd.Series(np.nan, index=frame.index)
        return pd.to_numeric(frame[name], errors="coerce")

    n_home = _col("home_games_played").fillna(0)
    n_away = _col("away_games_played").fillna(0)
    n = pd.concat([n_home, n_away], axis=1).min(axis=1).clip(lower=0)
    k = config.SHRINKAGE_K_FORM
    w_prior = k / (k + n)
    fpi = _col("home_prior_fpi") - _col("away_prior_fpi")
    sp = _col("home_prior_sp") - _col("away_prior_sp")
    elo = _col("elo_diff") / config.ELO_MARGIN_SCALE
    prior = fpi.fillna(sp).fillna(elo)
    blended = (1.0 - w_prior) * pred + w_prior * prior
    return blended.where(prior.notna(), pred)


def refresh_elasticity(state: dict[str, Any]) -> dict[str, Any]:
    """Re-estimate rush/pass opponent weights from the warehouse when boxes exist."""
    try:
        from cfb_model.identity import measure_elasticity

        measured = measure_elasticity()
    except Exception:
        return state
    if not measured or measured.get("n_team_games", 0) < 200:
        return state
    rush = measured.get("rush_opponent_weight")
    pas = measured.get("pass_opponent_weight")
    share = measured.get("rush_share_opponent_weight")
    if rush is not None and np.isfinite(rush):
        state["elasticity"]["rush_yards"] = float(rush)
    if pas is not None and np.isfinite(pas):
        state["elasticity"]["pass_yards"] = float(pas)
    if share is not None and np.isfinite(share):
        state["elasticity"]["rush_share"] = float(share)
    return state


def learn_from_frame(frame: pd.DataFrame, *, refresh_identity: bool = False) -> dict[str, Any]:
    scored = score_frame(frame)
    state = load_state()
    state = update_state(state, scored)
    if refresh_identity:
        state = refresh_elasticity(state)
    save_state(state)
    if not scored.empty:
        path = residuals_path()
        scored.to_csv(path, index=False, mode="a" if path.exists() else "w", header=not path.exists())
    return state


def load_logged_predictions() -> pd.DataFrame:
    path = log_path()
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "id" in frame.columns:
        frame = frame.drop_duplicates("id", keep="last")
    return frame


def compact_log() -> None:
    frame = load_logged_predictions()
    dest = log_path()
    if frame.empty:
        return
    with dest.open("w", encoding="utf-8") as handle:
        for row in frame.to_dict(orient="records"):
            handle.write(json.dumps(row, default=str) + "\n")


def run_learn(*, db: bool = True, refresh_identity: bool = True, scored: pd.DataFrame | None = None) -> dict[str, Any]:
    """Score published cards that have gone final, plus optional walk-forward residuals."""
    state = load_state()
    parts = []
    if scored is not None and not scored.empty:
        parts.append(score_frame(scored))
    if db:
        try:
            from cfb_model.features import build_features
            from cfb_model.models import completed_mask, load_models, predict_frame

            frame = build_features()
            done = frame.loc[completed_mask(frame)]
            logged = load_logged_predictions()
            if not logged.empty and not done.empty:
                actuals = done[["id", "home_points", "away_points", "completed", TARGET_COL]].copy() if "id" in done.columns else done
                merged = logged.merge(actuals, on="id", how="inner", suffixes=("", "_act"))
                if TARGET_COL not in merged.columns and f"{TARGET_COL}_act" in merged.columns:
                    merged[TARGET_COL] = merged[f"{TARGET_COL}_act"]
                parts.append(score_frame(merged))
            elif not done.empty and (config.MODELS_DIR / "xgb.json").exists():
                models = load_models()
                done = predict_frame(models, done, apply_learning=False)
                parts.append(score_frame(done))
        except Exception as exc:
            state.setdefault("lessons", [])
            state["lessons"] = [f"Learn skipped warehouse ({exc})."] + list(state.get("lessons") or [])
            save_state(state)
    combined = pd.concat([p for p in parts if p is not None and not p.empty], ignore_index=True) if parts else pd.DataFrame()
    if not combined.empty and "id" in combined.columns:
        combined = combined.drop_duplicates("id", keep="first")  # prefer logged cards
    if not combined.empty:
        state = learn_from_frame(combined, refresh_identity=refresh_identity)
    elif refresh_identity:
        state = refresh_elasticity(state)
        save_state(state)
    compact_log()
    return load_state()
