"""Last-season replay from SQLite + saved models. No CFBD calls."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.simulate import get_feature_frame, last_completed_season, load_holdout_models, score_slate

TRAIN_FRACTION = 0.75


def replay_week_catalog(slate: pd.DataFrame) -> list[dict[str, Any]]:
    if slate.empty:
        return []
    rows = []
    grouped = slate.groupby("week", dropna=False)
    for week, part in grouped:
        start = pd.to_datetime(part["start_date"], utc=True, errors="coerce").min()
        season_type = "regular"
        if "season_type" in part.columns:
            raw = part["season_type"].dropna()
            if not raw.empty:
                season_type = str(raw.iloc[0] or "regular")
        label = "Bowls / CFP" if season_type != "regular" else f"Week {int(week)}"
        rows.append({"week": int(week), "label": label, "season_type": season_type, "start": start})
    rows.sort(
        key=lambda row: (
            pd.Timestamp("2262-01-01", tz="UTC")
            if pd.isna(row["start"])
            else (row["start"] if getattr(row["start"], "tzinfo", None) else row["start"].tz_localize("UTC")),
            row["week"],
        )
    )
    return [{"week": row["week"], "label": row["label"], "season_type": row["season_type"]} for row in rows]


def replay_window(
    frame: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    season: int | None = None,
) -> dict[str, Any]:
    season = int(season) if season is not None else last_completed_season(frame)
    if season is None:
        raise ValueError("No fully completed season is available to replay.")
    slate = frame.copy()
    if "fbs_vs_fbs" in slate.columns:
        slate = slate[slate["fbs_vs_fbs"].astype(bool)]
    slate = slate[slate["completed"].astype(bool) & (slate["season"] == season)]
    slate = slate.sort_values(["start_date", "game_id"]).reset_index(drop=True)
    if slate.empty:
        raise ValueError(f"No completed FBS games for {season}.")
    cut = min(max(int(len(slate) * train_fraction), 0), len(slate) - 1)
    train = slate.iloc[:cut]
    score = slate.iloc[cut:].copy()
    cutoff = pd.to_datetime(score.iloc[0]["start_date"], utc=True, errors="coerce")
    return {
        "season": season,
        "train_fraction": float(train_fraction),
        "train_count": int(len(train)),
        "score_count": int(len(score)),
        "cutoff": cutoff,
        "train": train,
        "score": score,
    }


def replay_board(week: int | None = None, frame: pd.DataFrame | None = None, models=None) -> dict[str, Any]:
    frame = get_feature_frame() if frame is None else frame
    window = replay_window(frame)
    slate = window["score"]
    options = replay_week_catalog(slate)
    weeks = [row["week"] for row in options]
    selected = int(week) if week is not None else (weeks[0] if weeks else None)
    if selected is not None:
        slate = slate[slate["week"] == selected]
    out_of_sample = True
    if models is None:
        # Deployed models are refit on every completed game, including this season; replaying
        # with them would be in-sample. Use the pre-holdout models when they cover this season.
        held = load_holdout_models()
        if held is not None and int(held["margin"].get("holdout_season", -1)) == int(window["season"]):
            models = held
        else:
            out_of_sample = False
    scored = score_slate(slate, models=models) if not slate.empty else slate
    cutoff = window["cutoff"]
    cutoff_iso = cutoff.isoformat() if hasattr(cutoff, "isoformat") and pd.notna(cutoff) else None
    return {
        "season": window["season"],
        "train_fraction": window["train_fraction"],
        "train_count": window["train_count"],
        "score_count": window["score_count"],
        "cutoff": cutoff_iso,
        "week": selected,
        "weeks": weeks,
        "week_options": options,
        "games": scored,
        "source": "local",
        "out_of_sample": out_of_sample,
    }
