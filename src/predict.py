"""Score upcoming games with saved models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import MODELS_DIR
from src.simulate import get_feature_frame, load_models, score_slate


def predict_week(season: int, week: int) -> pd.DataFrame:
    load_models()
    frame = get_feature_frame()
    slate = frame[
        ~frame["completed"].astype(bool)
        & frame["fbs_vs_fbs"].astype(bool)
        & (frame["season"] == season)
        & (frame["week"] == week)
    ].copy()
    if slate.empty:
        raise RuntimeError(f"No upcoming FBS games for season {season} week {week}.")
    return score_slate(slate.sort_values("start_date"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--week", type=int, default=1)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()
    board = predict_week(args.season, args.week)
    path = args.csv or (MODELS_DIR / f"week{args.week}_{args.season}.csv")
    board.to_csv(path, index=False)
    cols = [
        "kick_label",
        "away_team",
        "home_team",
        "spread",
        "over_under",
        "pred_away_points",
        "pred_home_points",
        "pick",
        "pick_prob",
        "cover_side",
        "ats_edge",
        "wx_temp_max",
        "wx_precip",
    ]
    show = [col for col in cols if col in board.columns]
    print(board[show].to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print(f"\nWrote {path} ({len(board)} games)")
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        hgb = metrics.get("hgb_margin") or {}
        ats = (metrics.get("ats_from_margin_model") or {}).get("ats_ge_0") or {}
        print(
            f"Holdout winner acc {hgb.get('winner_accuracy')} MAE {hgb.get('mae')} "
            f"(market MAE {hgb.get('market_mae')}); margin-model ATS {ats.get('accuracy')} n={ats.get('n')}"
        )


if __name__ == "__main__":
    main()
