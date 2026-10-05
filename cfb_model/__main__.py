"""CLI: python -m cfb_model <ingest|migrate|features|train|evaluate|predict>."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cfb_model import config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cfb_model", description="College football prediction model")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest", help="Download CFBD data into data/cfb.db")
    ingest.add_argument("--start-year", type=int, default=config.START_YEAR)
    ingest.add_argument("--end-year", type=int, default=config.END_YEAR)
    ingest.add_argument(
        "--lite",
        action="store_true",
        help="Year-level endpoints only (skip week-looped havoc/box/Elo)",
    )
    ingest.add_argument(
        "--no-weather",
        action="store_true",
        help="Skip CFBD / Open-Meteo weather",
    )
    ingest.add_argument("--no-cache", action="store_true")
    ingest.add_argument("--force", action="store_true", help="Re-pull seasons already stored")
    ingest.add_argument("--no-injuries", action="store_true", help="Skip ESPN injury snapshot")

    sub.add_parser("migrate", help="Import collegeFootball.db into the normalized warehouse")

    feat = sub.add_parser("features", help="Build as-of features and write data/features.csv")
    feat.add_argument("--out", type=Path, default=config.DATA_DIR / "features.csv")

    train = sub.add_parser("train", help="Train Ridge + XGBoost on completed games")
    train.add_argument("--blend", action="store_true", help="Include opening spread as a feature")

    sub.add_parser("evaluate", help="Walk-forward evaluation vs Elo and closing spread")

    pred = sub.add_parser("predict", help="Score a season/week")
    pred.add_argument("--year", type=int, default=config.CURRENT_SEASON)
    pred.add_argument("--week", type=int, required=True)
    pred.add_argument("--out", type=Path, default=None)
    pred.add_argument("--retrain", action="store_true")

    sub.add_parser("learn", help="Score finalized games and update the self-improving state")

    inj = sub.add_parser("injuries", help="Snapshot ESPN injury reports (no CFBD quota)")
    inj.add_argument("--year", type=int, default=config.CURRENT_SEASON)

    prof = sub.add_parser("profile", help="Print a team / coach / QB identity card")
    prof.add_argument("--year", type=int, default=config.CURRENT_SEASON)
    prof.add_argument("--team", type=str, default=None)
    prof.add_argument("--coach", type=str, default=None)
    prof.add_argument("--qb", type=str, default=None)
    prof.add_argument("--write", action="store_true", help="Write HTML catalog under data/profiles")

    args = parser.parse_args(argv)
    config.ensure_dirs()

    if args.cmd == "ingest":
        from cfb_model.ingest import run_ingest

        counts = run_ingest(
            start_year=args.start_year,
            end_year=args.end_year,
            include_weather=not args.no_weather,
            include_injuries=not args.no_injuries,
            full=not args.lite,
            use_cache=not args.no_cache,
            force=args.force,
        )
        print(json.dumps(counts, indent=2))
        return 0

    if args.cmd == "migrate":
        from cfb_model.migrate import run_migrate

        counts = run_migrate()
        print(json.dumps(counts, indent=2))
        return 0

    if args.cmd == "features":
        from cfb_model.features import build_features

        frame = build_features()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.out, index=False)
        print(f"wrote {len(frame)} rows to {args.out}")
        return 0

    if args.cmd == "train":
        from cfb_model.evaluate import walk_forward, write_metrics
        from cfb_model.models import save_models

        scored, summary, models = walk_forward(blend=args.blend)
        save_models(models)
        write_metrics(summary)
        print(json.dumps(summary.get("overall", {}), indent=2))
        print(f"champion={models.get('champion')} n_train={models.get('n_train')}")
        from cfb_model.learn import run_learn

        state = run_learn(scored=scored, db=False, refresh_identity=False)
        if state.get("lessons"):
            print("learn:", state["lessons"][0])
        return 0

    if args.cmd == "evaluate":
        from cfb_model.evaluate import run_evaluate

        summary = run_evaluate(save=True)
        print(json.dumps(summary.get("overall", {}), indent=2))
        print("folds:", json.dumps(summary.get("folds", []), indent=2, default=str)[:2000])
        return 0

    if args.cmd == "learn":
        from cfb_model.learn import run_learn

        state = run_learn(db=True, refresh_identity=True)
        print(json.dumps({k: state[k] for k in ("n_scored", "sigma", "bias", "injury", "lessons") if k in state}, indent=2))
        return 0

    if args.cmd == "injuries":
        from cfb_model.injuries import ingest_injuries

        n = ingest_injuries()
        print(f"stored {n} injury rows")
        return 0

    if args.cmd == "profile":
        return _profile(args)

    if args.cmd == "predict":
        return _predict(args)

    raise ValueError(args.cmd)


def _predict(args) -> int:
    from cfb_model.features import build_features
    from cfb_model.models import completed_mask, load_models, predict_frame, save_models, train_from_db

    models_dir = config.MODELS_DIR
    if args.retrain or not (models_dir / "xgb.json").exists():
        models, frame = train_from_db()
        save_models(models)
    else:
        models = load_models()
        frame = build_features()
    subset = frame[(frame["season"] == args.year) & (frame["week"] == args.week)].copy()
    if subset.empty:
        print(f"No games for {args.year} week {args.week}")
        return 1
    pred = predict_frame(models, subset)
    cols = [
        c
        for c in [
            "season",
            "week",
            "start_date",
            "away_team",
            "home_team",
            "pred_margin",
            "pred_home_wp",
            "pred_spread",
            "spread_label",
            "cover_side",
            "home_qb",
            "away_qb",
            "home_qb_style",
            "away_qb_style",
            "close_spread",
            "edge",
            "confidence",
            "home_injury_note",
            "away_injury_note",
            "learning_applied",
            "home_margin",
            "completed",
        ]
        if c in pred.columns
    ]
    out = pred[cols].sort_values("confidence", ascending=False)
    dest = args.out or (config.DATA_DIR / f"predictions_{args.year}_w{args.week}.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(out.to_string(index=False))
    print(f"wrote {dest}")
    return 0


def _profile(args) -> int:
    from cfb_model.players import build_qb_season_table, qb_card
    from cfb_model.profiles import find_coach, find_team, format_card, load_identity, write_catalog
    from cfb_model import store

    team_table, coach_table = load_identity()
    if args.write:
        dest = write_catalog(team_table, coach_table, config.PROFILES_DIR, season=args.year)
        print(f"wrote {dest}")
    if args.team:
        print(format_card(find_team(team_table, args.team, args.year)))
    if args.coach:
        print(format_card(find_coach(coach_table, args.coach, args.year)))
    if args.qb:
        conn = store.init_schema()
        try:
            stats = store.read_table(conn, "player_season_stats")
            ppa = store.read_table(conn, "player_ppa")
            portal = store.read_table(conn, "transfer_portal")
            talent = store.read_table(conn, "talent")
        finally:
            conn.close()
        qb_table = build_qb_season_table(stats, ppa, portal, talent)
        print(format_card(qb_card(qb_table, args.qb, args.year)))
    if not any([args.team, args.coach, args.qb, args.write]):
        print("Pass --team, --coach, --qb, or --write")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
