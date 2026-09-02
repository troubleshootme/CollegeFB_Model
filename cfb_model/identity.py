"""Scheme identity: mix vs opponent, and opponent-of-opponent.

College football play-calling is not a team in isolation. A 55% rush rate
can mean "this offense is ground-and-pound" or "this defense takes away the
pass" or "this offense's other opponents all stuffed the run, so 55% is
actually pass-heavy for the schedule."

Ply 0 — raw
    What share of *this team's* plays were rushes / passes.

Ply 1 — vs the opponent's average
    Offense: our rush share minus what this defense *typically faces*.
    Defense: rush share we faced minus what this offense *typically does*.
    Inverse of each other; both are necessary.

Ply 2 — opponent of opponent
    The defense's "typically faces 48% rush" is itself a schedule. If they
    played three option teams, 48% is pass-leaning. We take the offenses
    they faced (our opponent's opponents) and use *their* identities, then
    iterate (SRS-style) so each team's number is net of who they played
    *and* who those teams played.

Pregame matchup
    run_fit  = off_rush_2ply + def_rush_2ply   (we want to run, they get run on)
    pass_fit = -run_fit

Elasticity (measured 2020–2024, n=10,412 team-games)
    Rushing yards move with the opponent (blend weight 0.35 toward D allowed).
    Passing yards do not (weight 0.15). It is common to hold a team under 80%
    of their rushing average and uncommon to do the same to their passing
    average. Play-call *share* stays with identity (weight 0.10).

All season-level numbers used as priors for season S come from seasons < S.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cfb_model.config import COACH_TENDENCY_K, TENDENCY_YEARS
from cfb_model.util import shrink_series

LEAGUE_RUSH = 0.48
IDENTITY_ITERS = 8
# Measured on 10,412 FBS team-games (2020–2024 box scores):
#   CV of (game / own average):  rush yards 0.53  vs  pass yards 0.40
#   Held under 80% of own avg:   rush 37%         vs  pass 30%
#   Corr(pct of own avg, D allowed): rush 0.24    vs  pass 0.04
#   Best blend actual ≈ (1-w)*own + w*opponent_allowed:
#     rush yards w=0.35, pass yards w=0.15, rush *share* w=0.10
# Passing production is sticky. Rushing yards move with who you play.
# Play-call mix (share) stays closer to identity than yards do.
RUSH_YARDS_OPPONENT_WEIGHT = 0.35
PASS_YARDS_OPPONENT_WEIGHT = 0.15
RUSH_SHARE_OPPONENT_WEIGHT = 0.10
RUSH_OPPONENT_WEIGHT = RUSH_YARDS_OPPONENT_WEIGHT
PASS_OPPONENT_WEIGHT = PASS_YARDS_OPPONENT_WEIGHT


def blend_own_and_opponent(own: float, opponent: float, weight: float) -> float:
    """weight=1 is 100% opponent quality; weight=0 is the team's own average."""
    if own is None or (isinstance(own, float) and np.isnan(own)):
        return opponent
    if opponent is None or (isinstance(opponent, float) and np.isnan(opponent)):
        return own
    return (1.0 - weight) * float(own) + weight * float(opponent)


def expected_rush_share(
    off_rush: float,
    def_rush_allowed: float,
    *,
    weight: float = RUSH_SHARE_OPPONENT_WEIGHT,
) -> float:
    """Play-call mix is mostly who they are; D only tugs it slightly."""
    return float(np.clip(blend_own_and_opponent(off_rush, def_rush_allowed, weight), 0.22, 0.78))


def expected_rush_yards(
    off_rush_avg: float,
    def_rush_allowed: float,
    *,
    weight: float = RUSH_YARDS_OPPONENT_WEIGHT,
) -> float:
    """Rushing yards are opponent-elastic — stout run D's suppress the average."""
    return blend_own_and_opponent(off_rush_avg, def_rush_allowed, weight)


def expected_pass_yards(
    off_pass_avg: float,
    def_pass_allowed: float,
    *,
    weight: float = PASS_YARDS_OPPONENT_WEIGHT,
) -> float:
    """Passing yards stay close to the offense's own average regardless of D."""
    return blend_own_and_opponent(off_pass_avg, def_pass_allowed, weight)

# Human-readable map of every angle we store. Keys are column names.
ANGLE_MAP: list[dict[str, str]] = [
    {
        "key": "rush_share",
        "ply": "0",
        "side": "offense",
        "meaning": "Raw rush play share (attempts / rush+pass).",
    },
    {
        "key": "def_rush_allowed",
        "ply": "0",
        "side": "defense",
        "meaning": "Raw rush share faced (what opponents called against us).",
    },
    {
        "key": "off_rush_vs_def",
        "ply": "1",
        "side": "offense",
        "meaning": "Our rush share minus what this defense typically faces. + we imposed the run.",
    },
    {
        "key": "def_rush_vs_off",
        "ply": "1",
        "side": "defense",
        "meaning": "Rush share we faced minus what this offense typically does. + they ran more than usual (we made them run or they attacked the run).",
    },
    {
        "key": "off_pass_vs_def",
        "ply": "1",
        "side": "offense",
        "meaning": "Inverse of off_rush_vs_def. + we threw more than this D usually sees.",
    },
    {
        "key": "def_pass_vs_off",
        "ply": "1",
        "side": "defense",
        "meaning": "Inverse of def_rush_vs_off. + they threw more than they usually do.",
    },
    {
        "key": "off_rush_2ply",
        "ply": "2",
        "side": "offense",
        "meaning": "Schedule-net rush identity: iterated vs opponent-of-opponent defenses.",
    },
    {
        "key": "def_rush_2ply",
        "ply": "2",
        "side": "defense",
        "meaning": "Schedule-net 'make you run' identity: iterated vs opponent-of-opponent offenses.",
    },
    {
        "key": "run_fit",
        "ply": "matchup",
        "side": "both",
        "meaning": "off_rush_2ply + opponent def_rush_2ply. High = this game should be run-heavy.",
    },
]


def scheme_game_panel(games: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game with our mix and the opponent's mix."""
    stats = team_stats.copy()
    if "team" in stats.columns and "school" not in stats.columns:
        stats = stats.rename(columns={"team": "school"})
    stats["rushing_attempts"] = pd.to_numeric(stats.get("rushing_attempts"), errors="coerce")
    stats["pass_attempts"] = pd.to_numeric(stats.get("pass_attempts"), errors="coerce")
    stats["plays"] = stats["rushing_attempts"].fillna(0) + stats["pass_attempts"].fillna(0)
    stats["rush_share"] = np.where(stats["plays"] > 0, stats["rushing_attempts"] / stats["plays"], np.nan)
    if stats.empty or "game_id" not in stats.columns:
        return pd.DataFrame()

    paired = stats.merge(
        stats[["game_id", "school", "rush_share", "plays", "rushing_attempts", "pass_attempts"]].rename(
            columns={
                "school": "opponent",
                "rush_share": "opp_rush_share",
                "plays": "opp_plays",
                "rushing_attempts": "opp_rush_att",
                "pass_attempts": "opp_pass_att",
            }
        ),
        on="game_id",
        how="left",
    )
    paired = paired[paired["school"] != paired["opponent"]].copy()

    g = games.copy()
    if "game_id" not in g.columns and "id" in g.columns:
        g = g.rename(columns={"id": "game_id"})
    meta_cols = [c for c in ("game_id", "season", "week", "start_date") if c in g.columns]
    if meta_cols:
        paired = paired.merge(g[meta_cols].drop_duplicates("game_id"), on="game_id", how="left")
    if "season" not in paired.columns:
        paired["season"] = np.nan
    return paired


def add_one_ply(panel: pd.DataFrame) -> pd.DataFrame:
    """Ply 1: vs opponent average, excluding this game (no self-leak inside the season)."""
    if panel.empty:
        return panel
    frame = panel.copy()
    totals = (
        frame.groupby(["season", "school"], as_index=False)
        .agg(
            season_rush=("rushing_attempts", "sum"),
            season_plays=("plays", "sum"),
            season_faced_rush=("opp_rush_att", "sum"),
            season_faced_plays=("opp_plays", "sum"),
        )
    )
    off = totals.rename(
        columns={
            "school": "opponent",
            "season_rush": "opp_season_rush",
            "season_plays": "opp_season_plays",
        }
    )[["season", "opponent", "opp_season_rush", "opp_season_plays"]]
    deff = totals.rename(
        columns={
            "school": "opponent",
            "season_faced_rush": "opp_faced_rush",
            "season_faced_plays": "opp_faced_plays",
        }
    )[["season", "opponent", "opp_faced_rush", "opp_faced_plays"]]
    frame = frame.merge(off, on=["season", "opponent"], how="left").merge(deff, on=["season", "opponent"], how="left")
    # Opponent offense identity excluding this game
    ex_off_plays = frame["opp_season_plays"] - frame["opp_plays"]
    ex_off_rush = frame["opp_season_rush"] - frame["opp_rush_att"]
    frame["opp_off_avg_ex"] = np.where(ex_off_plays > 0, ex_off_rush / ex_off_plays, np.nan)
    # Opponent defense typical faced mix excluding this game
    ex_def_plays = frame["opp_faced_plays"] - frame["plays"]
    ex_def_rush = frame["opp_faced_rush"] - frame["rushing_attempts"]
    frame["opp_def_avg_ex"] = np.where(ex_def_plays > 0, ex_def_rush / ex_def_plays, np.nan)

    frame["off_rush_vs_def"] = frame["rush_share"] - frame["opp_def_avg_ex"]
    frame["off_pass_vs_def"] = -frame["off_rush_vs_def"]
    frame["def_rush_vs_off"] = frame["opp_rush_share"] - frame["opp_off_avg_ex"]
    frame["def_pass_vs_off"] = -frame["def_rush_vs_off"]
    frame["def_rush_allowed"] = frame["opp_rush_share"]
    return frame


def iterative_two_ply(panel: pd.DataFrame, *, iters: int = IDENTITY_ITERS) -> pd.DataFrame:
    """Ply 2: SRS-style iteration so identities are net of opponent-of-opponent.

    O_i ← mean_j (rush_ij − (D_j − μ))
    D_j ← mean_i (faced_ij − (O_i − μ))
    """
    if panel.empty:
        return pd.DataFrame(columns=["season", "school", "off_rush_2ply", "def_rush_2ply", "games"])
    mu = float(panel["rush_share"].mean())
    if not np.isfinite(mu):
        mu = LEAGUE_RUSH
    keys = panel[["season", "school"]].drop_duplicates()
    off = panel.groupby(["season", "school"])["rush_share"].mean().rename("off_rush_2ply")
    deff = panel.groupby(["season", "school"])["opp_rush_share"].mean().rename("def_rush_2ply")
    ident = keys.merge(off.reset_index(), on=["season", "school"], how="left").merge(
        deff.reset_index(), on=["season", "school"], how="left"
    )
    ident["off_rush_2ply"] = ident["off_rush_2ply"].fillna(mu)
    ident["def_rush_2ply"] = ident["def_rush_2ply"].fillna(mu)

    work = panel[["season", "school", "opponent", "rush_share", "opp_rush_share"]].copy()
    for _ in range(iters):
        o_map = ident.set_index(["season", "school"])["off_rush_2ply"]
        d_map = ident.set_index(["season", "school"])["def_rush_2ply"]
        work["d_opp"] = [d_map.get((s, o), mu) for s, o in zip(work["season"], work["opponent"])]
        work["o_opp"] = [o_map.get((s, o), mu) for s, o in zip(work["season"], work["opponent"])]
        work["off_adj"] = work["rush_share"] - (work["d_opp"] - mu)
        work["def_adj"] = work["opp_rush_share"] - (work["o_opp"] - mu)
        ident = (
            work.groupby(["season", "school"], as_index=False)
            .agg(off_rush_2ply=("off_adj", "mean"), def_rush_2ply=("def_adj", "mean"), games=("rush_share", "size"))
        )
    n = ident["games"] if "games" in ident else pd.Series(1, index=ident.index)
    ident["off_rush_2ply"] = shrink_series(
        ident["off_rush_2ply"], pd.Series(mu, index=ident.index), n, COACH_TENDENCY_K
    )
    ident["def_rush_2ply"] = shrink_series(
        ident["def_rush_2ply"], pd.Series(mu, index=ident.index), n, COACH_TENDENCY_K
    )
    ident["off_pass_2ply"] = 1.0 - ident["off_rush_2ply"]
    ident["def_pass_2ply"] = 1.0 - ident["def_rush_2ply"]
    return ident


def season_scheme_table(games: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    """Season-level identity with ply 0/1/2. Safe to shift(1) for next-season priors."""
    panel = add_one_ply(scheme_game_panel(games, team_stats))
    if panel.empty:
        return pd.DataFrame()
    ply1 = panel.groupby(["season", "school"], as_index=False).agg(
        rush_share=("rush_share", "mean"),
        def_rush_allowed=("def_rush_allowed", "mean"),
        off_rush_vs_def=("off_rush_vs_def", "mean"),
        def_rush_vs_off=("def_rush_vs_off", "mean"),
        off_pass_vs_def=("off_pass_vs_def", "mean"),
        def_pass_vs_off=("def_pass_vs_off", "mean"),
        games=("game_id", "nunique"),
    )
    ply2 = iterative_two_ply(panel)
    out = ply1.merge(ply2, on=["season", "school"], how="left", suffixes=("", "_n"))
    out["def_rush_vs_opp"] = out["def_rush_vs_off"]
    out["def_pass_vs_opp"] = out["def_pass_vs_off"]
    return out


def prior_scheme_table(season_table: pd.DataFrame, *, years: int = TENDENCY_YEARS) -> pd.DataFrame:
    """Leakage-safe: for season S, identities from S-years … S-1 only."""
    if season_table is None or season_table.empty:
        return pd.DataFrame()
    frame = season_table.sort_values(["school", "season"]).copy()
    metrics = [
        "rush_share",
        "def_rush_allowed",
        "off_rush_vs_def",
        "def_rush_vs_off",
        "off_rush_2ply",
        "def_rush_2ply",
        "off_pass_vs_def",
        "def_pass_vs_off",
    ]
    for col in metrics:
        if col not in frame.columns:
            continue
        frame[f"prior_{col}"] = frame.groupby("school")[col].transform(
            lambda s: s.shift(1).rolling(years, min_periods=1).mean()
        )
    return frame


def matchup_fits(home: pd.Series, away: pd.Series) -> dict[str, float]:
    """Pregame: will this look like a run game or a pass game?"""

    def _g(row: pd.Series, *names: str, default: float = LEAGUE_RUSH) -> float:
        for name in names:
            if name in row.index and pd.notna(row.get(name)):
                return float(row.get(name))
        return default

    h_off = _g(home, "off_rush_2ply", "prior_off_rush_2ply", "rush_share")
    a_off = _g(away, "off_rush_2ply", "prior_off_rush_2ply", "rush_share")
    h_def = _g(home, "def_rush_2ply", "prior_def_rush_2ply", "def_rush_allowed")
    a_def = _g(away, "def_rush_2ply", "prior_def_rush_2ply", "def_rush_allowed")
    run_fit_home = (h_off - LEAGUE_RUSH) + (a_def - LEAGUE_RUSH)
    run_fit_away = (a_off - LEAGUE_RUSH) + (h_def - LEAGUE_RUSH)
    return {
        "home_run_fit": run_fit_home,
        "away_run_fit": run_fit_away,
        "run_fit_diff": run_fit_home - run_fit_away,
        "expected_home_rush_share": float(np.clip(LEAGUE_RUSH + run_fit_home, 0.28, 0.72)),
        "expected_away_rush_share": float(np.clip(LEAGUE_RUSH + run_fit_away, 0.28, 0.72)),
    }


def attach_scheme_features(frame: pd.DataFrame, games: pd.DataFrame, team_stats: pd.DataFrame | None) -> pd.DataFrame:
    """Prior-season 2-ply scheme and run_fit on a game frame. Seasons < S only."""
    out = frame.copy()
    for col in ("run_fit_diff", "off_rush_2ply_diff", "def_rush_2ply_diff"):
        if col not in out.columns:
            out[col] = np.nan
    if team_stats is None or team_stats.empty or games is None or games.empty:
        return out
    stats = team_stats.rename(columns={"team": "school"}) if "school" not in team_stats.columns else team_stats
    season_tbl = season_scheme_table(games, stats)
    priors = prior_scheme_table(season_tbl)
    if priors.empty:
        return out
    hp = priors.rename(columns={"school": "home_team"})
    ap = priors.rename(columns={"school": "away_team"})
    hcols = ["season", "home_team"] + [c for c in hp.columns if c.startswith("prior_")]
    acols = ["season", "away_team"] + [c for c in ap.columns if c.startswith("prior_")]
    hp = hp[hcols].rename(columns={c: f"home_{c}" for c in hcols if c.startswith("prior_")})
    ap = ap[acols].rename(columns={c: f"away_{c}" for c in acols if c.startswith("prior_")})
    out = out.merge(hp, on=["season", "home_team"], how="left").merge(ap, on=["season", "away_team"], how="left")
    out["off_rush_2ply_diff"] = pd.to_numeric(out.get("home_prior_off_rush_2ply"), errors="coerce") - pd.to_numeric(
        out.get("away_prior_off_rush_2ply"), errors="coerce"
    )
    out["def_rush_2ply_diff"] = pd.to_numeric(out.get("home_prior_def_rush_2ply"), errors="coerce") - pd.to_numeric(
        out.get("away_prior_def_rush_2ply"), errors="coerce"
    )
    home_fit = (pd.to_numeric(out.get("home_prior_off_rush_2ply"), errors="coerce") - LEAGUE_RUSH) + (
        pd.to_numeric(out.get("away_prior_def_rush_2ply"), errors="coerce") - LEAGUE_RUSH
    )
    away_fit = (pd.to_numeric(out.get("away_prior_off_rush_2ply"), errors="coerce") - LEAGUE_RUSH) + (
        pd.to_numeric(out.get("home_prior_def_rush_2ply"), errors="coerce") - LEAGUE_RUSH
    )
    out["run_fit_diff"] = home_fit - away_fit
    if "id" in out.columns and out["id"].duplicated().any():
        out = out.drop_duplicates("id", keep="first")
    return out


def tune_legacy_weights(legacy_path: Path | None = None) -> dict[str, Any]:
    """Estimate which ply predicts next-season margin. Uses collegeFootball.db box scores."""
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_absolute_error
    from sklearn.preprocessing import StandardScaler

    from cfb_model import config
    from cfb_model.ingest.flatten import flatten_legacy_box
    from cfb_model.migrate import STAT_TABLES, LEGACY_GAME_YEARS

    import sqlite3

    path = Path(legacy_path or config.LEGACY_DB_PATH)
    if not path.exists():
        return {"error": f"missing {path}"}
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        frames = []
        for year, table in STAT_TABLES.items():
            raw = pd.read_sql(f'SELECT * FROM "{table}"', conn)
            rows = [flatten_legacy_box(rec) for rec in raw.to_dict(orient="records")]
            frame = pd.DataFrame(rows).dropna(subset=["game_id", "team"])
            frame["season"] = year
            frames.append(frame)
    finally:
        conn.close()
    stats = pd.concat(frames, ignore_index=True)
    stats = stats.rename(columns={"team": "school"})
    dummy_games = stats[["game_id", "season"]].drop_duplicates()
    season_tbl = season_scheme_table(dummy_games, stats)
    if season_tbl.empty:
        return {"error": "empty scheme table"}
    priors = prior_scheme_table(season_tbl)

    # Game-level margin from the two box rows
    home = stats[stats.get("home_away", "") == "home"] if "home_away" in stats else stats.iloc[0:0]
    if "home_away" not in stats.columns:
        return {"error": "no home_away in box stats"}
    home = stats[stats["home_away"].astype(str).str.lower().eq("home")][
        ["game_id", "season", "school", "points"]
    ].rename(columns={"school": "home_team", "points": "home_points"})
    away = stats[stats["home_away"].astype(str).str.lower().eq("away")][
        ["game_id", "season", "school", "points"]
    ].rename(columns={"school": "away_team", "points": "away_points"})
    games = home.merge(away, on=["game_id", "season"])
    games["home_margin"] = pd.to_numeric(games["home_points"], errors="coerce") - pd.to_numeric(
        games["away_points"], errors="coerce"
    )
    keep = [
        "season",
        "school",
        "prior_rush_share",
        "prior_def_rush_allowed",
        "prior_off_rush_vs_def",
        "prior_def_rush_vs_off",
        "prior_off_rush_2ply",
        "prior_def_rush_2ply",
    ]
    keep = [c for c in keep if c in priors.columns]
    hp = priors[keep].rename(columns={"school": "home_team", **{c: f"home_{c}" for c in keep if c not in {"season", "school"}}})
    ap = priors[keep].rename(columns={"school": "away_team", **{c: f"away_{c}" for c in keep if c not in {"season", "school"}}})
    # fix rename: school -> home_team already, prior_* -> home_prior_*
    hp = priors.rename(columns={"school": "home_team"})
    ap = priors.rename(columns={"school": "away_team"})
    hcols = ["season", "home_team"] + [c for c in hp.columns if c.startswith("prior_")]
    acols = ["season", "away_team"] + [c for c in ap.columns if c.startswith("prior_")]
    hp = hp[hcols].rename(columns={c: f"home_{c}" for c in hcols if c.startswith("prior_")})
    ap = ap[acols].rename(columns={c: f"away_{c}" for c in acols if c.startswith("prior_")})
    merged = games.merge(hp, on=["season", "home_team"], how="left").merge(ap, on=["season", "away_team"], how="left")
    merged["off_2ply_diff"] = merged["home_prior_off_rush_2ply"] - merged["away_prior_off_rush_2ply"]
    merged["def_2ply_diff"] = merged["home_prior_def_rush_2ply"] - merged["away_prior_def_rush_2ply"]
    merged["off_1ply_diff"] = merged["home_prior_off_rush_vs_def"] - merged["away_prior_off_rush_vs_def"]
    merged["def_1ply_diff"] = merged["home_prior_def_rush_vs_off"] - merged["away_prior_def_rush_vs_off"]
    merged["raw_rush_diff"] = merged["home_prior_rush_share"] - merged["away_prior_rush_share"]
    merged["run_fit"] = (merged["home_prior_off_rush_2ply"] - LEAGUE_RUSH) + (
        merged["away_prior_def_rush_2ply"] - LEAGUE_RUSH
    )
    merged["run_fit_diff"] = merged["run_fit"] - (
        (merged["away_prior_off_rush_2ply"] - LEAGUE_RUSH) + (merged["home_prior_def_rush_2ply"] - LEAGUE_RUSH)
    )
    feat_cols = ["off_2ply_diff", "def_2ply_diff", "off_1ply_diff", "def_1ply_diff", "raw_rush_diff", "run_fit_diff"]
    train = merged.dropna(subset=["home_margin"] + feat_cols)
    # Need prior season: drop first year
    train = train[train["season"] > min(LEGACY_GAME_YEARS)]
    corrs = {c: float(train[c].corr(train["home_margin"])) for c in feat_cols}
    scaler = StandardScaler()
    X = scaler.fit_transform(train[feat_cols])
    y = train["home_margin"].astype(float)
    ridge = RidgeCV(alphas=np.logspace(-2, 3, 12))
    ridge.fit(X, y)
    coefs = {c: float(w) for c, w in zip(feat_cols, ridge.coef_)}
    pred = ridge.predict(X)
    baseline = float(np.abs(y - y.mean()).mean())
    mae = float(mean_absolute_error(y, pred))
    # Normalize |coef| to blend weights that sum to 1
    abs_c = {k: abs(v) for k, v in coefs.items()}
    total = sum(abs_c.values()) or 1.0
    blend = {k: abs_c[k] / total for k in feat_cols}
    return {
        "n_games": int(len(train)),
        "years": sorted(int(y) for y in train["season"].unique()),
        "correlations": corrs,
        "ridge_coefs": coefs,
        "ridge_alpha": float(ridge.alpha_),
        "mae": mae,
        "mae_mean_baseline": baseline,
        "blend_weights": blend,
        "angles": ANGLE_MAP,
    }


def load_legacy_boxes(legacy_path: Path | None = None) -> pd.DataFrame:
    import sqlite3

    from cfb_model import config
    from cfb_model.ingest.flatten import flatten_legacy_box
    from cfb_model.migrate import STAT_TABLES

    path = Path(legacy_path or config.LEGACY_DB_PATH)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        frames = []
        for year, table in STAT_TABLES.items():
            raw = pd.read_sql(f'SELECT * FROM "{table}"', conn)
            rows = [flatten_legacy_box(rec) for rec in raw.to_dict(orient="records")]
            frame = pd.DataFrame(rows).dropna(subset=["game_id", "team"])
            frame["season"] = year
            frames.append(frame)
    finally:
        conn.close()
    return pd.concat(frames, ignore_index=True).rename(columns={"team": "school"})


def _best_opponent_weight(own: pd.Series, opp: pd.Series, actual: pd.Series) -> tuple[float, float]:
    """Grid-search w in actual ≈ (1-w)*own + w*opp. Returns (w, mae)."""
    mask = own.notna() & opp.notna() & actual.notna() & (own.abs() > 1e-6)
    if mask.sum() < 50:
        return float("nan"), float("nan")
    y = actual[mask].to_numpy(dtype=float)
    o = own[mask].to_numpy(dtype=float)
    d = opp[mask].to_numpy(dtype=float)
    best_w, best_mae = 0.0, float("inf")
    for w in np.linspace(0.0, 1.0, 21):
        pred = (1.0 - w) * o + w * d
        mae = float(np.mean(np.abs(y - pred)))
        if mae < best_mae:
            best_mae, best_w = mae, float(w)
    return best_w, best_mae


def measure_elasticity(legacy_path: Path | None = None) -> dict[str, Any]:
    """Rushing vs passing: how much does the opponent move you off your own average?

    Hypothesis we encode in the model: it is much easier to hold a team below
    their rushing average than their passing average. Passing production is
    sticky; rushing volume and yards swing with who you play (and who they played).
    """
    stats = load_legacy_boxes(legacy_path)
    stats["rushing_yards"] = pd.to_numeric(stats["rushing_yards"], errors="coerce")
    stats["net_passing_yards"] = pd.to_numeric(stats["net_passing_yards"], errors="coerce")
    stats["rushing_attempts"] = pd.to_numeric(stats["rushing_attempts"], errors="coerce")
    stats["pass_attempts"] = pd.to_numeric(stats["pass_attempts"], errors="coerce")
    stats["plays"] = stats["rushing_attempts"].fillna(0) + stats["pass_attempts"].fillna(0)
    stats["rush_share"] = np.where(stats["plays"] > 0, stats["rushing_attempts"] / stats["plays"], np.nan)

    # Opponent row (the other team in the same game) = what they *did* and what they *allow*
    opp = stats.rename(
        columns={
            "school": "opponent",
            "rushing_yards": "opp_rush_yds",
            "net_passing_yards": "opp_pass_yds",
            "rushing_attempts": "opp_rush_att",
            "pass_attempts": "opp_pass_att",
            "rush_share": "opp_rush_share",
        }
    )[["game_id", "opponent", "opp_rush_yds", "opp_pass_yds", "opp_rush_att", "opp_pass_att", "opp_rush_share"]]
    panel = stats.merge(opp, on="game_id")
    panel = panel[panel["school"] != panel["opponent"]].copy()

    # Season totals so we can drop this game (no self-leak)
    tot = panel.groupby(["season", "school"], as_index=False).agg(
        n=("game_id", "nunique"),
        rush_yds=("rushing_yards", "sum"),
        pass_yds=("net_passing_yards", "sum"),
        rush_att=("rushing_attempts", "sum"),
        pass_att=("pass_attempts", "sum"),
        plays=("plays", "sum"),
        allowed_rush_yds=("opp_rush_yds", "sum"),
        allowed_pass_yds=("opp_pass_yds", "sum"),
        allowed_rush_att=("opp_rush_att", "sum"),
        faced_plays=("opp_rush_att", "sum"),  # placeholder replaced below
    )
    # faced plays = opponent plays = our defensive snaps approx opp rush+pass att
    faced = panel.groupby(["season", "school"], as_index=False).agg(
        faced_plays=("opp_rush_att", "sum"),
        faced_pass=("opp_pass_att", "sum"),
    )
    faced["faced_plays"] = faced["faced_plays"] + faced["faced_pass"]
    tot = tot.drop(columns=["faced_plays"]).merge(faced[["season", "school", "faced_plays"]], on=["season", "school"])

    me = tot.rename(columns={c: f"own_{c}" for c in tot.columns if c not in {"season", "school"}})
    you = tot.rename(
        columns={"school": "opponent", **{c: f"def_{c}" for c in tot.columns if c not in {"season", "school"}}}
    )
    panel = panel.merge(me, on=["season", "school"]).merge(you, on=["season", "opponent"])

    n = (panel["own_n"] - 1).clip(lower=1)
    panel["own_rush_avg"] = (panel["own_rush_yds"] - panel["rushing_yards"]) / n
    panel["own_pass_avg"] = (panel["own_pass_yds"] - panel["net_passing_yards"]) / n
    panel["own_rush_share_avg"] = (panel["own_rush_att"] - panel["rushing_attempts"]) / (
        (panel["own_plays"] - panel["plays"]).clip(lower=1)
    )
    panel["own_ypa"] = (panel["own_pass_yds"] - panel["net_passing_yards"]) / (
        (panel["own_pass_att"] - panel["pass_attempts"]).clip(lower=1)
    )
    panel["own_ypc"] = (panel["own_rush_yds"] - panel["rushing_yards"]) / (
        (panel["own_rush_att"] - panel["rushing_attempts"]).clip(lower=1)
    )

    dn = (panel["def_n"] - 1).clip(lower=1)
    panel["def_rush_allowed"] = (panel["def_allowed_rush_yds"] - panel["rushing_yards"]) / dn
    panel["def_pass_allowed"] = (panel["def_allowed_pass_yds"] - panel["net_passing_yards"]) / dn

    panel["rush_pct_of_avg"] = panel["rushing_yards"] / panel["own_rush_avg"].replace(0, np.nan)
    panel["pass_pct_of_avg"] = panel["net_passing_yards"] / panel["own_pass_avg"].replace(0, np.nan)
    panel["share_pct_of_avg"] = panel["rush_share"] / panel["own_rush_share_avg"].replace(0, np.nan)

    usable = panel.replace([np.inf, -np.inf], np.nan)
    rush_w, rush_mae = _best_opponent_weight(usable["own_rush_avg"], usable["def_rush_allowed"], usable["rushing_yards"])
    pass_w, pass_mae = _best_opponent_weight(
        usable["own_pass_avg"], usable["def_pass_allowed"], usable["net_passing_yards"]
    )
    share_w, share_mae = _best_opponent_weight(
        usable["own_rush_share_avg"],
        (usable["def_allowed_rush_att"] - usable["rushing_attempts"])
        / (usable["def_faced_plays"] - usable["plays"]).clip(lower=1),
        usable["rush_share"],
    )

    def _cv(s: pd.Series) -> float:
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        s = s[(s > 0) & (s < 5)]
        if s.empty or s.mean() == 0:
            return float("nan")
        return float(s.std() / s.mean())

    held_rush = float((usable["rush_pct_of_avg"] < 0.80).mean())
    held_pass = float((usable["pass_pct_of_avg"] < 0.80).mean())
    corr_rush = float(usable["rush_pct_of_avg"].clip(0, 3).corr(usable["def_rush_allowed"]))
    corr_pass = float(usable["pass_pct_of_avg"].clip(0, 3).corr(usable["def_pass_allowed"]))

    return {
        "n_team_games": int(usable["rushing_yards"].notna().sum()),
        "rush_pct_of_avg_cv": _cv(usable["rush_pct_of_avg"]),
        "pass_pct_of_avg_cv": _cv(usable["pass_pct_of_avg"]),
        "share_pct_of_avg_cv": _cv(usable["share_pct_of_avg"]),
        "pct_games_held_under_80pct_rush_avg": held_rush,
        "pct_games_held_under_80pct_pass_avg": held_pass,
        "corr_rush_pct_vs_def_allowed": corr_rush,
        "corr_pass_pct_vs_def_allowed": corr_pass,
        "rush_opponent_weight": rush_w,
        "pass_opponent_weight": pass_w,
        "rush_share_opponent_weight": share_w,
        "rush_blend_mae": rush_mae,
        "pass_blend_mae": pass_mae,
        "share_blend_mae": share_mae,
        "note": (
            "Higher opponent weight = production moves with the defense. "
            "We expect rush_opponent_weight >> pass_opponent_weight."
        ),
    }
