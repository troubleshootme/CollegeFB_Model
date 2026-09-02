"""Leakage-safe coach and team identity profiles.

A profile follows the *person* across schools and the *program* across coaches.
It is scheme and behavior, not just 4th-down rate:

- run/pass identity (air raid vs ground-and-pound)
- tempo / game control
- explosiveness and success rate
- 4th-down aggression
- whether they pile on or take the air out when a game is in hand

Built from stored box scores, advanced stats, PPA, and quarter scores.
Never calls CFBD /plays. For season S every number uses seasons < S.
Current score and clock still override these priors in the simulator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cfb_model.config import (
    BLOWOUT_LEAD,
    COACH_AGGRESSION_K,
    COACH_TENDENCY_K,
    COACH_YEARS,
    TENDENCY_YEARS,
)
from cfb_model.util import shrink_series, zscore

# Columns written onto a coach-season (and mirrored onto team identity).
PROFILE_METRICS = [
    "rush_share",
    "plays_pg",
    "sec_per_play",
    "explosiveness",
    "success_rate",
    "pass_tilt",
    "fourth_att_pg",
    "run_up",
    "off_ppa",
    "def_ppa",
    "rush_quality",
    "grind",
    "explode_fast",
    "def_rush_allowed",
    "def_rush_vs_opp",
    "def_pass_vs_opp",
]

# (column, label, kind, low-anchor, high-anchor) for viewers.
PROFILE_VIEW = [
    ("rush_share", "Run/pass identity", "rush", "air raid", "ground-and-pound"),
    ("plays_pg", "Tempo", "plays", "grind", "hurry-up"),
    ("sec_per_play", "Seconds per play", "sec", "hurry-up", "game control"),
    ("explosiveness", "Explosiveness", "num", "methodical", "chunk plays"),
    ("success_rate", "Success rate", "pct", "boom-bust", "stay on schedule"),
    ("rush_quality", "Can they run it (vs competition)", "num", "stuffed", "imposes the run"),
    ("grind", "Clock / possession", "num", "gives it back", "sits on the ball"),
    ("explode_fast", "Score-too-fast risk", "num", "methodical TOP", "chunk TDs, short TOP"),
    ("def_rush_allowed", "Rush share faced (raw)", "pct", "pass-heavy looks", "run-heavy looks"),
    ("def_rush_vs_opp", "Rush allowed vs opponent average", "delta", "make you throw", "make you run"),
    ("def_pass_vs_opp", "Pass allowed vs opponent average", "delta", "make you run", "make you throw"),
    ("pass_tilt", "Pass PPA − rush PPA", "tilt", "run efficiency", "pass efficiency"),
    ("fourth_att_pg", "4th-down attempts / g", "num", "punt", "go for it"),
    ("run_up", "When the game is in hand", "mercy", "takes the air out", "piles on"),
    ("off_ppa", "Offensive PPA (prior)", "num", "struggling", "efficient"),
    ("def_ppa", "Defensive PPA (prior)", "num", "stout", "leaky"),
]

COACH_Z_FIELDS = [
    "rush_share",
    "plays_pg",
    "explosiveness",
    "fourth_att_pg",
    "run_up",
    "pass_tilt",
    "rush_quality",
    "grind",
    "explode_fast",
    "def_rush_vs_opp",
    "def_pass_vs_opp",
]


def classify_scheme(
    rush_share: float | None,
    plays_pg: float | None,
    sec_per_play: float | None = None,
    explosiveness: float | None = None,
) -> str:
    rush = 0.48 if rush_share is None or pd.isna(rush_share) else float(rush_share)
    fast = False
    slow = False
    if plays_pg is not None and pd.notna(plays_pg):
        fast = float(plays_pg) >= 70
        slow = float(plays_pg) <= 62
    elif sec_per_play is not None and pd.notna(sec_per_play):
        fast = float(sec_per_play) <= 24.5
        slow = float(sec_per_play) >= 30.0
    expl = None if explosiveness is None or pd.isna(explosiveness) else float(explosiveness)
    if rush <= 0.40 and (fast or (expl is not None and expl >= 1.25)):
        return "air raid"
    if rush <= 0.43:
        return "spread passing"
    if rush >= 0.60 and slow:
        return "ground and pound"
    if rush >= 0.56:
        return "run-first"
    if fast:
        return "hurry-up"
    return "balanced"


def classify_mercy(run_up: float | None) -> str:
    if run_up is None or pd.isna(run_up):
        return "standard"
    if float(run_up) >= 0.65:
        return "piles on"
    if float(run_up) <= -0.65:
        return "takes the air out"
    return "standard"


def build_coach_season_table(
    coaches: pd.DataFrame,
    games: pd.DataFrame,
    team_stats: pd.DataFrame,
    advanced: pd.DataFrame | None = None,
    ppa: pd.DataFrame | None = None,
    teams: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per head-coach season with *prior-season* identity."""
    empty = _empty_coach_frame()
    if coaches is None or coaches.empty or games is None or games.empty:
        return empty
    heads = _head_coaches(coaches, games, teams)
    if heads.empty:
        return empty
    style = _season_team_style(games, team_stats, advanced, ppa)
    if style.empty:
        joined = heads.copy()
        for col in PROFILE_METRICS:
            joined[col] = np.nan
    else:
        joined = heads.merge(style, on=["season", "school"], how="left")
        if "team_id" in joined.columns and "team_id_y" in joined.columns:
            joined["team_id"] = joined["team_id"].fillna(joined["team_id_y"])
            joined = joined.drop(columns=["team_id_y"], errors="ignore")
        elif "team_id_x" in joined.columns:
            joined["team_id"] = joined["team_id_x"].fillna(joined.get("team_id_y"))
            joined = joined.drop(columns=["team_id_x", "team_id_y"], errors="ignore")
    joined = joined.sort_values(["coach_key", "season"])
    prior_n = joined.groupby("coach_key")["rush_share"].transform(
        lambda s: s.shift(1).expanding().count()
    )
    # 4th-down sample can exist even when rush_share is missing
    fourth_n = joined.groupby("coach_key")["fourth_att_pg"].transform(
        lambda s: s.shift(1).expanding().count()
    )
    prior_n = prior_n.fillna(0).combine(fourth_n.fillna(0), max)
    for col in PROFILE_METRICS:
        if col not in joined.columns:
            joined[col] = np.nan
        raw = joined.groupby("coach_key")[col].transform(
            lambda s, years=COACH_YEARS: s.shift(1).rolling(years, min_periods=1).mean()
        )
        league = _league_prior(joined, col)
        k = COACH_AGGRESSION_K if col == "fourth_att_pg" else COACH_TENDENCY_K
        shrunk = shrink_series(raw, league, prior_n.fillna(0), k)
        school = _school_prior(joined, col)
        filled = shrunk.where(raw.notna(), school)
        joined[f"coach_{col}"] = filled.fillna(league)
    joined["coach_seasons_prior"] = prior_n.fillna(0)
    joined["coach_aggression"] = joined["coach_fourth_att_pg"]
    for col in COACH_Z_FIELDS:
        joined[f"coach_{col}_z"] = joined.groupby("season")[f"coach_{col}"].transform(zscore)
    joined["coach_aggression_z"] = joined["coach_fourth_att_pg_z"]
    joined["coach_tempo_z"] = joined["coach_plays_pg_z"]
    joined["coach_scheme"] = [
        classify_scheme(r, p, s, e)
        for r, p, s, e in zip(
            joined["coach_rush_share"],
            joined["coach_plays_pg"],
            joined["coach_sec_per_play"],
            joined["coach_explosiveness"],
        )
    ]
    joined["coach_mercy"] = joined["coach_run_up_z"].map(classify_mercy)
    keep = [
        "season",
        "team_id",
        "school",
        "coach_id",
        "coach_key",
        "coach_name",
        "coach_scheme",
        "coach_mercy",
        "coach_seasons_prior",
        "coach_aggression",
        "coach_aggression_z",
        "coach_tempo_z",
    ]
    keep += [f"coach_{c}" for c in PROFILE_METRICS]
    keep += [f"coach_{c}_z" for c in COACH_Z_FIELDS]
    keep = list(dict.fromkeys([c for c in keep if c in joined.columns]))
    out = joined[keep].drop_duplicates(["season", "school"], keep="first")
    return out.reset_index(drop=True)


def build_coach_profiles(*args: Any, **kwargs: Any) -> pd.DataFrame:
    return build_coach_season_table(*args, **kwargs)


def build_team_profiles(
    games: pd.DataFrame,
    team_stats: pd.DataFrame,
    advanced: pd.DataFrame | None = None,
    ppa: pd.DataFrame | None = None,
    coach_table: pd.DataFrame | None = None,
    teams: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Program identity for season S from seasons < S, with the HC overlaid.

    This is what a team *is* entering the year: three-year style, plus the
    coach who just took the job (whose career may disagree with the program).
    """
    style = _season_team_style(games, team_stats, advanced, ppa)
    if style.empty:
        return pd.DataFrame()
    style = style.sort_values(["school", "season"])
    prior_n = style.groupby("school")["rush_share"].transform(
        lambda s: s.shift(1).expanding().count()
    )
    for col in PROFILE_METRICS:
        if col not in style.columns:
            style[col] = np.nan
        raw = style.groupby("school")[col].transform(
            lambda s, years=TENDENCY_YEARS: s.shift(1).rolling(years, min_periods=1).mean()
        )
        league = _league_prior(style, col)
        k = COACH_TENDENCY_K
        style[f"team_{col}"] = shrink_series(raw, league, prior_n.fillna(0), k).fillna(league)
        style[f"team_{col}_z"] = style.groupby("season")[f"team_{col}"].transform(zscore)
    style["team_seasons_prior"] = prior_n.fillna(0)
    style["team_scheme"] = [
        classify_scheme(r, p, s, e)
        for r, p, s, e in zip(
            style["team_rush_share"],
            style["team_plays_pg"],
            style["team_sec_per_play"],
            style["team_explosiveness"],
        )
    ]
    style["team_mercy"] = style["team_run_up_z"].map(classify_mercy)
    if coach_table is not None and not coach_table.empty:
        ckeep = [c for c in coach_table.columns if c.startswith("coach_") or c in {"season", "school", "team_id"}]
        style = style.merge(coach_table[ckeep], on=["season", "school"], how="left", suffixes=("", "_c"))
        if "team_id" in style.columns and "team_id_c" in style.columns:
            style["team_id"] = style["team_id"].fillna(style["team_id_c"])
            style = style.drop(columns=["team_id_c"], errors="ignore")
    if teams is not None and not teams.empty and "school" in teams:
        meta = teams.rename(columns={"id": "team_id_meta", "school": "school"})
        keep_m = [c for c in ("school", "conference", "classification") if c in meta.columns]
        if keep_m:
            style = style.merge(meta[keep_m].drop_duplicates("school"), on="school", how="left")
    return style.reset_index(drop=True)


def attach_coaches(frame: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Left-join home/away coach identity onto a feature frame."""
    out = frame.copy()
    zeros = [
        "home_coach_aggression",
        "away_coach_aggression",
        "coach_aggression_diff",
        "home_coach_rush_share_z",
        "away_coach_rush_share_z",
        "coach_rush_share_diff",
        "home_coach_tempo_z",
        "away_coach_tempo_z",
        "coach_tempo_diff",
        "home_coach_expl_z",
        "away_coach_expl_z",
        "coach_expl_diff",
        "home_coach_run_up_z",
        "away_coach_run_up_z",
        "coach_run_up_diff",
        "home_coach_pass_tilt_z",
        "away_coach_pass_tilt_z",
        "coach_pass_tilt_diff",
        "home_coach_off_ppa",
        "away_coach_off_ppa",
        "coach_off_ppa_diff",
    ]
    if profiles is None or profiles.empty:
        for col in zeros:
            out[col] = 0.0
        out["home_coach"] = None
        out["away_coach"] = None
        out["home_coach_scheme"] = "balanced"
        out["away_coach_scheme"] = "balanced"
        out["home_coach_mercy"] = "standard"
        out["away_coach_mercy"] = "standard"
        return out

    rename_h = {
        "school": "home_team",
        "team_id": "home_id",
        "coach_name": "home_coach",
        "coach_scheme": "home_coach_scheme",
        "coach_mercy": "home_coach_mercy",
        "coach_rush_share": "home_coach_rush_share",
        "coach_rush_share_z": "home_coach_rush_share_z",
        "coach_plays_pg": "home_coach_plays_pg",
        "coach_tempo_z": "home_coach_tempo_z",
        "coach_sec_per_play": "home_coach_sec_per_play",
        "coach_explosiveness": "home_coach_explosiveness",
        "coach_expl_z": "home_coach_expl_z",
        "coach_success_rate": "home_coach_success_rate",
        "coach_pass_tilt": "home_coach_pass_tilt",
        "coach_pass_tilt_z": "home_coach_pass_tilt_z",
        "coach_fourth_att_pg": "home_coach_fourth_att_pg",
        "coach_aggression": "home_coach_aggression_raw",
        "coach_aggression_z": "home_coach_aggression",
        "coach_run_up": "home_coach_run_up",
        "coach_run_up_z": "home_coach_run_up_z",
        "coach_off_ppa": "home_coach_off_ppa",
        "coach_def_ppa": "home_coach_def_ppa",
        "coach_seasons_prior": "home_coach_seasons_prior",
    }
    if "coach_explosiveness_z" in profiles.columns:
        rename_h["coach_explosiveness_z"] = "home_coach_expl_z"
    rename_a = {k: v.replace("home_", "away_") for k, v in rename_h.items()}

    h = profiles.rename(columns=rename_h)
    a = profiles.rename(columns=rename_a)
    h_cols = ["season"] + [c for c in rename_h.values() if c in h.columns]
    a_cols = ["season"] + [c for c in rename_a.values() if c in a.columns]
    h = h[h_cols].drop_duplicates(["season", "home_team"] if "home_team" in h_cols else ["season"])
    a = a[a_cols].drop_duplicates(["season", "away_team"] if "away_team" in a_cols else ["season"])

    if "home_team" in h.columns and "home_team" in out.columns:
        out = out.merge(h, on=["season", "home_team"], how="left")
    elif "home_id" in h.columns and "home_id" in out.columns:
        out = out.merge(h, on=["season", "home_id"], how="left")
    if "away_team" in a.columns and "away_team" in out.columns:
        out = out.merge(a, on=["season", "away_team"], how="left")
    elif "away_id" in a.columns and "away_id" in out.columns:
        out = out.merge(a, on=["season", "away_id"], how="left")

    if "id" in out.columns and out["id"].duplicated().any():
        out = out.drop_duplicates("id", keep="first")

    out["home_coach_aggression"] = pd.to_numeric(out.get("home_coach_aggression"), errors="coerce").fillna(0.0)
    out["away_coach_aggression"] = pd.to_numeric(out.get("away_coach_aggression"), errors="coerce").fillna(0.0)
    out["coach_aggression_diff"] = out["home_coach_aggression"] - out["away_coach_aggression"]
    for stem, out_name in [
        ("coach_rush_share_z", "coach_rush_share_diff"),
        ("coach_tempo_z", "coach_tempo_diff"),
        ("coach_expl_z", "coach_expl_diff"),
        ("coach_run_up_z", "coach_run_up_diff"),
        ("coach_pass_tilt_z", "coach_pass_tilt_diff"),
        ("coach_off_ppa", "coach_off_ppa_diff"),
    ]:
        left = pd.to_numeric(out.get(f"home_{stem}"), errors="coerce").fillna(0.0)
        right = pd.to_numeric(out.get(f"away_{stem}"), errors="coerce").fillna(0.0)
        out[f"home_{stem}"] = left
        out[f"away_{stem}"] = right
        out[out_name] = left - right
    out["home_coach_scheme"] = out.get("home_coach_scheme", pd.Series("balanced", index=out.index)).fillna("balanced")
    out["away_coach_scheme"] = out.get("away_coach_scheme", pd.Series("balanced", index=out.index)).fillna("balanced")
    out["home_coach_mercy"] = out.get("home_coach_mercy", pd.Series("standard", index=out.index)).fillna("standard")
    out["away_coach_mercy"] = out.get("away_coach_mercy", pd.Series("standard", index=out.index)).fillna("standard")
    return out


def team_card(profiles: pd.DataFrame, school: str, season: int) -> dict[str, Any] | None:
    if profiles is None or profiles.empty:
        return None
    hit = profiles[(profiles["school"].astype(str).str.lower() == school.lower()) & (profiles["season"] == season)]
    if hit.empty:
        hit = profiles[profiles["school"].astype(str).str.contains(school, case=False, na=False) & (profiles["season"] == season)]
    if hit.empty:
        return None
    return _row_card(hit.iloc[0], kind="team")


def coach_card(profiles: pd.DataFrame, name: str, season: int | None = None) -> dict[str, Any] | None:
    if profiles is None or profiles.empty or "coach_name" not in profiles:
        return None
    names = profiles["coach_name"].astype(str)
    hit = profiles[names.str.contains(name, case=False, na=False)]
    if hit.empty:
        return None
    if season is not None:
        seasonal = hit[hit["season"] == season]
        if not seasonal.empty:
            hit = seasonal
    hit = hit.sort_values("season")
    row = hit.iloc[-1]
    card = _row_card(row, kind="coach")
    card["stops"] = [
        {
            "season": int(r["season"]),
            "school": r.get("school"),
            "scheme": r.get("coach_scheme"),
            "mercy": r.get("coach_mercy"),
        }
        for _, r in hit.iterrows()
    ]
    return card


def _row_card(row: pd.Series, *, kind: str) -> dict[str, Any]:
    prefix = "coach_" if kind == "coach" else "team_"
    attrs = []
    for col, label, _kind, lo, hi in PROFILE_VIEW:
        key = f"{prefix}{col}"
        zkey = f"{prefix}{col}_z"
        value = row[key] if key in row.index else None
        zval = row[zkey] if zkey in row.index else None
        attrs.append(
            {
                "key": col,
                "label": label,
                "value": None if value is None or (isinstance(value, float) and pd.isna(value)) else float(value)
                if _is_number(value)
                else value,
                "z": None if zval is None or (isinstance(zval, float) and pd.isna(zval)) else float(zval)
                if _is_number(zval)
                else zval,
                "low": lo,
                "high": hi,
            }
        )
    return {
        "kind": kind,
        "season": int(row["season"]) if pd.notna(row.get("season")) else None,
        "school": row.get("school"),
        "team_id": _maybe_int(row.get("team_id")),
        "coach_name": row.get("coach_name"),
        "scheme": row.get(f"{prefix}scheme") or row.get("coach_scheme") or row.get("team_scheme"),
        "mercy": row.get(f"{prefix}mercy") or row.get("coach_mercy") or row.get("team_mercy"),
        "seasons_prior": row.get(f"{prefix}seasons_prior"),
        "attributes": attrs,
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and pd.notna(value)


def _maybe_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _empty_coach_frame() -> pd.DataFrame:
    cols = [
        "season",
        "team_id",
        "school",
        "coach_id",
        "coach_key",
        "coach_name",
        "coach_scheme",
        "coach_mercy",
        "coach_seasons_prior",
        "coach_aggression",
        "coach_aggression_z",
        "coach_tempo_z",
    ]
    cols += [f"coach_{c}" for c in PROFILE_METRICS]
    cols += [f"coach_{c}_z" for c in COACH_Z_FIELDS]
    return pd.DataFrame(columns=list(dict.fromkeys(cols)))


def _head_coaches(
    coaches: pd.DataFrame,
    games: pd.DataFrame,
    teams: pd.DataFrame | None,
) -> pd.DataFrame:
    c = coaches.copy()
    if "year" in c.columns and ("season" not in c.columns or c["season"].isna().all()):
        c["season"] = c["year"]
    c["season"] = pd.to_numeric(c["season"], errors="coerce")
    c["school"] = c["school"].astype(str)
    first = c["first_name"].fillna("").astype(str) if "first_name" in c else ""
    last = c["last_name"].fillna("").astype(str) if "last_name" in c else c.get("coach_name", "")
    if "coach_name" in c and c["coach_name"].notna().any():
        c["coach_name"] = c["coach_name"].fillna((first + " " + last).str.strip())
    else:
        c["coach_name"] = (first + " " + last).str.strip()
    c.loc[c["coach_name"].eq(""), "coach_name"] = last if isinstance(last, pd.Series) else c["school"]
    if "coach_id" in c:
        cid = pd.to_numeric(c["coach_id"], errors="coerce")
        c["coach_key"] = np.where(cid.notna(), "id:" + cid.astype("Int64").astype(str), "name:" + c["coach_name"])
    else:
        c["coach_key"] = "name:" + c["coach_name"]
        c["coach_id"] = np.nan
    c["games"] = pd.to_numeric(c["games"], errors="coerce").fillna(0) if "games" in c else 0
    c = c.sort_values(["school", "season", "games", "coach_name"], ascending=[True, True, False, True])
    c = c.drop_duplicates(["school", "season"], keep="first")
    if "team_id" not in c.columns:
        c["team_id"] = np.nan
    c["team_id"] = pd.to_numeric(c["team_id"], errors="coerce")
    if teams is not None and not teams.empty and "school" in teams and "id" in teams:
        t = teams[["id", "school"]].rename(columns={"id": "tid"})
        c = c.merge(t.drop_duplicates("school"), on="school", how="left")
        c["team_id"] = c["team_id"].fillna(c["tid"])
        c = c.drop(columns=["tid"], errors="ignore")
    g = _games_norm(games)
    ids = pd.concat(
        [
            g[["season", "home_team", "home_id"]].rename(columns={"home_team": "school", "home_id": "gid"}),
            g[["season", "away_team", "away_id"]].rename(columns={"away_team": "school", "away_id": "gid"}),
        ],
        ignore_index=True,
    )
    ids["gid"] = pd.to_numeric(ids["gid"], errors="coerce")
    ids = ids.dropna(subset=["school", "gid"]).drop_duplicates(["season", "school"])
    c = c.merge(ids, on=["season", "school"], how="left")
    c["team_id"] = c["team_id"].fillna(c["gid"])
    c = c.drop(columns=["gid"], errors="ignore")
    return c.dropna(subset=["season", "school", "coach_name"])


def _games_norm(games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    if "game_id" not in g.columns and "id" in g.columns:
        g = g.rename(columns={"id": "game_id"})
    need = [
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "home_id",
        "away_id",
        "home_points",
        "away_points",
        "home_q1",
        "home_q2",
        "home_q3",
        "home_q4",
        "away_q1",
        "away_q2",
        "away_q3",
        "away_q4",
    ]
    for col in need:
        if col not in g.columns:
            g[col] = np.nan
    return g


def _season_team_style(
    games: pd.DataFrame,
    team_stats: pd.DataFrame,
    advanced: pd.DataFrame | None,
    ppa: pd.DataFrame | None,
) -> pd.DataFrame:
    g = _games_norm(games)
    long = _team_game_long(g, team_stats)
    if long.empty:
        return pd.DataFrame()
    long["plays"] = long["rushing_attempts"].fillna(0) + long["pass_attempts"].fillna(0)
    long["rush_share"] = np.where(long["plays"] > 0, long["rushing_attempts"] / long["plays"], np.nan)
    long["pass_share"] = np.where(long["plays"] > 0, long["pass_attempts"] / long["plays"], np.nan)
    long["sec_per_play"] = np.where(long["plays"] > 0, long["possession_seconds"] / long["plays"], np.nan)
    long["margin"] = long["points_for"] - long["points_against"]
    long["blowout_win"] = long["margin"] >= BLOWOUT_LEAD
    long["close_game"] = long["margin"].abs() <= 10
    long["lead_after3"] = (
        long["q1"].fillna(0) + long["q2"].fillna(0) + long["q3"].fillna(0)
    ) - (long["opp_q1"].fillna(0) + long["opp_q2"].fillna(0) + long["opp_q3"].fillna(0))
    has_q = long[["q1", "q2", "q3", "q4"]].notna().any(axis=1)
    long["in_hand_q4"] = has_q & (long["lead_after3"] >= BLOWOUT_LEAD)
    long["q4_in_hand"] = np.where(long["in_hand_q4"], long["q4"], np.nan)

    grouped = long.groupby(["season", "school"], as_index=False)
    style = grouped.agg(
        team_id=("team_id", "first"),
        games=("game_id", "nunique"),
        rush_share=("rush_share", "mean"),
        plays_pg=("plays", "mean"),
        sec_per_play=("sec_per_play", "mean"),
        possession_pg=("possession_seconds", "mean"),
        fourth_att_pg=("fourth_down_att", "mean"),
        off_ppa=("off_ppa", "mean"),
        def_ppa=("def_ppa", "mean"),
        pass_ppa=("pass_ppa", "mean"),
        rush_ppa=("rush_ppa", "mean"),
        explosiveness=("explosiveness", "mean"),
        success_rate=("success_rate", "mean"),
        q4_in_hand=("q4_in_hand", "mean"),
        n_in_hand=("in_hand_q4", "sum"),
    )
    blow = long.loc[long["blowout_win"]].groupby(["season", "school"], as_index=False).agg(
        pass_blowout=("pass_share", "mean"),
        fourth_blowout=("fourth_down_att", "mean"),
        points_blowout=("points_for", "mean"),
    )
    close = long.loc[long["close_game"]].groupby(["season", "school"], as_index=False).agg(
        pass_close=("pass_share", "mean"),
        fourth_close=("fourth_down_att", "mean"),
    )
    style = style.merge(blow, on=["season", "school"], how="left").merge(close, on=["season", "school"], how="left")
    style["pass_tilt"] = style["pass_ppa"] - style["rush_ppa"]
    style["pass_delta"] = style["pass_blowout"] - style["pass_close"]
    style["run_up"] = _run_up_score(style)

    if advanced is not None and not advanced.empty:
        adv = _season_mean(
            advanced,
            g,
            {
                "off_explosiveness": "explosiveness",
                "off_success_rate": "success_rate",
                "off_plays": "adv_plays",
            },
        )
        style = style.merge(adv, on=["season", "school"], how="left", suffixes=("", "_adv"))
        style["explosiveness"] = style["explosiveness"].fillna(style.get("explosiveness_adv"))
        style["success_rate"] = style["success_rate"].fillna(style.get("success_rate_adv"))
        if "adv_plays" in style:
            style["plays_pg"] = style["plays_pg"].fillna(style["adv_plays"])
    if ppa is not None and not ppa.empty:
        pstats = _season_mean(
            ppa,
            g,
            {
                "off_overall": "off_ppa",
                "def_overall": "def_ppa",
                "off_passing": "pass_ppa",
                "off_rushing": "rush_ppa",
            },
        )
        style = style.merge(pstats, on=["season", "school"], how="left", suffixes=("", "_ppa"))
        for col in ("off_ppa", "def_ppa", "pass_ppa", "rush_ppa"):
            alt = f"{col}_ppa"
            if alt in style.columns:
                style[col] = style[col].fillna(style[alt])
        style["pass_tilt"] = style["pass_ppa"] - style["rush_ppa"]
        style["run_up"] = _run_up_score(style)
    return style


def _run_up_score(style: pd.DataFrame) -> pd.Series:
    """Unitless composite: + piles on, − takes the air out."""
    parts = []
    weights = []
    q4 = pd.to_numeric(style.get("q4_in_hand"), errors="coerce")
    if q4 is not None:
        parts.append((q4 - 7.0) / 7.0)
        weights.append(1.2)
    delta = pd.to_numeric(style.get("pass_delta"), errors="coerce")
    if delta is not None:
        parts.append(delta / 0.12)
        weights.append(1.0)
    fourth_b = pd.to_numeric(style.get("fourth_blowout"), errors="coerce")
    if fourth_b is not None:
        parts.append((fourth_b - 0.7) / 0.6)
        weights.append(0.6)
    if not parts:
        return pd.Series(np.nan, index=style.index)
    stack = pd.concat(parts, axis=1)
    w = np.array(weights, dtype=float)
    weighted = stack.mul(w, axis=1).sum(axis=1, min_count=1)
    denom = stack.notna().mul(w, axis=1).sum(axis=1)
    return weighted / denom.replace(0, np.nan)


def _season_mean(table: pd.DataFrame, games: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    t = table.copy()
    if "season" not in t.columns and "game_id" in t.columns:
        t = t.merge(games[["game_id", "season"]], on="game_id", how="left")
    team_col = "team" if "team" in t.columns else "school"
    t = t.rename(columns={team_col: "school", **{k: v for k, v in mapping.items() if k in t.columns}})
    cols = ["season", "school"] + [v for v in mapping.values() if v in t.columns]
    if t.empty or "season" not in t.columns:
        return pd.DataFrame(columns=["season", "school"])
    return t[cols].groupby(["season", "school"], as_index=False).mean(numeric_only=True)


def _team_game_long(games: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "season": games["season"],
            "school": games["home_team"],
            "team_id": games["home_id"],
            "points_for": games["home_points"],
            "points_against": games["away_points"],
            "q1": games["home_q1"],
            "q2": games["home_q2"],
            "q3": games["home_q3"],
            "q4": games["home_q4"],
            "opp_q1": games["away_q1"],
            "opp_q2": games["away_q2"],
            "opp_q3": games["away_q3"],
            "opp_q4": games["away_q4"],
        }
    )
    away = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "season": games["season"],
            "school": games["away_team"],
            "team_id": games["away_id"],
            "points_for": games["away_points"],
            "points_against": games["home_points"],
            "q1": games["away_q1"],
            "q2": games["away_q2"],
            "q3": games["away_q3"],
            "q4": games["away_q4"],
            "opp_q1": games["home_q1"],
            "opp_q2": games["home_q2"],
            "opp_q3": games["home_q3"],
            "opp_q4": games["home_q4"],
        }
    )
    long = pd.concat([home, away], ignore_index=True)
    long["rushing_attempts"] = np.nan
    long["pass_attempts"] = np.nan
    long["possession_seconds"] = np.nan
    long["fourth_down_att"] = np.nan
    long["off_ppa"] = np.nan
    long["def_ppa"] = np.nan
    long["pass_ppa"] = np.nan
    long["rush_ppa"] = np.nan
    long["explosiveness"] = np.nan
    long["success_rate"] = np.nan
    if team_stats is not None and not team_stats.empty:
        s = team_stats.copy()
        s = s.rename(columns={"team": "school"})
        keep = [
            c
            for c in (
                "game_id",
                "school",
                "team_id",
                "rushing_attempts",
                "pass_attempts",
                "possession_seconds",
                "fourth_down_att",
            )
            if c in s.columns
        ]
        if "game_id" in keep and "school" in keep:
            slim = s[keep].drop_duplicates(["game_id", "school"])
            long = long.drop(
                columns=["rushing_attempts", "pass_attempts", "possession_seconds", "fourth_down_att", "team_id"],
                errors="ignore",
            )
            long = long.merge(slim, on=["game_id", "school"], how="left")
    return long


def _league_prior(frame: pd.DataFrame, col: str) -> pd.Series:
    means = frame.groupby("season")[col].mean()
    overall = frame[col].mean()
    mapped = frame["season"].map(lambda year: means.get(year - 1, overall))
    return mapped.fillna(overall)


def _school_prior(frame: pd.DataFrame, col: str) -> pd.Series:
    tmp = frame.sort_values(["school", "season"])
    prior = tmp.groupby("school")[col].transform(
        lambda s: s.shift(1).rolling(TENDENCY_YEARS, min_periods=1).mean()
    )
    return prior.reindex(frame.index)
