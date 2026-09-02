"""Quarterback and position-weighted player impact.

The starter at QB moves a card more than any other listing. A dual-threat /
dynamic QB also lets a roster punch above its talent and FPI — the run threat
is a second offense the box score of "team rushing" only partly captures.

Profiles follow the *person* across schools (portal + prior seasons), the same
way coach identity follows the coach. Season S never sees season S stats.

Who plays this week
    Expected starter = last year's pass leader, moved by the transfer portal,
    then injury status. Out/doubtful → blend toward the backup. Live unplayed
    games may use year-to-date attempts (completed games never do — that leaks).

Position weights
    QB 1.00 (× up to 1.45 if dual-threat). OT / EDGE next. Specialists last.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from cfb_model.config import (
    QB_DYNAMIC_INJURY_BONUS,
    QB_OUT_POINTS,
    QB_TENDENCY_K,
    QB_YEARS,
)
from cfb_model.util import shrink_series, zscore

# Injury / availability: one "full Out" at this position vs a replacement.
POSITION_WEIGHTS: dict[str, float] = {
    "QB": 1.00,
    "OT": 0.24,
    "LT": 0.26,
    "RT": 0.22,
    "T": 0.24,
    "OL": 0.20,
    "IOL": 0.18,
    "G": 0.18,
    "OG": 0.18,
    "C": 0.20,
    "OC": 0.20,
    "WR": 0.16,
    "WR1": 0.22,
    "RB": 0.15,
    "HB": 0.15,
    "TB": 0.15,
    "FB": 0.08,
    "TE": 0.12,
    "EDGE": 0.18,
    "DE": 0.16,
    "DL": 0.14,
    "DT": 0.13,
    "NT": 0.12,
    "LB": 0.10,
    "ILB": 0.10,
    "OLB": 0.12,
    "MLB": 0.10,
    "DB": 0.10,
    "CB": 0.11,
    "S": 0.10,
    "SAF": 0.10,
    "FS": 0.10,
    "SS": 0.10,
    "K": 0.07,
    "PK": 0.07,
    "P": 0.04,
    "LS": 0.02,
    "KR": 0.03,
    "PR": 0.03,
    "ATH": 0.14,
}
DEFAULT_POSITION_WEIGHT = 0.08

STATUS_WEIGHTS: dict[str, float] = {
    "out": 1.00,
    "inactive": 1.00,
    "suspended": 1.00,
    "doubtful": 0.75,
    "questionable": 0.35,
    "day-to-day": 0.30,
    "day to day": 0.30,
    "probable": 0.12,
    "active": 0.0,
    "available": 0.0,
}

QB_METRICS = [
    "pass_att",
    "cmp_rate",
    "pass_ypa",
    "td_rate",
    "int_rate",
    "rush_att",
    "rush_ypg",
    "rush_ypc",
    "qb_play_rush_share",
    "ppa_overall",
    "ppa_pass",
    "ppa_rush",
    "usage",
    "dynamic",
    "punch",
]

QB_VIEW = [
    ("pass_ypa", "Yards per attempt", "ypa", "dink and dunk", "downfield"),
    ("cmp_rate", "Completion rate", "pct", "volatile", "on schedule"),
    ("qb_play_rush_share", "QB run rate", "rush", "pocket statue", "dual-threat"),
    ("rush_ypg", "Rush yards / game", "ypg", "stays in the pocket", "second offense"),
    ("dynamic", "Dynamism", "dyn", "predictable dropback", "punches above weight"),
    ("punch", "Vs roster talent", "punch", "needs the roster", "elevates the roster"),
    ("ppa_overall", "PPA (prior)", "ppa", "replacement", "elite"),
    ("ppa_rush", "Rush PPA", "ppa", "statue", "designed + scramble"),
    ("int_rate", "INT rate", "pct", "careful", "live-wire"),
]

STARTER_PROB = {
    "out": 0.0,
    "inactive": 0.0,
    "suspended": 0.0,
    "doubtful": 0.22,
    "questionable": 0.58,
    "day-to-day": 0.62,
    "day to day": 0.62,
    "probable": 0.88,
}


def position_weight(position: str | None) -> float:
    if not position:
        return DEFAULT_POSITION_WEIGHT
    key = str(position).strip().upper()
    if key in POSITION_WEIGHTS:
        return POSITION_WEIGHTS[key]
    if "/" in key:
        return max(position_weight(part) for part in key.split("/"))
    return DEFAULT_POSITION_WEIGHT


def status_weight(status: str | None) -> float:
    if not status:
        return 0.0
    return STATUS_WEIGHTS.get(str(status).strip().lower(), 0.20)


def starter_probability(status: str | None) -> float:
    if not status:
        return 1.0
    return STARTER_PROB.get(str(status).strip().lower(), 1.0)


def is_quarterback(position: str | None, pass_att: float | None = None) -> bool:
    pos = str(position or "").upper()
    if "QB" in pos.split("/") or pos == "QB":
        return True
    if pass_att is not None and float(pass_att) >= 150 and pos in {"ATH", "WR", "RB", ""}:
        return True
    return False


def dynamic_score(
    rush_att: float | None,
    pass_att: float | None,
    rush_yards: float | None,
    games: float | None = None,
    ppa_rush: float | None = None,
) -> float:
    """0–1: how much this QB is a second run game, not just a passer."""
    rush = float(rush_att or 0.0)
    passing = float(pass_att or 0.0)
    plays = rush + passing
    share = rush / plays if plays > 0 else 0.0
    n = max(float(games or 0.0), 1.0)
    ypg = float(rush_yards or 0.0) / n
    ppa = float(ppa_rush or 0.0)
    score = (
        0.50 * min(1.0, share / 0.28)
        + 0.35 * min(1.0, max(0.0, ypg) / 70.0)
        + 0.15 * min(1.0, max(0.0, ppa) / 0.25)
    )
    return float(np.clip(score, 0.0, 1.0))


def classify_qb(
    dynamic: float | None,
    pass_ypa: float | None = None,
    int_rate: float | None = None,
    rush_ypg: float | None = None,
) -> str:
    dyn = 0.0 if dynamic is None or pd.isna(dynamic) else float(dynamic)
    ypa = 0.0 if pass_ypa is None or pd.isna(pass_ypa) else float(pass_ypa)
    ints = 0.0 if int_rate is None or pd.isna(int_rate) else float(int_rate)
    ypg = 0.0 if rush_ypg is None or pd.isna(rush_ypg) else float(rush_ypg)
    if dyn >= 0.55 or ypg >= 45:
        return "dual-threat"
    if dyn >= 0.32:
        return "mobile"
    if ypa >= 8.8 and ints >= 0.025:
        return "gunslinger"
    if ypa <= 7.0 and ints <= 0.022 and dyn < 0.2:
        return "game manager"
    return "pocket passer"


def qb_injury_load(dynamic: float | None, status: str | None) -> float:
    """Availability penalty in 'QB-out equivalents'. Dual-threat sits cost more."""
    sw = status_weight(status)
    if sw <= 0:
        return 0.0
    dyn = 0.0 if dynamic is None or pd.isna(dynamic) else float(dynamic)
    return sw * (1.0 + QB_DYNAMIC_INJURY_BONUS * dyn)


def qb_out_points(dynamic: float | None, status: str | None) -> float:
    return QB_OUT_POINTS * qb_injury_load(dynamic, status)


def player_injury_load(position: str | None, status: str | None, *, dynamic: float | None = None) -> float:
    base = position_weight(position) * status_weight(status)
    if is_quarterback(position) and dynamic:
        base *= 1.0 + QB_DYNAMIC_INJURY_BONUS * float(dynamic)
    return float(base)


def normalize_player_name(name: str | None) -> str:
    text = str(name or "").lower()
    text = text.replace(".", "").replace("'", "").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def player_key(player_id: Any, name: str | None) -> str:
    if player_id is not None and str(player_id).strip() not in {"", "None", "nan"}:
        return f"id:{player_id}"
    return f"name:{normalize_player_name(name)}"


def pivot_player_season(stats: pd.DataFrame) -> pd.DataFrame:
    """One row per player-season-team from category-long CFBD stats."""
    if stats is None or stats.empty:
        return pd.DataFrame()
    frame = stats.copy()
    if "player" not in frame.columns and "name" in frame.columns:
        frame = frame.rename(columns={"name": "player"})
    if "team" not in frame.columns and "school" in frame.columns:
        frame = frame.rename(columns={"school": "team"})
    if "season" not in frame.columns and "year" in frame.columns:
        frame = frame.rename(columns={"year": "season"})
    if "category" in frame.columns and frame["category"].notna().any():
        pieces = []
        for cat, part in frame.groupby(frame["category"].astype(str).str.lower()):
            rename = {}
            if cat in {"passing", "pass"}:
                rename = {
                    "attempts": "pass_att",
                    "completions": "pass_cmp",
                    "yards": "pass_yards",
                    "touchdowns": "pass_td",
                    "interceptions": "pass_int",
                    "yards_per_attempt": "pass_ypa",
                }
            elif cat in {"rushing", "rush"}:
                rename = {
                    "attempts": "rush_att",
                    "yards": "rush_yards",
                    "touchdowns": "rush_td",
                    "yards_per_carry": "rush_ypc",
                }
            part = part.rename(columns=rename)
            keep = [
                c
                for c in [
                    "season",
                    "player_id",
                    "player",
                    "position",
                    "team",
                    "conference",
                    "games",
                    "pass_att",
                    "pass_cmp",
                    "pass_yards",
                    "pass_td",
                    "pass_int",
                    "pass_ypa",
                    "rush_att",
                    "rush_yards",
                    "rush_td",
                    "rush_ypc",
                ]
                if c in part.columns
            ]
            pieces.append(part[keep])
        if not pieces:
            return pd.DataFrame()
        out = pieces[0]
        for extra in pieces[1:]:
            keys = [c for c in ("season", "player_id", "player", "team") if c in out.columns and c in extra.columns]
            extra_cols = [c for c in extra.columns if c not in keys]
            out = out.merge(extra[keys + extra_cols], on=keys, how="outer", suffixes=("", "_r"))
            if "games_r" in out.columns:
                out["games"] = pd.to_numeric(out.get("games"), errors="coerce").fillna(pd.to_numeric(out["games_r"], errors="coerce"))
                out = out.drop(columns=["games_r"])
        frame = out
    for col in ("pass_att", "pass_cmp", "pass_yards", "pass_td", "pass_int", "rush_att", "rush_yards", "rush_td", "games"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "pass_ypa" not in frame.columns or frame["pass_ypa"].isna().all():
        att = pd.to_numeric(frame.get("pass_att"), errors="coerce")
        yds = pd.to_numeric(frame.get("pass_yards"), errors="coerce")
        frame["pass_ypa"] = np.where(att > 0, yds / att, np.nan)
    if "rush_ypc" not in frame.columns or frame.get("rush_ypc", pd.Series(dtype=float)).isna().all():
        att = pd.to_numeric(frame.get("rush_att"), errors="coerce")
        yds = pd.to_numeric(frame.get("rush_yards"), errors="coerce")
        frame["rush_ypc"] = np.where(att > 0, yds / att, np.nan)
    cmp_ = pd.to_numeric(frame.get("pass_cmp"), errors="coerce")
    att = pd.to_numeric(frame.get("pass_att"), errors="coerce")
    frame["cmp_rate"] = np.where(att > 0, cmp_ / att, np.nan)
    td = pd.to_numeric(frame.get("pass_td"), errors="coerce")
    ints = pd.to_numeric(frame.get("pass_int"), errors="coerce")
    frame["td_rate"] = np.where(att > 0, td / att, np.nan)
    frame["int_rate"] = np.where(att > 0, ints / att, np.nan)
    games = pd.to_numeric(frame.get("games"), errors="coerce").fillna(0)
    rush_yds = pd.to_numeric(frame.get("rush_yards"), errors="coerce")
    frame["rush_ypg"] = np.where(games > 0, rush_yds / games, rush_yds)
    rush_att = pd.to_numeric(frame.get("rush_att"), errors="coerce").fillna(0)
    pass_att = pd.to_numeric(frame.get("pass_att"), errors="coerce").fillna(0)
    plays = rush_att + pass_att
    frame["qb_play_rush_share"] = np.where(plays > 0, rush_att / plays, np.nan)
    frame["player_key"] = [
        player_key(i, n) for i, n in zip(frame.get("player_id", pd.Series(index=frame.index)), frame.get("player"))
    ]
    return frame


def merge_player_ppa(players: pd.DataFrame, ppa: pd.DataFrame | None) -> pd.DataFrame:
    if players is None or players.empty:
        return pd.DataFrame() if players is None else players
    out = players.copy()
    for col in ("ppa_overall", "ppa_pass", "ppa_rush", "usage"):
        if col not in out.columns:
            out[col] = np.nan
    if ppa is None or ppa.empty:
        return out
    slim = ppa.copy()
    if "player" not in slim.columns and "name" in slim.columns:
        slim = slim.rename(columns={"name": "player"})
    if "season" not in slim.columns and "year" in slim.columns:
        slim = slim.rename(columns={"year": "season"})
    if "team" not in slim.columns and "school" in slim.columns:
        slim = slim.rename(columns={"school": "team"})
    rename = {
        "average_ppa": "ppa_overall",
        "overall": "ppa_overall",
        "passing": "ppa_pass",
        "rushing": "ppa_rush",
    }
    slim = slim.rename(columns={k: v for k, v in rename.items() if k in slim.columns})
    if "player_key" not in slim.columns:
        slim["player_key"] = [
            player_key(i, n) for i, n in zip(slim.get("player_id", pd.Series(index=slim.index)), slim.get("player"))
        ]
    keys = ["season", "player_key"]
    if "team" in slim.columns and "team" in out.columns:
        keys = ["season", "player_key", "team"]
    keep = keys + [c for c in ("ppa_overall", "ppa_pass", "ppa_rush", "usage") if c in slim.columns]
    slim = slim[keep].drop_duplicates(keys)
    out = out.merge(slim, on=keys, how="left", suffixes=("", "_ppa"))
    for col in ("ppa_overall", "ppa_pass", "ppa_rush", "usage"):
        alt = f"{col}_ppa"
        if alt in out.columns:
            out[col] = out[col].fillna(out[alt])
            out = out.drop(columns=[alt])
    return out


def add_dynamic_and_punch(players: pd.DataFrame, talent: pd.DataFrame | None = None) -> pd.DataFrame:
    out = players.copy()
    out["dynamic"] = [
        dynamic_score(r, p, y, g, pr)
        for r, p, y, g, pr in zip(
            out.get("rush_att", pd.Series(np.nan, index=out.index)),
            out.get("pass_att", pd.Series(np.nan, index=out.index)),
            out.get("rush_yards", pd.Series(np.nan, index=out.index)),
            out.get("games", pd.Series(np.nan, index=out.index)),
            out.get("ppa_rush", pd.Series(np.nan, index=out.index)),
        )
    ]
    out["qb_style"] = [
        classify_qb(d, y, i, r)
        for d, y, i, r in zip(out["dynamic"], out.get("pass_ypa"), out.get("int_rate"), out.get("rush_ypg"))
    ]
    ppa = pd.to_numeric(out.get("ppa_overall"), errors="coerce")
    if ppa.notna().sum() >= 8:
        out["ppa_z"] = out.groupby("season")["ppa_overall"].transform(zscore)
    else:
        ypa = pd.to_numeric(out.get("pass_ypa"), errors="coerce")
        out["ppa_z"] = zscore(ypa.fillna(ypa.median())) if ypa.notna().any() else 0.0
        if not isinstance(out["ppa_z"], pd.Series):
            out["ppa_z"] = pd.Series(out["ppa_z"], index=out.index)
    talent_z = pd.Series(0.0, index=out.index)
    if talent is not None and not talent.empty and "team" in out.columns:
        t = talent.rename(columns={"year": "season", "school": "team"})
        if "season" in t.columns and "team" in t.columns and "talent" in t.columns:
            tz = t[["season", "team", "talent"]].drop_duplicates(["season", "team"])
            tz["talent_z"] = tz.groupby("season")["talent"].transform(zscore)
            out = out.merge(tz[["season", "team", "talent_z"]], on=["season", "team"], how="left")
            talent_z = pd.to_numeric(out["talent_z"], errors="coerce").fillna(0.0)
    out["punch"] = pd.to_numeric(out["ppa_z"], errors="coerce").fillna(0.0) - talent_z
    return out


def team_pass_leaders(players: pd.DataFrame, *, ranks: int = 2) -> pd.DataFrame:
    """Rank 1/2 passers per team-season. Used only as *next season's* expected starter."""
    if players is None or players.empty:
        return pd.DataFrame()
    frame = players.copy()
    frame["pass_att"] = pd.to_numeric(frame.get("pass_att"), errors="coerce").fillna(0)
    qb_mask = [
        is_quarterback(p, a) or a >= 80
        for p, a in zip(frame.get("position", pd.Series(index=frame.index)), frame["pass_att"])
    ]
    frame = frame.loc[qb_mask].copy()
    if frame.empty:
        return frame
    frame = frame.sort_values(["season", "team", "pass_att"], ascending=[True, True, False])
    frame["qb_rank"] = frame.groupby(["season", "team"]).cumcount() + 1
    return frame[frame["qb_rank"] <= ranks].copy()


def apply_portal(leaders: pd.DataFrame, portal: pd.DataFrame | None) -> pd.DataFrame:
    """Move last year's passers to destination schools for season S.

    Portal year S is preseason — safe for season S games. Origin loses the QB;
    destination gains them as rank 1 if they out-passed the returning QB.
    """
    if leaders is None or leaders.empty:
        return pd.DataFrame() if leaders is None else leaders
    staying = leaders.copy()
    staying["origin_season"] = staying["season"]
    staying["season"] = staying["season"] + 1
    if portal is None or portal.empty:
        staying["via_portal"] = 0
        return staying
    port = portal.copy()
    if "season" not in port.columns and "year" in port.columns:
        port = port.rename(columns={"year": "season"})
    port["player_key"] = [
        player_key(i, n)
        for i, n in zip(
            port.get("player_id", pd.Series(index=port.index)),
            port.get("player", port.get("name")),
        )
    ]
    port["origin_key"] = port.get("origin", pd.Series(index=port.index)).map(
        lambda s: normalize_player_name(s) if pd.notna(s) else ""
    )
    port["dest_key"] = port.get("destination", pd.Series(index=port.index)).map(
        lambda s: normalize_player_name(s) if pd.notna(s) else ""
    )
    port["is_qb"] = [
        is_quarterback(p, None) or str(p or "").upper() in {"QB", "ATH"}
        for p in port.get("position", pd.Series(index=port.index))
    ]
    qb_port = port[port["is_qb"]].copy()
    staying["team_key"] = staying["team"].map(normalize_player_name)
    staying["via_portal"] = 0
    if qb_port.empty:
        return staying

    moved_rows = []
    leave_keys = set()
    for row in qb_port.itertuples(index=False):
        season = getattr(row, "season", None)
        pkey = getattr(row, "player_key", None)
        origin = getattr(row, "origin_key", "")
        dest = getattr(row, "dest_key", "")
        dest_school = getattr(row, "destination", None)
        if not season or not pkey:
            continue
        match = staying[(staying["season"] == season) & (staying["player_key"] == pkey)]
        if match.empty and origin:
            match = staying[(staying["season"] == season) & (staying["team_key"] == origin) & (staying["qb_rank"] == 1)]
            if len(match) > 1:
                match = match.head(1)
        if match.empty:
            continue
        src = match.iloc[0].to_dict()
        leave_keys.add((season, src.get("player_key"), src.get("team")))
        if dest_school and str(dest_school).strip() and str(dest_school).lower() not in {"none", "nan", ""}:
            src["team"] = dest_school
            src["via_portal"] = 1
            src["qb_rank"] = 1
            moved_rows.append(src)

    if leave_keys:
        staying = staying[
            ~staying.apply(lambda r: (r["season"], r["player_key"], r["team"]) in leave_keys, axis=1)
        ]
    if moved_rows:
        staying = pd.concat([staying, pd.DataFrame(moved_rows)], ignore_index=True)
    # After arrivals, re-rank by prior pass_att so a star transfer starts over a backup.
    staying = staying.sort_values(["season", "team", "pass_att"], ascending=[True, True, False])
    staying["qb_rank"] = staying.groupby(["season", "team"]).cumcount() + 1
    staying = staying[staying["qb_rank"] <= 2]
    return staying.drop(columns=["team_key"], errors="ignore")


def person_priors(players: pd.DataFrame) -> pd.DataFrame:
    """For season S, each player's identity is seasons < S only."""
    if players is None or players.empty:
        return pd.DataFrame()
    frame = players.sort_values(["player_key", "season"]).copy()
    n = frame.groupby("player_key")["pass_att"].transform(lambda s: s.shift(1).expanding().count())
    for col in QB_METRICS:
        if col not in frame.columns:
            frame[col] = np.nan
        raw = frame.groupby("player_key")[col].transform(
            lambda s, years=QB_YEARS: s.shift(1).rolling(years, min_periods=1).mean()
        )
        league = pd.to_numeric(frame[col], errors="coerce").median()
        prior = pd.Series(league, index=frame.index)
        frame[f"prior_{col}"] = shrink_series(raw, prior, n.fillna(0), QB_TENDENCY_K)
    frame["prior_seasons"] = n.fillna(0)
    return frame


def build_qb_season_table(
    player_stats: pd.DataFrame,
    player_ppa: pd.DataFrame | None = None,
    portal: pd.DataFrame | None = None,
    talent: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per team-season expected starter (and backup) with prior identity."""
    pivoted = pivot_player_season(player_stats)
    if pivoted.empty:
        return pd.DataFrame()
    pivoted = merge_player_ppa(pivoted, player_ppa)
    pivoted = add_dynamic_and_punch(pivoted, talent)
    priors = person_priors(pivoted)
    leaders = team_pass_leaders(pivoted)
    assigned = apply_portal(leaders, portal)
    if assigned.empty:
        return pd.DataFrame()
    prior_cols = ["player_key", "season"] + [c for c in priors.columns if c.startswith("prior_")]
    prior_cols += [c for c in ("qb_style", "player", "position") if c in priors.columns]
    # Person prior for season S lives on the S row of `priors` (already shifted).
    identity = priors[list(dict.fromkeys(prior_cols))].drop_duplicates(["player_key", "season"])
    out = assigned.merge(identity, on=["player_key", "season"], how="left", suffixes=("", "_id"))
    if "player_id" in out.columns and "player" not in out.columns:
        out["player"] = out.get("player_id")
    if "player_id" in identity.columns:
        pass
    # Prefer prior_* as the profile numbers used at kickoff.
    for col in QB_METRICS:
        prior_c = f"prior_{col}"
        if prior_c in out.columns:
            out[f"qb_{col}"] = pd.to_numeric(out[prior_c], errors="coerce")
        elif col in out.columns:
            out[f"qb_{col}"] = np.nan
    out["qb_name"] = out.get("player")
    out["qb_seasons_prior"] = pd.to_numeric(out.get("prior_seasons"), errors="coerce").fillna(0)
    if "qb_style" not in out.columns or out["qb_style"].isna().any():
        out["qb_style"] = [
            classify_qb(d, y, i, r)
            for d, y, i, r in zip(out.get("qb_dynamic"), out.get("qb_pass_ypa"), out.get("qb_int_rate"), out.get("qb_rush_ypg"))
        ]
    return out.reset_index(drop=True)


def _side_table(profiles: pd.DataFrame, rank: int, prefix: str, team_col: str) -> pd.DataFrame:
    part = profiles[profiles["qb_rank"] == rank].copy() if "qb_rank" in profiles.columns else profiles
    rename = {
        "team": team_col,
        "qb_name": f"{prefix}_qb",
        "player_key": f"{prefix}_qb_key",
        "qb_style": f"{prefix}_qb_style",
        "qb_seasons_prior": f"{prefix}_qb_seasons_prior",
        "via_portal": f"{prefix}_qb_portal",
    }
    for col in QB_METRICS:
        rename[f"qb_{col}"] = f"{prefix}_qb_{col}"
    part = part.rename(columns=rename)
    keep = list(dict.fromkeys(["season", team_col] + [c for c in rename.values() if c in part.columns]))
    return part[keep].drop_duplicates(["season", team_col], keep="first")


def attach_quarterbacks(
    frame: pd.DataFrame,
    profiles: pd.DataFrame,
    injuries: pd.DataFrame | None = None,
    live_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join expected starter (and backup) onto games. Injury-aware blend."""
    out = frame.copy()
    empty_cols = [
        "home_qb",
        "away_qb",
        "home_qb_style",
        "away_qb_style",
        "home_qb_backup",
        "away_qb_backup",
        "qb_ppa_diff",
        "qb_dynamic_diff",
        "qb_punch_diff",
        "qb_rush_share_diff",
        "qb_ypa_diff",
        "home_qb_starter_p",
        "away_qb_starter_p",
        "home_qb_out",
        "away_qb_out",
        "qb_out_points_diff",
    ]
    if profiles is None or profiles.empty:
        for col in empty_cols:
            out[col] = np.nan if col not in {"home_qb", "away_qb", "home_qb_style", "away_qb_style"} else None
        out["home_qb_starter_p"] = 1.0
        out["away_qb_starter_p"] = 1.0
        out["home_qb_out"] = 0.0
        out["away_qb_out"] = 0.0
        return out

    live = _live_starters(out, live_stats)
    use = profiles if live is None else _prefer_live(profiles, live)

    home_s = _side_table(use, 1, "home", "home_team")
    away_s = _side_table(use, 1, "away", "away_team")
    home_b = _side_table(use, 2, "home_backup", "home_team").rename(columns={"home_backup_qb": "home_qb_backup"})
    away_b = _side_table(use, 2, "away_backup", "away_team").rename(columns={"away_backup_qb": "away_qb_backup"})

    if "home_team" in out.columns:
        out = out.merge(home_s, on=["season", "home_team"], how="left")
        bkeep = ["season", "home_team"] + [c for c in home_b.columns if c not in {"season", "home_team"}]
        out = out.merge(home_b[bkeep], on=["season", "home_team"], how="left")
    if "away_team" in out.columns:
        out = out.merge(away_s, on=["season", "away_team"], how="left")
        bkeep = ["season", "away_team"] + [c for c in away_b.columns if c not in {"season", "away_team"}]
        out = out.merge(away_b[bkeep], on=["season", "away_team"], how="left")

    out = _apply_injury_starter_blend(out, injuries)
    out["qb_ppa_diff"] = pd.to_numeric(out.get("home_qb_ppa_overall"), errors="coerce") - pd.to_numeric(
        out.get("away_qb_ppa_overall"), errors="coerce"
    )
    out["qb_dynamic_diff"] = pd.to_numeric(out.get("home_qb_dynamic"), errors="coerce") - pd.to_numeric(
        out.get("away_qb_dynamic"), errors="coerce"
    )
    out["qb_punch_diff"] = pd.to_numeric(out.get("home_qb_punch"), errors="coerce") - pd.to_numeric(
        out.get("away_qb_punch"), errors="coerce"
    )
    out["qb_rush_share_diff"] = pd.to_numeric(out.get("home_qb_qb_play_rush_share"), errors="coerce") - pd.to_numeric(
        out.get("away_qb_qb_play_rush_share"), errors="coerce"
    )
    out["qb_ypa_diff"] = pd.to_numeric(out.get("home_qb_pass_ypa"), errors="coerce") - pd.to_numeric(
        out.get("away_qb_pass_ypa"), errors="coerce"
    )
    if "id" in out.columns and out["id"].duplicated().any():
        out = out.drop_duplicates("id", keep="first")
    return out


def _live_starters(frame: pd.DataFrame, live_stats: pd.DataFrame | None) -> pd.DataFrame | None:
    """YTD pass leaders for *unplayed* current-season games only."""
    if live_stats is None or live_stats.empty or frame.empty:
        return None
    if "completed" in frame.columns and frame["completed"].fillna(0).astype(int).eq(1).all():
        return None
    ytd = team_pass_leaders(pivot_player_season(live_stats))
    if ytd.empty:
        return None
    ytd = add_dynamic_and_punch(ytd, None)
    for col in QB_METRICS:
        src = col if col in ytd.columns else None
        ytd[f"qb_{col}"] = pd.to_numeric(ytd[src], errors="coerce") if src else np.nan
    ytd["qb_name"] = ytd.get("player")
    ytd["qb_seasons_prior"] = 0
    ytd["via_portal"] = 0
    if "qb_style" not in ytd.columns:
        ytd["qb_style"] = [
            classify_qb(d, y, i, r)
            for d, y, i, r in zip(ytd.get("dynamic"), ytd.get("pass_ypa"), ytd.get("int_rate"), ytd.get("rush_ypg"))
        ]
    return ytd


def _prefer_live(profiles: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """Replace expected starter with YTD leader for matching season/team when live exists."""
    keys = ["season", "team", "qb_rank"]
    live_keys = live.drop_duplicates(keys)
    # Keep prior identity numbers from profiles when the same player; else use YTD style only for name.
    merged = profiles.merge(
        live_keys[keys + [c for c in ("player_key", "player", "qb_name") if c in live_keys.columns]],
        on=keys,
        how="left",
        suffixes=("", "_live"),
    )
    if "player_key_live" in merged.columns:
        same = merged["player_key"] == merged["player_key_live"]
        swapped = merged["player_key_live"].notna() & ~same
        # If a different QB is clearly the YTD starter, swap the row from live.
        if swapped.any():
            live_full = live.rename(columns={"player": "qb_name"})
            replacement = live_full.merge(
                merged.loc[swapped, keys],
                on=keys,
                how="inner",
            )
            keep_cols = [c for c in profiles.columns if c in replacement.columns]
            profiles = pd.concat([profiles.loc[~swapped], replacement[keep_cols]], ignore_index=True)
    return profiles


def _apply_injury_starter_blend(frame: pd.DataFrame, injuries: pd.DataFrame | None) -> pd.DataFrame:
    out = frame.copy()
    out["home_qb_starter_p"] = 1.0
    out["away_qb_starter_p"] = 1.0
    out["home_qb_out"] = 0.0
    out["away_qb_out"] = 0.0
    out["home_qb_out_points"] = 0.0
    out["away_qb_out_points"] = 0.0
    if injuries is None or injuries.empty:
        out["qb_out_points_diff"] = 0.0
        return out

    def _status_for(team_col: str, qb_col: str) -> pd.Series:
        teams = out.get(team_col, pd.Series(index=out.index)).astype(str)
        names = out.get(qb_col, pd.Series(index=out.index)).map(normalize_player_name)
        inj = injuries.copy()
        inj["team_key"] = inj.get("team", pd.Series(index=inj.index)).map(lambda s: normalize_player_name(s) if pd.notna(s) else "")
        inj["player_key"] = inj.get("player", pd.Series(index=inj.index)).map(normalize_player_name)
        status = []
        for team, name in zip(teams, names):
            hit = inj[(inj["team_key"] == normalize_player_name(team)) & (inj["player_key"] == name)]
            if hit.empty and name:
                hit = inj[inj["player_key"] == name]
            if hit.empty:
                status.append(None)
            else:
                status.append(hit.iloc[0].get("status"))
        return pd.Series(status, index=out.index)

    for side in ("home", "away"):
        qb_col = f"{side}_qb"
        if qb_col not in out.columns:
            continue
        status = _status_for(f"{side}_team", qb_col)
        p_start = status.map(starter_probability).fillna(1.0)
        dyn = pd.to_numeric(out.get(f"{side}_qb_dynamic"), errors="coerce")
        out[f"{side}_qb_starter_p"] = p_start
        out[f"{side}_qb_out"] = 1.0 - p_start
        out[f"{side}_qb_out_points"] = [
            qb_out_points(d, s) for d, s in zip(dyn, status)
        ]
        # Blend starter quality toward backup when he might not play.
        p = p_start.clip(0, 1)
        for metric in ("ppa_overall", "dynamic", "punch", "pass_ypa", "qb_play_rush_share"):
            starter = pd.to_numeric(out.get(f"{side}_qb_{metric}"), errors="coerce")
            backup = pd.to_numeric(out.get(f"{side}_backup_qb_{metric}"), errors="coerce")
            if starter is None:
                continue
            blended = p * starter + (1.0 - p) * backup.fillna(starter)
            out[f"{side}_qb_{metric}"] = blended.where(backup.notna() | (p >= 0.99), starter)
    out["qb_out_points_diff"] = pd.to_numeric(out["home_qb_out_points"], errors="coerce").fillna(0) - pd.to_numeric(
        out["away_qb_out_points"], errors="coerce"
    ).fillna(0)
    return out


def qb_card(profiles: pd.DataFrame, query: str, season: int | None = None) -> dict[str, Any] | None:
    if profiles is None or profiles.empty:
        return None
    names = profiles.get("qb_name", profiles.get("player", pd.Series(dtype=str))).astype(str)
    hit = profiles[names.str.contains(query, case=False, na=False)]
    if hit.empty and "team" in profiles.columns:
        hit = profiles[profiles["team"].astype(str).str.contains(query, case=False, na=False)]
        if season is not None:
            hit = hit[hit["season"] == season]
        hit = hit[hit.get("qb_rank", 1) == 1] if "qb_rank" in hit.columns else hit
    if hit.empty:
        return None
    if season is not None:
        seasonal = hit[hit["season"] == season]
        if not seasonal.empty:
            hit = seasonal
    hit = hit.sort_values("season")
    row = hit.iloc[-1]
    attrs = []
    for col, label, _kind, lo, hi in QB_VIEW:
        key = f"qb_{col}"
        value = row[key] if key in row.index else row.get(col)
        attrs.append(
            {
                "key": col,
                "label": label,
                "value": None if value is None or (isinstance(value, float) and pd.isna(value)) else float(value)
                if _is_number(value)
                else value,
                "z": None,
                "low": lo,
                "high": hi,
            }
        )
    return {
        "kind": "qb",
        "season": int(row["season"]) if pd.notna(row.get("season")) else None,
        "school": row.get("team"),
        "qb_name": row.get("qb_name") or row.get("player"),
        "scheme": row.get("qb_style"),
        "mercy": "elevates roster" if pd.to_numeric(row.get("qb_punch"), errors="coerce") > 0.4 else "roster-dependent",
        "attributes": attrs,
        "stops": [
            {"season": int(r["season"]), "school": r.get("team"), "scheme": r.get("qb_style"), "mercy": ""}
            for _, r in hit.iterrows()
        ],
    }


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return value is not None and not (isinstance(value, float) and pd.isna(value))
    except (TypeError, ValueError):
        return False
