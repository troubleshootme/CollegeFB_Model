from __future__ import annotations

import pandas as pd

# A real week-0 slate is followed by a ~5 day gap; Sun->Tue (Labor Day) is only 2.
GAP_DAYS = 3


def _is_regular(series: pd.Series) -> pd.Series:
    text = series.fillna("regular").astype(str).str.lower()
    return text.eq("regular") | text.eq("nan") | text.eq("")


def assign_board_week(frame: pd.DataFrame) -> pd.DataFrame:
    """Map the early CFBD week-1 cluster to board week 0. Leave model `week` alone."""
    out = frame.copy()
    if out.empty or "week" not in out.columns:
        return out
    out["board_week"] = pd.to_numeric(out["week"], errors="coerce")
    if "start_date" not in out.columns or "season" not in out.columns:
        return out
    starts = pd.to_datetime(out["start_date"], utc=True, errors="coerce")
    days = starts.dt.normalize()
    regular = _is_regular(out["season_type"]) if "season_type" in out.columns else pd.Series(True, index=out.index)
    opening = out["board_week"].eq(1) & regular
    for _, idx in out.loc[opening].groupby("season").groups.items():
        season_days = sorted(days.loc[idx].dropna().unique())
        if len(season_days) < 2:
            continue
        cut = None
        for prev, nxt in zip(season_days, season_days[1:]):
            if (pd.Timestamp(nxt) - pd.Timestamp(prev)).days >= GAP_DAYS:
                cut = pd.Timestamp(prev)
                break
        if cut is None:
            continue
        week0 = [i for i in idx if pd.notna(days.loc[i]) and days.loc[i] <= cut]
        # The week-0 slate is the small early cluster; never relabel most of week 1.
        if len(week0) > 0.6 * len(idx):
            continue
        out.loc[week0, "board_week"] = 0
    # CFBD numbers postseason games from 1 again; keep them after the regular season.
    if "season_type" in out.columns:
        post = ~regular
        for season, idx in out.loc[post].groupby("season").groups.items():
            last_regular = out.loc[regular & out["season"].eq(season), "board_week"].max()
            if pd.notna(last_regular):
                out.loc[idx, "board_week"] = last_regular + out.loc[idx, "board_week"].fillna(1)
    return out


def slate_for_week(frame: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    col = "board_week" if "board_week" in frame.columns else "week"
    out = frame[
        (frame["season"] == int(season))
        & (pd.to_numeric(frame[col], errors="coerce") == int(week))
    ]
    if "fbs_vs_fbs" in out.columns:
        out = out[out["fbs_vs_fbs"].astype(bool)]
    return out.copy()
