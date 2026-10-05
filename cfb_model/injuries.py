"""As-of injury snapshots from ESPN (no CFBD quota, never /plays).

CFBD has no injury feed. ESPN's public report is current-only, so we store
dated snapshots and join the latest snapshot *before kickoff*. Stale rows
(report date before the season) are dropped — ESPN sometimes leaves 2020
entries hanging around in September.

Historical games without a pre-kickoff snapshot stay NaN; the model imputes
and the explicit QB-out prior still moves live cards. Each weekly ingest
accumulates training data so the closed loop can learn the real weights.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from cfb_model import config, store
from cfb_model.players import (
    normalize_player_name,
    player_injury_load,
    position_weight,
    status_weight,
)
from cfb_model.util import getv

ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/injuries"
MASCOT_ALIASES = {
    "miami hurricanes": "Miami",
    "miami (oh) redhawks": "Miami (OH)",
    "miami redhawks": "Miami (OH)",
    "nc state wolfpack": "NC State",
    "north carolina state wolfpack": "NC State",
    "ole miss rebels": "Ole Miss",
    "lsu tigers": "LSU",
    "ucla bruins": "UCLA",
    "usc trojans": "USC",
    "tcu horned frogs": "TCU",
    "smu mustangs": "SMU",
    "ucf knights": "UCF",
    "usf bulls": "South Florida",
    "south florida bulls": "South Florida",
    "app state mountaineers": "App State",
    "appalachian state mountaineers": "App State",
    "louisiana ragin cajuns": "Louisiana",
    "louisiana ragin' cajuns": "Louisiana",
    "utsa roadrunners": "UTSA",
    "utep miners": "UTEP",
    "texas am aggies": "Texas A&M",
    "texas a&m aggies": "Texas A&M",
    "mississippi state bulldogs": "Mississippi State",
    "boston college eagles": "Boston College",
    "florida atlantic owls": "Florida Atlantic",
    "florida international panthers": "Florida International",
    "georgia tech yellow jackets": "Georgia Tech",
    "virginia tech hokies": "Virginia Tech",
    "pitt panthers": "Pittsburgh",
    "pittsburgh panthers": "Pittsburgh",
    "byu cougars": "BYU",
    "army black knights": "Army",
    "navy midshipmen": "Navy",
    "air force falcons": "Air Force",
    "colorado state rams": "Colorado State",
    "oklahoma state cowboys": "Oklahoma State",
    "michigan state spartans": "Michigan State",
    "penn state nittany lions": "Penn State",
    "ohio state buckeyes": "Ohio State",
    "florida state seminoles": "Florida State",
    "oregon state beavers": "Oregon State",
    "washington state cougars": "Washington State",
    "arizona state sun devils": "Arizona State",
    "kansas state wildcats": "Kansas State",
    "iowa state cyclones": "Iowa State",
    "north carolina tar heels": "North Carolina",
    "west virginia mountaineers": "West Virginia",
    "southern miss golden eagles": "Southern Mississippi",
    "middle tennessee blue raiders": "Middle Tennessee",
    "western kentucky hilltoppers": "Western Kentucky",
    "eastern michigan eagles": "Eastern Michigan",
    "western michigan broncos": "Western Michigan",
    "central michigan chippewas": "Central Michigan",
    "northern illinois huskies": "Northern Illinois",
    "bowling green falcons": "Bowling Green",
    "miami (fl) hurricanes": "Miami",
}


def flatten_espn_injuries(
    payload: Any,
    *,
    as_of: str | None = None,
    season: int | None = None,
    schools: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Parse ESPN site-v2 injury JSON into warehouse rows."""
    if payload is None:
        return []
    blob = payload if isinstance(payload, dict) else {"injuries": payload}
    season = season or _season_from_payload(blob)
    as_of = as_of or blob.get("timestamp") or datetime.now(timezone.utc).isoformat()
    snapshot_id = f"espn_{as_of[:19].replace(':', '')}"
    season_start = datetime(int(season or config.CURRENT_SEASON), 8, 1, tzinfo=timezone.utc)
    rows: list[dict[str, Any]] = []
    for team in blob.get("injuries") or []:
        raw_team = getv(team, "displayName") or getv(team, "name") or ""
        espn_id = str(getv(team, "id") or "")
        school = map_espn_team(raw_team, schools)
        for inj in getv(team, "injuries") or []:
            athlete = getv(inj, "athlete") or {}
            pos = _position(athlete)
            status = getv(inj, "status") or getv(getv(inj, "type") or {}, "description")
            report_date = getv(inj, "date")
            if _stale(report_date, season_start, as_of):
                continue
            sw = status_weight(status)
            if sw <= 0:
                continue
            player = getv(athlete, "displayName") or getv(athlete, "shortName")
            player_id = _athlete_id(athlete)
            dyn = None
            load = player_injury_load(pos, status, dynamic=dyn)
            rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "as_of": as_of,
                    "season": season,
                    "week": None,
                    "espn_team_id": espn_id,
                    "team": school,
                    "team_raw": raw_team,
                    "player": player,
                    "player_id": player_id,
                    "position": pos,
                    "status": status,
                    "status_weight": sw,
                    "position_weight": position_weight(pos),
                    "load": load,
                    "report_date": report_date,
                    "comment": getv(inj, "shortComment"),
                    "source": "espn",
                }
            )
    return rows


def map_espn_team(display_name: str | None, schools: list[str] | None = None) -> str | None:
    if not display_name:
        return None
    key = normalize_player_name(display_name)
    if key in MASCOT_ALIASES:
        return MASCOT_ALIASES[key]
    schools = list(schools or [])
    schools_sorted = sorted(schools, key=lambda s: len(str(s)), reverse=True)
    for school in schools_sorted:
        sk = normalize_player_name(school)
        if key == sk or key.startswith(sk + " "):
            return school
    return display_name


def attach_injuries(frame: pd.DataFrame, reports: pd.DataFrame) -> pd.DataFrame:
    """Latest snapshot strictly before kickoff, per team. Never uses post-game news."""
    out = frame.copy()
    for col in (
        "home_injury_load",
        "away_injury_load",
        "injury_load_diff",
        "home_qb_injury",
        "away_qb_injury",
        "skill_injury_diff",
        "ol_injury_diff",
        "def_injury_diff",
        "home_injury_note",
        "away_injury_note",
    ):
        out[col] = np.nan if not col.endswith("_note") else None
    if reports is None or reports.empty or out.empty:
        out["injury_load_diff"] = np.nan
        return out

    agg = _team_snapshot_loads(reports)
    if agg.empty:
        return out
    agg["as_of_ts"] = pd.to_datetime(agg["as_of"], utc=True, errors="coerce")
    starts = pd.to_datetime(out.get("start_date"), utc=True, errors="coerce")
    home_load, away_load = [], []
    home_note, away_note = [], []
    home_qb, away_qb = [], []
    skill_h, skill_a, ol_h, ol_a, def_h, def_a = [], [], [], [], [], []

    for i, row in out.iterrows():
        kick = starts.loc[i] if starts is not None else pd.NaT
        h = _pick_snapshot(agg, row.get("home_team"), kick)
        a = _pick_snapshot(agg, row.get("away_team"), kick)
        home_load.append(None if h is None else h.get("injury_load"))
        away_load.append(None if a is None else a.get("injury_load"))
        home_note.append(None if h is None else h.get("note"))
        away_note.append(None if a is None else a.get("note"))
        home_qb.append(0.0 if h is None else h.get("qb_load") or 0.0)
        away_qb.append(0.0 if a is None else a.get("qb_load") or 0.0)
        skill_h.append(0.0 if h is None else h.get("skill_load") or 0.0)
        skill_a.append(0.0 if a is None else a.get("skill_load") or 0.0)
        ol_h.append(0.0 if h is None else h.get("ol_load") or 0.0)
        ol_a.append(0.0 if a is None else a.get("ol_load") or 0.0)
        def_h.append(0.0 if h is None else h.get("def_load") or 0.0)
        def_a.append(0.0 if a is None else a.get("def_load") or 0.0)

    out["home_injury_load"] = pd.to_numeric(pd.Series(home_load, index=out.index), errors="coerce")
    out["away_injury_load"] = pd.to_numeric(pd.Series(away_load, index=out.index), errors="coerce")
    out["injury_load_diff"] = out["home_injury_load"] - out["away_injury_load"]
    out["home_qb_injury"] = pd.to_numeric(pd.Series(home_qb, index=out.index), errors="coerce")
    out["away_qb_injury"] = pd.to_numeric(pd.Series(away_qb, index=out.index), errors="coerce")
    out["qb_injury_diff"] = out["home_qb_injury"] - out["away_qb_injury"]
    out["skill_injury_diff"] = pd.Series(skill_h, index=out.index) - pd.Series(skill_a, index=out.index)
    out["ol_injury_diff"] = pd.Series(ol_h, index=out.index) - pd.Series(ol_a, index=out.index)
    out["def_injury_diff"] = pd.Series(def_h, index=out.index) - pd.Series(def_a, index=out.index)
    out["home_injury_note"] = home_note
    out["away_injury_note"] = away_note
    return out


def injury_prior_margin(frame: pd.DataFrame, *, scale: float = 1.0) -> pd.Series:
    """Points of home margin: injured home QBs (and others) pull this down.

    Scale is learned (EWMA) so if we overreact to QB outs the next cards shrink.
    """
    qb = pd.to_numeric(frame["qb_out_points_diff"], errors="coerce") if "qb_out_points_diff" in frame.columns else pd.Series(np.nan, index=frame.index)
    if qb.isna().all():
        home_q = pd.to_numeric(frame["home_qb_injury"], errors="coerce") if "home_qb_injury" in frame.columns else 0.0
        away_q = pd.to_numeric(frame["away_qb_injury"], errors="coerce") if "away_qb_injury" in frame.columns else 0.0
        qb = config.QB_OUT_POINTS * (pd.Series(home_q, index=frame.index).fillna(0) - pd.Series(away_q, index=frame.index).fillna(0))
    rest = pd.Series(0.0, index=frame.index)
    if "injury_load_diff" in frame.columns:
        rest = pd.to_numeric(frame["injury_load_diff"], errors="coerce").fillna(0)
        if "qb_injury_diff" in frame.columns:
            rest = rest - pd.to_numeric(frame["qb_injury_diff"], errors="coerce").fillna(0)
    rest_points = 6.5 * rest
    return -float(scale) * (qb.fillna(0) + rest_points)


def ingest_injuries(
    conn=None,
    *,
    db_path: Path | None = None,
    timeout: int = 25,
    payload: Any | None = None,
) -> int:
    own = conn is None
    conn = conn or store.init_schema(store.connect(db_path) if db_path else None)
    try:
        if payload is None:
            payload = _fetch_espn(timeout=timeout)
        teams = store.read_table(conn, "teams")
        schools = teams["school"].dropna().astype(str).tolist() if not teams.empty and "school" in teams else []
        games = store.read_table(conn, "games")
        season = config.CURRENT_SEASON
        week = None
        if not games.empty:
            live = games[games["season"] == season]
            if not live.empty:
                unfinished = live[live["completed"].fillna(0).astype(int).eq(0)]
                week_src = unfinished if not unfinished.empty else live
                week = int(pd.to_numeric(week_src["week"], errors="coerce").dropna().min())
        rows = flatten_espn_injuries(payload, season=season, schools=schools)
        if week is not None:
            for row in rows:
                row["week"] = week
        if not rows:
            store.set_meta(conn, "last_injury_as_of", datetime.now(timezone.utc).isoformat())
            store.set_meta(conn, "last_injury_n", "0")
            return 0
        frame = pd.DataFrame(rows)
        n = store.replace_rows(conn, "injury_reports", frame)
        store.set_meta(conn, "last_injury_as_of", str(frame["as_of"].iloc[0]))
        store.set_meta(conn, "last_injury_n", str(n))
        return n
    finally:
        if own:
            conn.close()


def _fetch_espn(timeout: int = 25) -> dict:
    response = requests.get(
        ESPN_INJURIES_URL,
        timeout=timeout,
        headers={"User-Agent": "cfb-model/2.0 (research; leakage-safe snapshots)"},
    )
    response.raise_for_status()
    return response.json()


def _season_from_payload(blob: dict) -> int:
    season = blob.get("season") or {}
    if isinstance(season, dict) and season.get("year"):
        return int(season["year"])
    return config.CURRENT_SEASON


def _stale(report_date: Any, season_start: datetime, as_of: str | None) -> bool:
    if not report_date:
        return False
    text = str(report_date).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if parsed < season_start:
        return True
    if as_of:
        try:
            snap = datetime.fromisoformat(str(as_of).replace("Z", "+00:00"))
            if snap.tzinfo is None:
                snap = snap.replace(tzinfo=timezone.utc)
            if parsed > snap:
                return True
        except ValueError:
            pass
    return False


def _position(athlete: Any) -> str | None:
    pos = getv(athlete, "position")
    if isinstance(pos, dict):
        return getv(pos, "abbreviation") or getv(pos, "displayName")
    return pos


def _athlete_id(athlete: Any) -> str:
    ident = getv(athlete, "id")
    if ident:
        return str(ident)
    for link in getv(athlete, "links") or []:
        href = getv(link, "href") or ""
        match = re.search(r"/id/(\d+)", href)
        if match:
            return match.group(1)
    return normalize_player_name(getv(athlete, "displayName")) or "unknown"


def _team_snapshot_loads(reports: pd.DataFrame) -> pd.DataFrame:
    frame = reports.copy()
    frame["load"] = pd.to_numeric(frame.get("load"), errors="coerce").fillna(0)
    frame["position"] = frame.get("position", pd.Series(index=frame.index)).astype(str)
    frame["is_qb"] = frame["position"].str.upper().str.contains("QB", na=False)
    frame["is_skill"] = frame["position"].str.upper().isin({"WR", "RB", "TE", "HB", "ATH"})
    frame["is_ol"] = frame["position"].str.upper().isin({"OL", "OT", "OG", "C", "T", "G", "LT", "RT", "IOL", "OC"})
    frame["is_def"] = frame["position"].str.upper().isin(
        {"DL", "DE", "DT", "NT", "EDGE", "LB", "ILB", "OLB", "MLB", "DB", "CB", "S", "SAF", "FS", "SS"}
    )
    grouped = frame.groupby(["snapshot_id", "as_of", "season", "team"], dropna=False)
    agg = grouped.agg(
        injury_load=("load", "sum"),
        qb_load=("load", lambda s: s[frame.loc[s.index, "is_qb"]].sum() if len(s) else 0),
        n_out=("status", lambda s: int(s.astype(str).str.lower().isin(["out", "inactive"]).sum())),
    ).reset_index()
    # pandas named-agg with lambda on other columns is fragile; compute group slices.
    rows = []
    for (snapshot_id, as_of, season, team), part in frame.groupby(
        ["snapshot_id", "as_of", "season", "team"], dropna=False
    ):
        notes = []
        qb = part[part["is_qb"]]
        if not qb.empty:
            top = qb.sort_values("load", ascending=False).iloc[0]
            notes.append(f"QB {top.get('player')} {top.get('status')}")
        n_ol = int(part.loc[part["is_ol"], "status"].astype(str).str.lower().isin(["out", "doubtful"]).sum())
        if n_ol:
            notes.append(f"{n_ol} OL out/doubtful")
        n_skill = int(part.loc[part["is_skill"], "status"].astype(str).str.lower().eq("out").sum())
        if n_skill:
            notes.append(f"{n_skill} skill Out")
        rows.append(
            {
                "snapshot_id": snapshot_id,
                "as_of": as_of,
                "season": season,
                "team": team,
                "injury_load": float(part["load"].sum()),
                "qb_load": float(part.loc[part["is_qb"], "load"].sum()),
                "skill_load": float(part.loc[part["is_skill"], "load"].sum()),
                "ol_load": float(part.loc[part["is_ol"], "load"].sum()),
                "def_load": float(part.loc[part["is_def"], "load"].sum()),
                "note": "; ".join(notes) if notes else None,
            }
        )
    return pd.DataFrame(rows)


def _pick_snapshot(agg: pd.DataFrame, team: Any, kickoff: pd.Timestamp) -> dict | None:
    if team is None or pd.isna(team) or agg.empty:
        return None
    part = agg[agg["team"].astype(str) == str(team)]
    if part.empty:
        # try prefix / alias
        key = normalize_player_name(team)
        part = agg[agg["team"].map(lambda t: normalize_player_name(t) == key)]
    if part.empty:
        return None
    if kickoff is None or pd.isna(kickoff):
        return part.sort_values("as_of_ts").iloc[-1].to_dict()
    before = part[part["as_of_ts"].notna() & (part["as_of_ts"] < kickoff)]
    if before.empty:
        return None
    return before.sort_values("as_of_ts").iloc[-1].to_dict()
