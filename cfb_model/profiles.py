"""Readable team and coach identity cards (terminal + HTML).

These are the assigned attributes the model actually uses: scheme, tempo,
explosiveness, 4th-down aggression, and whether they pile on or take the air
out when a game is in hand. No extra CFBD calls.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd

from cfb_model import config, store
from cfb_model.coaches import (
    PROFILE_VIEW,
    build_coach_season_table,
    build_team_profiles,
    coach_card,
    team_card,
)


def load_identity(
    conn=None,
    *,
    db_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    own = conn is None
    conn = conn or store.init_schema(store.connect(db_path) if db_path else None)
    try:
        games = store.read_table(conn, "games")
        coaches = store.read_table(conn, "coaches_seasons")
        team_stats = store.read_table(conn, "team_game_stats")
        advanced = store.read_table(conn, "advanced_game_stats")
        ppa = store.read_table(conn, "ppa_games")
        teams = store.read_table(conn, "teams")
    finally:
        if own:
            conn.close()
    coach_table = build_coach_season_table(coaches, games, team_stats, advanced, ppa, teams)
    team_table = build_team_profiles(games, team_stats, advanced, ppa, coach_table, teams)
    return team_table, coach_table


def find_team(team_table: pd.DataFrame, query: str, season: int) -> dict[str, Any] | None:
    return team_card(team_table, query, season)


def find_coach(coach_table: pd.DataFrame, query: str, season: int | None = None) -> dict[str, Any] | None:
    return coach_card(coach_table, query, season)


def format_card(card: dict[str, Any]) -> str:
    if not card:
        return "No profile found."
    title = card.get("coach_name") or card.get("qb_name") or card.get("school") or "Unknown"
    kind = "Coach" if card.get("kind") == "coach" else "QB" if card.get("kind") == "qb" else "Team"
    lines = [
        f"{title}",
        f"{kind} profile · {card.get('season')} · {card.get('school') or ''}".strip(),
        f"Scheme: {card.get('scheme')} · When up big: {card.get('mercy')}",
        "",
    ]
    for attr in card.get("attributes") or []:
        value = attr.get("value")
        zval = attr.get("z")
        shown = _fmt_value(attr["key"], value)
        bar = _bar(zval)
        ztxt = "" if zval is None else f"  {zval:+.2f}σ"
        lines.append(f"  {attr['label']:<28} {shown:<10} {bar}{ztxt}")
        lines.append(f"  {'':<28} {attr['low']} ← → {attr['high']}")
    stops = card.get("stops")
    if stops:
        lines.append("")
        lines.append("Stops")
        for stop in stops:
            lines.append(
                f"  {stop.get('season')}  {stop.get('school')}  {stop.get('scheme')}  {stop.get('mercy')}"
            )
    return "\n".join(lines)


def render_card_html(card: dict[str, Any], *, related: dict[str, Any] | None = None) -> str:
    title = html.escape(str(card.get("coach_name") or card.get("school") or "Profile"))
    kind = "Coach" if card.get("kind") == "coach" else "Team"
    scheme = html.escape(str(card.get("scheme") or "balanced"))
    mercy = html.escape(str(card.get("mercy") or "standard"))
    season = card.get("season") or ""
    school = html.escape(str(card.get("school") or ""))
    rows = []
    for attr in card.get("attributes") or []:
        zval = attr.get("z")
        pct = 50 if zval is None else max(4, min(96, 50 + 18 * float(zval)))
        rows.append(
            f"<tr><th>{html.escape(attr['label'])}</th>"
            f"<td class='val'>{html.escape(_fmt_value(attr['key'], attr.get('value')))}</td>"
            f"<td class='track'><span class='fill' style='width:{pct:.0f}%'></span></td>"
            f"<td class='z'>{'' if zval is None else f'{float(zval):+.2f}σ'}</td>"
            f"<td class='hint'>{html.escape(attr['low'])} → {html.escape(attr['high'])}</td></tr>"
        )
    related_html = ""
    if related:
        related_html = (
            "<section><h2>Assigned coach</h2>"
            f"<p class='lede'><a href='coach_{_slug(related.get('coach_name'))}.html'>"
            f"{html.escape(str(related.get('coach_name')))}</a> · "
            f"{html.escape(str(related.get('scheme')))} · "
            f"{html.escape(str(related.get('mercy')))}</p></section>"
        )
    stops = card.get("stops") or []
    stop_html = ""
    if stops:
        items = "".join(
            f"<li>{s.get('season')} · {html.escape(str(s.get('school')))} · "
            f"{html.escape(str(s.get('scheme')))} · {html.escape(str(s.get('mercy')))}</li>"
            for s in stops
        )
        stop_html = f"<section><h2>Stops</h2><ol class='stops'>{items}</ol></section>"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title} · {kind} profile</title>
<style>
  :root {{ --ink:#e8f0e4; --mute:#8aa08a; --line:#2a3d2e; --fill:#3d6b4a; }}
  body {{ margin:0; background:#0c110e; color:var(--ink); font:15px/1.45 "IBM Plex Sans", ui-sans-serif, system-ui; }}
  header {{ padding:28px 8vw 8px; }}
  .kicker {{ letter-spacing:.18em; text-transform:uppercase; font-size:11px; color:var(--mute); }}
  h1 {{ font:700 34px/1.1 Impact, "Teko", sans-serif; margin:8px 0 8px; }}
  .chips {{ display:flex; gap:8px; flex-wrap:wrap; margin:12px 0 20px; }}
  .chip {{ border:1px solid var(--line); padding:4px 10px; font-size:12px; letter-spacing:.06em; text-transform:uppercase; }}
  section {{ padding:8px 8vw 28px; }}
  h2 {{ font:600 12px/1 "IBM Plex Mono", monospace; letter-spacing:.14em; text-transform:uppercase; color:var(--mute); }}
  table {{ width:min(920px, 92vw); border-collapse:collapse; }}
  th {{ text-align:left; font-weight:500; color:var(--mute); padding:8px 8px 8px 0; width:14rem; }}
  td {{ padding:8px; }}
  .val {{ font-variant-numeric:tabular-nums; width:6rem; }}
  .track {{ width:180px; background:#152018; height:8px; }}
  .track {{ position:relative; }}
  .fill {{ display:block; height:8px; background:var(--fill); }}
  .z {{ color:var(--mute); font:12px/1 "IBM Plex Mono", monospace; }}
  .hint {{ color:#6d806c; font-size:12px; }}
  a {{ color:#b4d7b0; }}
  .stops {{ padding-left:1.2rem; }}
  .lede {{ color:#c5d4c4; }}
</style>
</head>
<body>
<header>
  <div class="kicker">{kind} identity · {season} · no play-by-play ingest</div>
  <h1>{title}</h1>
  <p class="lede">{school}</p>
  <div class="chips">
    <span class="chip">{scheme}</span>
    <span class="chip">{mercy}</span>
  </div>
</header>
<section>
  <h2>Assigned attributes</h2>
  <table><tbody>{''.join(rows)}</tbody></table>
</section>
{related_html}
{stop_html}
</body>
</html>
"""


def render_index_html(teams: list[dict[str, Any]], coaches: list[dict[str, Any]], *, season: int) -> str:
    team_links = "".join(
        f"<li><a href='team_{_slug(t.get('school'))}.html'>{html.escape(str(t.get('school')))}</a> "
        f"· {html.escape(str(t.get('scheme')))} · {html.escape(str(t.get('mercy')))}"
        f"{' · ' + html.escape(str(t.get('coach_name'))) if t.get('coach_name') else ''}</li>"
        for t in teams
    )
    coach_links = "".join(
        f"<li><a href='coach_{_slug(c.get('coach_name'))}.html'>{html.escape(str(c.get('coach_name')))}</a> "
        f"· {html.escape(str(c.get('school') or ''))} · {html.escape(str(c.get('scheme')))} "
        f"· {html.escape(str(c.get('mercy')))}</li>"
        for c in coaches
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Identity profiles {season}</title>
<style>
  body {{ margin:0; background:#0c110e; color:#e8f0e4; font:15px/1.5 "IBM Plex Sans", system-ui; }}
  header, section {{ padding:24px 8vw; }}
  a {{ color:#b4d7b0; }}
  .kicker {{ letter-spacing:.18em; text-transform:uppercase; font-size:11px; color:#8aa08a; }}
  h1 {{ font:700 34px/1.1 Impact, sans-serif; }}
  h2 {{ letter-spacing:.12em; text-transform:uppercase; font-size:12px; color:#8aa08a; }}
  li {{ padding:6px 0; border-bottom:1px solid #1c2a20; }}
</style></head>
<body>
<header>
  <div class="kicker">College football identity</div>
  <h1>{season} team and coach profiles</h1>
  <p>Scheme, tempo, 4th-down aggression, and whether they pile on or take the air out.</p>
</header>
<section><h2>Teams</h2><ul>{team_links}</ul></section>
<section><h2>Coaches</h2><ul>{coach_links}</ul></section>
</body></html>
"""


def write_catalog(team_table: pd.DataFrame, coach_table: pd.DataFrame, dest: Path, *, season: int) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    teams = _season_rows(team_table, season, kind="team")
    coaches = _season_rows(coach_table, season, kind="coach")
    (dest / "index.html").write_text(render_index_html(teams, coaches, season=season), encoding="utf-8")
    for team in teams:
        related = None
        if team.get("coach_name") and not coach_table.empty:
            related = find_coach(coach_table, str(team["coach_name"]), season)
        (dest / f"team_{_slug(team.get('school'))}.html").write_text(
            render_card_html(team, related=related), encoding="utf-8"
        )
    for coach in coaches:
        (dest / f"coach_{_slug(coach.get('coach_name'))}.html").write_text(
            render_card_html(coach), encoding="utf-8"
        )
    return dest / "index.html"


def _season_rows(table: pd.DataFrame, season: int, *, kind: str) -> list[dict[str, Any]]:
    if table is None or table.empty:
        return []
    part = table[table["season"] == season]
    cards = []
    if kind == "team":
        for school in sorted(part["school"].dropna().unique()):
            card = team_card(part, str(school), season)
            if card:
                cards.append(card)
    else:
        names = part["coach_name"].dropna().unique() if "coach_name" in part else []
        for name in sorted(names, key=lambda n: str(n)):
            card = coach_card(part, str(name), season)
            if card:
                cards.append(card)
    return cards


def _fmt_value(key: str, value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if key in {"rush_share", "success_rate"}:
        return f"{100 * float(value):.1f}%"
    if key in {"plays_pg", "fourth_att_pg", "sec_per_play"}:
        return f"{float(value):.1f}"
    if key == "run_up":
        return f"{float(value):+.2f}"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _bar(z: Any, width: int = 11) -> str:
    if z is None or (isinstance(z, float) and pd.isna(z)):
        return "·" * width
    mid = width // 2
    pos = int(round(mid + max(-mid, min(mid, float(z) * 2))))
    return "".join("█" if i == pos else "─" for i in range(width))


def _slug(value: Any) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value or "unknown"))
    while "--" in text:
        text = text.replace("--", "-")
    return text.strip("-") or "unknown"
