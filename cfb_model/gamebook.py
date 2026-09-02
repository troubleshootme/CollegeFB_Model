"""Broadcast-style HTML gamebook. No CFBD calls — render a simulated game dict."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _clock(seconds: int) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _stat_rows(home: dict, away: dict) -> list[tuple[str, str, str]]:
    def pair(label: str, fmt_home: str, fmt_away: str) -> tuple[str, str, str]:
        return (label, fmt_home, fmt_away)

    def frac(made, att) -> str:
        return f"{int(made or 0)}-{int(att or 0)}"

    return [
        pair("Total yards", str(home.get("total_yards", 0)), str(away.get("total_yards", 0))),
        pair("Plays", str(home.get("plays", 0)), str(away.get("plays", 0))),
        pair(
            "Rushing (att-yds-TD)",
            f"{home.get('rush_att', 0)}-{home.get('rush_yds', 0)}-{home.get('rush_td', 0)}",
            f"{away.get('rush_att', 0)}-{away.get('rush_yds', 0)}-{away.get('rush_td', 0)}",
        ),
        pair(
            "Passing (cmp-att-yds-TD-INT)",
            f"{home.get('pass_cmp', 0)}-{home.get('pass_att', 0)}-{home.get('pass_yds', 0)}-{home.get('pass_td', 0)}-{home.get('interceptions', 0)}",
            f"{away.get('pass_cmp', 0)}-{away.get('pass_att', 0)}-{away.get('pass_yds', 0)}-{away.get('pass_td', 0)}-{away.get('interceptions', 0)}",
        ),
        pair("Sacks-yards", f"{home.get('sacks', 0)}-{abs(int(home.get('sack_yds', 0)))}", f"{away.get('sacks', 0)}-{abs(int(away.get('sack_yds', 0)))}"),
        pair("First downs", str(home.get("first_downs", 0)), str(away.get("first_downs", 0))),
        pair("3rd down", frac(home.get("third_conv"), home.get("third_att")), frac(away.get("third_conv"), away.get("third_att"))),
        pair("4th down", frac(home.get("fourth_conv"), home.get("fourth_att")), frac(away.get("fourth_conv"), away.get("fourth_att"))),
        pair("Turnovers", str(home.get("turnovers", 0)), str(away.get("turnovers", 0))),
        pair("Punts-yds", f"{home.get('punts', 0)}-{home.get('punt_yds', 0)}", f"{away.get('punts', 0)}-{away.get('punt_yds', 0)}"),
        pair("FG", frac(home.get("fg_made"), home.get("fg_att")), frac(away.get("fg_made"), away.get("fg_att"))),
        pair("Red zone TD-FG-att", f"{home.get('red_zone_td', 0)}-{home.get('red_zone_fg', 0)}-{home.get('red_zone_att', 0)}", f"{away.get('red_zone_td', 0)}-{away.get('red_zone_fg', 0)}-{away.get('red_zone_att', 0)}"),
        pair("Time of possession", _clock(home.get("top_seconds", 0)), _clock(away.get("top_seconds", 0))),
    ]


def render_gamebook(result: dict[str, Any], *, title: str | None = None) -> str:
    book = result.get("gamebook") or result
    home = book["home"]
    away = book["away"]
    heading = title or f"{away['name']} at {home['name']}"
    wp = result.get("win_prob")
    spread = result.get("spread_label") or "no market"
    cover = result.get("cover_side") or "—"
    qh = [home.get("q1", 0), home.get("q2", 0), home.get("q3", 0), home.get("q4", 0), home.get("ot", 0)]
    qa = [away.get("q1", 0), away.get("q2", 0), away.get("q3", 0), away.get("q4", 0), away.get("ot", 0)]
    show_ot = qh[4] or qa[4]

    def qrow(name: str, qs: list[int], total: int) -> str:
        cells = "".join(f"<td>{n}</td>" for n in (qs[:4] + ([qs[4]] if show_ot else [])))
        return f"<tr><th>{_esc(name)}</th>{cells}<td class='tot'>{total}</td></tr>"

    headers = "Q1 Q2 Q3 Q4" + (" OT" if show_ot else "")
    head_cells = "".join(f"<th>{h}</th>" for h in headers.split()) + "<th>F</th>"

    stats = "".join(
        f"<tr><th>{_esc(label)}</th><td>{_esc(a)}</td><td>{_esc(h)}</td></tr>"
        for label, h, a in _stat_rows(home, away)
    )

    drives = book.get("drives") or []
    drive_html = "".join(
        f"<li><span class='d-team'>{_esc(d.get('team'))}</span> "
        f"Q{d.get('quarter')} · {d.get('plays', 0)} plays · {d.get('yards', 0)} yds · "
        f"<b>{_esc(d.get('result'))}</b>"
        f"{' · +' + str(d.get('points')) if d.get('points') else ''}</li>"
        for d in drives
    )

    plays = book.get("plays") or []
    pbp = "".join(
        f"<article class='snap q{p.get('quarter', 1)}'>"
        f"<div class='meta'><span>Q{p.get('quarter')}</span><span>{_esc(p.get('clock'))}</span>"
        f"<span>{p.get('down')}&{p.get('distance')}</span><span>{_esc(p.get('spot_label'))}</span></div>"
        f"<p>{_esc(p.get('description'))}</p>"
        f"<div class='score'>{_esc(away['name'][0:3])} {p.get('away_score')} · {_esc(home['name'][0:3])} {p.get('home_score')}</div>"
        f"</article>"
        for p in plays
        if p.get("play_type") not in {"kickoff"} or True
    )

    wp_line = ""
    if wp is not None:
        wp_line = (
            f"<p class='lede'>Ensemble {result.get('n_sims', 0)} worlds: "
            f"<b>{_esc(home['name'])} {float(wp)*100:.0f}%</b> · { _esc(spread) } · cover { _esc(cover) }. "
            f"This gamebook is the simulated world closest to the projected score "
            f"({result.get('mean_away', away.get('total')):.0f}–{result.get('mean_home', home.get('total')):.0f} expected).</p>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{_esc(heading)} · gamebook</title>
<style>
  :root {{ --ink:#e8f0e4; --mute:#8aa08a; --away:#6f1d1b; --home:#1b3d2f; --line:#2a3d2e; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:#0c110e; color:var(--ink); font:15px/1.45 "IBM Plex Sans", ui-sans-serif, system-ui; }}
  header {{ padding:28px 8vw 12px; }}
  .kicker {{ letter-spacing:.18em; text-transform:uppercase; font-size:11px; color:var(--mute); }}
  h1 {{ font:700 34px/1.1 "Teko", Impact, sans-serif; margin:8px 0 12px; }}
  .lede {{ max-width:62rem; color:#c5d4c4; }}
  .board {{ width:min(920px, 92vw); margin:0 8vw 28px; border:1px solid var(--line); }}
  table {{ width:100%; border-collapse:collapse; }}
  .board th, .board td {{ padding:8px 10px; text-align:center; }}
  .board th:first-child, .board td:first-child {{ text-align:left; }}
  .board thead {{ background:#152018; color:var(--mute); font-size:12px; letter-spacing:.08em; }}
  .tot {{ font-weight:700; }}
  .split {{ display:grid; grid-template-columns:1fr 1fr; min-height:8px; }}
  .split .a {{ background:var(--away); }}
  .split .h {{ background:var(--home); }}
  section {{ padding:8px 8vw 24px; }}
  h2 {{ font:600 13px/1 "IBM Plex Mono", monospace; letter-spacing:.14em; text-transform:uppercase; color:var(--mute); }}
  .box td, .box th {{ padding:6px 10px; border-bottom:1px solid #1c2a20; }}
  .box th {{ text-align:left; color:var(--mute); font-weight:500; }}
  .drives {{ list-style:none; padding:0; }}
  .drives li {{ padding:8px 0; border-bottom:1px solid #1c2a20; }}
  .d-team {{ display:inline-block; min-width:9rem; font-weight:650; }}
  .ticker {{ display:flex; flex-direction:column; gap:8px; }}
  .snap {{ border-left:3px solid #3d6b4a; padding:8px 12px; background:#101610; }}
  .snap.q2 {{ border-color:#6b8f4a; }}
  .snap.q3 {{ border-color:#b08b2e; }}
  .snap.q4 {{ border-color:#a33b2b; }}
  .meta {{ display:flex; gap:14px; color:var(--mute); font:12px/1 "IBM Plex Mono", monospace; }}
  .snap p {{ margin:6px 0 4px; }}
  .score {{ color:var(--mute); font-size:12px; }}
</style>
</head>
<body>
<header>
  <div class="kicker">Simulated gamebook · no CFBD play-by-play pulled</div>
  <h1>{_esc(heading)}</h1>
  {wp_line}
</header>
<div class="split"><div class="a"></div><div class="h"></div></div>
<section>
  <h2>Scoring by quarter</h2>
  <table class="board">
    <thead><tr><th></th>{head_cells}</tr></thead>
    <tbody>
      {qrow(away["name"], qa, away.get("total", 0))}
      {qrow(home["name"], qh, home.get("total", 0))}
    </tbody>
  </table>
</section>
<section>
  <h2>Offense / defense</h2>
  <table class="box">
    <thead><tr><th></th><th>{_esc(away["name"])}</th><th>{_esc(home["name"])}</th></tr></thead>
    <tbody>{stats}</tbody>
  </table>
</section>
<section>
  <h2>Drives</h2>
  <ol class="drives">{drive_html}</ol>
</section>
<section>
  <h2>Play by play</h2>
  <div class="ticker">{pbp}</div>
</section>
</body>
</html>
"""


def write_gamebook(result: dict[str, Any], path: Path, *, title: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_gamebook(result, title=title), encoding="utf-8")
    return path
