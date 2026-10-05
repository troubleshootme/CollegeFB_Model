"""Drive-level college football simulator.

Monte Carlo possessions: success rate + explosiveness, tempo, red-zone finishing,
special teams, weather, and a game-level efficiency shock (so one drive is not
independent of the next). One seedable world is the published gamebook; an
ensemble of worlds supplies win/cover/total probabilities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from cfb_model.identity import (
    PASS_YARDS_OPPONENT_WEIGHT,
    RUSH_SHARE_OPPONENT_WEIGHT,
    RUSH_YARDS_OPPONENT_WEIGHT,
    expected_rush_share,
)

Q_SECONDS = 15 * 60
MAX_PLAYS = 280
OWN_GOAL = 0
OPP_GOAL = 100


@dataclass
class Ratings:
    name: str
    off_ppa: float = 0.0
    def_ppa: float = 0.0
    off_success: float = 0.42
    def_success: float = 0.42
    explosiveness: float = 1.15
    havoc: float = 0.12
    rush_share: float = 0.48
    finish: float = 0.55
    def_rush_allowed: float = 0.48
    rush_quality: float = 0.0
    grind: float = 0.5
    explode_fast: float = 0.0
    fourth_aggression: float = 0.0
    seconds_per_play: float = 28.0
    run_up: float = 0.0
    scheme: str = "balanced"
    coach: str = ""


@dataclass
class TeamBox:
    name: str
    q1: int = 0
    q2: int = 0
    q3: int = 0
    q4: int = 0
    ot: int = 0
    total: int = 0
    plays: int = 0
    rush_att: int = 0
    rush_yds: int = 0
    rush_td: int = 0
    pass_cmp: int = 0
    pass_att: int = 0
    pass_yds: int = 0
    pass_td: int = 0
    interceptions: int = 0
    sacks: int = 0
    sack_yds: int = 0
    first_downs: int = 0
    first_rush: int = 0
    first_pass: int = 0
    third_att: int = 0
    third_conv: int = 0
    fourth_att: int = 0
    fourth_conv: int = 0
    fumbles: int = 0
    fumbles_lost: int = 0
    turnovers: int = 0
    punts: int = 0
    punt_yds: int = 0
    fg_att: int = 0
    fg_made: int = 0
    penalties: int = 0
    penalty_yds: int = 0
    red_zone_att: int = 0
    red_zone_td: int = 0
    red_zone_fg: int = 0
    top_seconds: int = 0
    total_yards: int = 0

    def add_points(self, quarter: int, pts: int) -> None:
        if quarter <= 1:
            self.q1 += pts
        elif quarter == 2:
            self.q2 += pts
        elif quarter == 3:
            self.q3 += pts
        elif quarter == 4:
            self.q4 += pts
        else:
            self.ot += pts
        self.total += pts


@dataclass
class Engine:
    home: Ratings
    away: Ratings
    rng: np.random.Generator
    pred_margin: float = 0.0
    total: float = 52.0
    precip: float = 0.0
    wind: float = 0.0
    indoor: bool = False
    home_off_shock: float = 0.0
    away_off_shock: float = 0.0
    q: int = 1
    qclock: int = Q_SECONDS
    down: int = 1
    dist: int = 10
    spot: int = 25
    home_has_ball: bool = False
    waiting_ko: bool = True
    ko_to_home: bool = False
    home_box: TeamBox = field(init=False)
    away_box: TeamBox = field(init=False)
    plays: list[dict[str, Any]] = field(default_factory=list)
    drives: list[dict[str, Any]] = field(default_factory=list)
    drive_no: int = 0
    drive_plays: int = 0
    drive_yards: int = 0
    drive_start: int = 25
    in_red_zone: bool = False

    def __post_init__(self) -> None:
        self.home_box = TeamBox(name=self.home.name)
        self.away_box = TeamBox(name=self.away.name)

    @property
    def offense(self) -> Ratings:
        return self.home if self.home_has_ball else self.away

    @property
    def defense(self) -> Ratings:
        return self.away if self.home_has_ball else self.home

    @property
    def off_box(self) -> TeamBox:
        return self.home_box if self.home_has_ball else self.away_box

    @property
    def def_box(self) -> TeamBox:
        return self.away_box if self.home_has_ball else self.home_box

    def other_name(self) -> str:
        return self.away.name if self.home_has_ball else self.home.name

    def clock_label(self) -> str:
        m, s = divmod(max(0, self.qclock), 60)
        return f"{m}:{s:02d}"

    def spot_label(self) -> str:
        if self.spot <= 0:
            return f"{self.offense.name} GL"
        if self.spot >= 100:
            return f"{self.other_name()} GL"
        if self.spot < 50:
            return f"{self.offense.name} {self.spot}"
        if self.spot > 50:
            return f"{self.other_name()} {100 - self.spot}"
        return "50"

    def off_delta(self) -> float:
        shock = self.home_off_shock if self.home_has_ball else self.away_off_shock
        weather = 0.0
        if not self.indoor:
            weather -= 0.03 * min(self.precip, 8.0) / 8.0
            weather -= 0.02 * min(self.wind, 35.0) / 35.0
        bias = 0.012 * self.pred_margin * (1 if self.home_has_ball else -1)
        return float(self.offense.off_ppa - self.defense.def_ppa + shock + weather + bias)

    def success_p(self) -> float:
        base = 0.5 * (self.offense.off_success + (1.0 - self.defense.def_success) + 0.42)
        return float(np.clip(base + 0.28 * self.off_delta(), 0.24, 0.62))

    def explosive_p(self) -> float:
        expl = self.offense.explosiveness
        p = 0.07 + 0.06 * (expl - 1.0) + 0.08 * max(0.0, self.off_delta())
        if not self.indoor:
            p -= 0.02 * min(self.wind, 30.0) / 30.0
        return float(np.clip(p, 0.04, 0.20))

    def to_p(self) -> float:
        return float(np.clip(0.012 + 0.35 * self.defense.havoc - 0.04 * self.off_delta(), 0.008, 0.07))

    def rush_share(self) -> float:
        share = expected_rush_share(self.offense.rush_share, self.defense.def_rush_allowed)
        if self.precip >= 1.5:
            share += 0.08
        if self.q == 4 and self.qclock < 180:
            trailing = (self.home_box.total < self.away_box.total) if self.home_has_ball else (
                self.away_box.total < self.home_box.total
            )
            share += -0.18 if trailing else 0.12
        if self.lead() >= 14 and self.q >= 4:
            share += 0.08 * max(0.0, self.offense.grind)
        return float(np.clip(share, 0.22, 0.78))

    def lead(self) -> int:
        off = self.off_box.total
        deff = self.def_box.total
        return off - deff

    def rush_success_p(self) -> float:
        """Rushing success/yards swing with the opponent. Stout run D's take you off your average."""
        own = self.success_p()
        opp = float(np.clip(0.42 - 0.5 * self.defense.havoc + 0.15 * (0.42 - self.defense.def_success), 0.20, 0.62))
        blended = (1.0 - RUSH_YARDS_OPPONENT_WEIGHT) * own + RUSH_YARDS_OPPONENT_WEIGHT * opp
        return float(np.clip(blended + 0.10 * self.offense.rush_quality, 0.18, 0.68))

    def pass_success_p(self) -> float:
        """Passing stays close to the offense's own average."""
        own = self.success_p()
        opp = float(np.clip(0.42 - 0.35 * self.defense.havoc, 0.24, 0.60))
        blended = (1.0 - PASS_YARDS_OPPONENT_WEIGHT) * own + PASS_YARDS_OPPONENT_WEIGHT * opp
        return float(np.clip(blended, 0.22, 0.64))

    def tick(self, seconds: int) -> None:
        used = min(self.qclock, max(1, seconds))
        self.qclock -= used
        self.off_box.top_seconds += used

    def emit(
        self,
        play_type: str,
        yards: int,
        description: str,
        *,
        down: int | None = None,
        complete: bool | None = None,
    ) -> None:
        self.plays.append(
            {
                "play_id": len(self.plays) + 1,
                "drive": self.drive_no,
                "quarter": self.q,
                "clock": self.clock_label(),
                "down": self.down if down is None else down,
                "distance": self.dist,
                "spot": self.spot,
                "spot_label": self.spot_label(),
                "possession": self.offense.name,
                "play_type": play_type,
                "yards": int(yards),
                "complete": complete,
                "description": description,
                "home_score": self.home_box.total,
                "away_score": self.away_box.total,
            }
        )

    def start_drive(self, spot: int, *, kickoff: bool = False) -> None:
        self.down = 1
        self.dist = 10
        self.spot = int(np.clip(spot, 1, 99))
        self.drive_no += 1
        self.drive_plays = 0
        self.drive_yards = 0
        self.drive_start = self.spot
        self.in_red_zone = False
        if kickoff:
            self.emit("kickoff", 0, f"Kickoff, {self.offense.name} at the {self.spot_label()}")

    def close_drive(self, result: str, points: int = 0) -> None:
        self.drives.append(
            {
                "drive": self.drive_no,
                "team": self.offense.name,
                "quarter": self.q,
                "start": self.drive_start,
                "plays": self.drive_plays,
                "yards": self.drive_yards,
                "result": result,
                "points": points,
            }
        )

    def first_down(self, how: str) -> None:
        self.down = 1
        self.dist = 10
        self.off_box.first_downs += 1
        if how == "rush":
            self.off_box.first_rush += 1
        else:
            self.off_box.first_pass += 1

    def gain(self, yards: int, how: str) -> str:
        yards = int(yards)
        self.drive_yards += yards
        self.spot = int(np.clip(self.spot + yards, 0, 100))
        if self.spot >= 80 and not self.in_red_zone:
            self.in_red_zone = True
            self.off_box.red_zone_att += 1
        if self.spot >= OPP_GOAL:
            return "td"
        if self.spot <= OWN_GOAL:
            return "safety"
        if yards >= self.dist:
            self.first_down(how)
            return "first"
        self.dist -= yards
        self.down += 1
        return "continue"

    def score_td(self) -> int:
        pts = 6
        self.off_box.add_points(self.q, 6)
        if self.rng.random() < 0.94:
            self.off_box.add_points(self.q, 1)
            pts += 1
            extra = "PAT good"
        else:
            extra = "PAT missed"
        if self.in_red_zone:
            self.off_box.red_zone_td += 1
        self.close_drive("Touchdown", pts)
        self.emit("td", 0, f"TOUCHDOWN {self.offense.name}. {extra}")
        self.waiting_ko = True
        self.ko_to_home = not self.home_has_ball
        return pts

    def field_goal(self) -> bool:
        attempt = max(18, (100 - self.spot) + 17)
        make_p = float(1.0 / (1.0 + np.exp((attempt - 41.0) / 5.5)))
        if not self.indoor:
            make_p -= 0.08 * min(self.wind, 30.0) / 30.0
            make_p -= 0.04 * min(self.precip, 6.0) / 6.0
        make_p = float(np.clip(make_p, 0.08, 0.97))
        self.off_box.fg_att += 1
        self.tick(6)
        good = self.rng.random() < make_p
        if good:
            self.off_box.fg_made += 1
            self.off_box.add_points(self.q, 3)
            if self.in_red_zone:
                self.off_box.red_zone_fg += 1
            self.emit("fg", 0, f"{self.offense.name} {attempt}-yard field goal is GOOD")
            self.close_drive("Field goal", 3)
            self.waiting_ko = True
            self.ko_to_home = not self.home_has_ball
            return True
        self.emit("fg", 0, f"{self.offense.name} {attempt}-yard field goal is NO GOOD")
        self.close_drive("Missed FG")
        self.home_has_ball = not self.home_has_ball
        self.start_drive(100 - self.spot)
        return False

    def punt(self) -> None:
        net = int(np.clip(self.rng.normal(39, 8), 18, 62))
        self.off_box.punts += 1
        self.off_box.punt_yds += net
        self.tick(7)
        dest = self.spot + net
        self.emit("punt", net, f"{self.offense.name} punts {net} yards")
        self.close_drive("Punt")
        self.home_has_ball = not self.home_has_ball
        if dest >= 100:
            self.start_drive(20)
        else:
            self.start_drive(100 - dest)

    def turnover(self, kind: str, yards: int = 0) -> None:
        self.off_box.turnovers += 1
        if kind == "int":
            self.off_box.interceptions += 1
        else:
            self.off_box.fumbles += 1
            self.off_box.fumbles_lost += 1
        self.tick(8)
        self.emit(kind, yards, f"{kind.upper()} {self.offense.name}")
        self.close_drive("Interception" if kind == "int" else "Fumble")
        self.home_has_ball = not self.home_has_ball
        self.start_drive(int(np.clip(100 - (self.spot + yards), 1, 99)))

    def safety(self) -> None:
        self.def_box.add_points(self.q, 2)
        self.emit("safety", 0, f"Safety — {self.defense.name}")
        self.close_drive("Safety")
        self.waiting_ko = True
        self.ko_to_home = self.home_has_ball

    def maybe_fourth(self) -> str | None:
        if self.down != 4:
            return None
        self.off_box.fourth_att += 1
        fg_dist = (100 - self.spot) + 17
        trailing = (self.home_box.total < self.away_box.total) if self.home_has_ball else (
            self.away_box.total < self.home_box.total
        )
        late = self.q >= 4 and self.qclock < 240
        go = self.dist <= 1 or (self.spot >= 60 and self.dist <= 3) or (late and trailing)
        if (not go) and 27 <= fg_dist <= 54 and not (late and trailing and fg_dist > 48):
            return "fg"
        if go:
            return "go"
        return "punt"

    def snap(self) -> None:
        fourth = self.maybe_fourth()
        if fourth == "fg":
            self.field_goal()
            return
        if fourth == "punt":
            self.punt()
            return

        if self.down == 3:
            self.off_box.third_att += 1

        is_rush = self.rng.random() < self.rush_share()
        to = self.rng.random() < self.to_p()
        success = self.rng.random() < (self.rush_success_p() if is_rush else self.pass_success_p())
        boom = self.rng.random() < self.explosive_p()

        if is_rush:
            self.run_play(success, boom, to)
        else:
            self.pass_play(success, boom, to)

        if fourth == "go" and self.down > 4 and self.spot < OPP_GOAL:
            self.off_box.fourth_conv += 0  # failed
            self.emit("turnover_downs", 0, f"{self.offense.name} turnover on downs")
            self.close_drive("Downs")
            self.home_has_ball = not self.home_has_ball
            self.start_drive(100 - self.spot)

    def after_gain(self, result: str, how: str, td_kind: str) -> None:
        if self.down == 3 and result in {"first", "td"}:
            self.off_box.third_conv += 1
        if self.down == 4 and result in {"first", "td"}:
            self.off_box.fourth_conv += 1
        if result == "td":
            if td_kind == "rush":
                self.off_box.rush_td += 1
            else:
                self.off_box.pass_td += 1
            self.score_td()
        elif result == "safety":
            self.safety()
        elif result == "continue" and self.down > 4:
            self.emit("turnover_downs", 0, f"{self.offense.name} turnover on downs")
            self.close_drive("Downs")
            self.home_has_ball = not self.home_has_ball
            self.start_drive(100 - max(1, self.spot))

    def run_play(self, success: bool, boom: bool, to: bool) -> None:
        box = self.off_box
        box.plays += 1
        box.rush_att += 1
        self.drive_plays += 1
        if to and self.rng.random() < 0.45:
            yds = int(self.rng.normal(1, 3))
            box.rush_yds += yds
            self.tick(6)
            self.emit("rush", yds, f"{self.offense.name} rush for {yds}, FUMBLE")
            self.turnover("fumble", yds)
            return
        if boom:
            yds = int(np.clip(self.rng.normal(18, 7), 9, 75))
        elif success:
            yds = int(np.clip(self.rng.normal(5.2, 2.2), 2, 14))
        else:
            yds = int(np.clip(self.rng.normal(-0.4, 2.0), -6, 3))
        box.rush_yds += yds
        self.tick(int(self.rng.integers(26, 38)))
        result = self.gain(yds, "rush")
        dest = self.spot_label()
        note = "FIRST DOWN" if result == "first" else ("TOUCHDOWN" if result == "td" else f"{self.down} & {self.dist}")
        self.emit("rush", yds, f"{self.offense.name} rush for {yds} to the {dest}. {note}")
        self.after_gain(result, "rush", "rush")

    def pass_play(self, success: bool, boom: bool, to: bool) -> None:
        box = self.off_box
        box.plays += 1
        box.pass_att += 1
        self.drive_plays += 1
        sack_p = float(np.clip(0.05 + 0.4 * self.defense.havoc, 0.03, 0.12))
        if self.rng.random() < sack_p:
            yds = int(np.clip(self.rng.normal(-7.0, 2.2), -16, -3))
            box.sacks += 1
            box.sack_yds += yds
            box.pass_yds += yds
            self.tick(int(self.rng.integers(22, 32)))
            self.emit("sack", yds, f"{self.offense.name} sacked for {yds}", complete=False)
            result = self.gain(yds, "pass")
            self.after_gain(result, "pass", "pass")
            return
        if to:
            self.tick(7)
            self.emit("pass", 0, f"{self.offense.name} pass intercepted", complete=False)
            self.turnover("int", int(self.rng.integers(0, 12)))
            return
        if not success and self.rng.random() < 0.72:
            self.tick(int(self.rng.integers(6, 10)))
            self.emit("pass", 0, f"{self.offense.name} incomplete pass", complete=False)
            self.down += 1
            if self.down > 4:
                self.emit("turnover_downs", 0, f"{self.offense.name} turnover on downs")
                self.close_drive("Downs")
                self.home_has_ball = not self.home_has_ball
                self.start_drive(100 - self.spot)
            return
        box.pass_cmp += 1
        if boom:
            yds = int(np.clip(self.rng.normal(26, 10), 12, 80))
        elif success:
            yds = int(np.clip(self.rng.normal(9.5, 4.0), 4, 22))
        else:
            yds = int(np.clip(self.rng.normal(3.5, 2.5), -2, 8))
        box.pass_yds += yds
        self.tick(int(self.rng.integers(8, 28)))
        result = self.gain(yds, "pass")
        dest = self.spot_label()
        note = "FIRST DOWN" if result == "first" else ("TOUCHDOWN" if result == "td" else f"{self.down} & {self.dist}")
        self.emit("pass", yds, f"{self.offense.name} pass complete for {yds} to the {dest}. {note}", complete=True)
        self.after_gain(result, "pass", "pass")

    def kickoff_touchback(self) -> None:
        self.home_has_ball = self.ko_to_home
        self.waiting_ko = False
        self.start_drive(25, kickoff=True)

    def end_period(self) -> None:
        if self.q == 2:
            if self.drive_plays:
                self.close_drive("End of half")
            self.q = 3
            self.qclock = Q_SECONDS
            self.waiting_ko = True
            self.ko_to_home = True
        elif self.q == 4:
            return
        else:
            self.q += 1
            self.qclock = Q_SECONDS

    def maybe_kneel(self) -> bool:
        if self.q != 4 or self.qclock > 45:
            return False
        leading = (self.home_box.total > self.away_box.total) if self.home_has_ball else (
            self.away_box.total > self.home_box.total
        )
        if not leading:
            return False
        self.tick(self.qclock)
        self.emit("kneel", -1, f"{self.offense.name} kneels, game over")
        self.close_drive("Kneel")
        return True

    def overtime(self) -> None:
        self.q = 5
        self.qclock = Q_SECONDS
        first_home = bool(self.rng.integers(0, 2))
        for poss in (first_home, not first_home):
            self.home_has_ball = poss
            self.waiting_ko = False
            self.start_drive(25)
            snaps = 0
            start_score = self.off_box.total
            while snaps < 12 and self.off_box.total == start_score and self.down <= 4:
                snaps += 1
                self.snap()
                if self.waiting_ko:
                    break
            if self.off_box.total == start_score and self.down > 4:
                self.field_goal()
        if self.home_box.total == self.away_box.total:
            self.home_box.add_points(5, 0)

    def play_game(self) -> dict[str, Any]:
        self.waiting_ko = True
        self.ko_to_home = False
        plays = 0
        while plays < MAX_PLAYS:
            plays += 1
            if self.qclock <= 0:
                if self.q >= 4:
                    break
                self.end_period()
                continue
            if self.waiting_ko:
                self.kickoff_touchback()
                continue
            if self.maybe_kneel():
                break
            self.snap()
        if self.home_box.total == self.away_box.total:
            self.overtime()
        for box in (self.home_box, self.away_box):
            box.total_yards = box.rush_yds + box.pass_yds
        return _game_dict(self)


def _num(row: pd.Series, key: str, default: float) -> float:
    if key not in row.index:
        return default
    value = pd.to_numeric(row.get(key), errors="coerce")
    if pd.isna(value):
        return default
    return float(value)


def ratings_from_row(row: pd.Series) -> tuple[Ratings, Ratings]:
    def side(prefix: str, name_key: str) -> Ratings:
        name = str(row.get(name_key) or prefix)
        rush = _num(row, f"{prefix}_off_rushing_ppa_std", _num(row, f"{prefix}_prior_off_ppa", 0.0))
        pass_p = _num(row, f"{prefix}_off_passing_ppa_std", rush)
        share = float(np.clip(0.48 + 0.45 * (rush - pass_p), 0.32, 0.68))
        off = _num(row, f"{prefix}_off_ppa_adj_std", _num(row, f"{prefix}_off_ppa_std", _num(row, f"{prefix}_prior_off_ppa", 0.0)))
        deff = _num(row, f"{prefix}_def_ppa_adj_std", _num(row, f"{prefix}_def_ppa_std", _num(row, f"{prefix}_prior_def_ppa", 0.0)))
        if f"{prefix}_qb_ppa_overall" in row.index or f"{prefix}_qb_dynamic" in row.index:
            qb_ppa = _num(row, f"{prefix}_qb_ppa_overall", 0.0)
            qb_dyn = _num(row, f"{prefix}_qb_dynamic", 0.0)
            qb_out = _num(row, f"{prefix}_qb_out", 0.0)
            off = off + 0.35 * qb_ppa - 0.18 * qb_out * (1.0 + 0.45 * qb_dyn)
            share = float(np.clip(share + 0.12 * qb_dyn - 0.08 * qb_out, 0.28, 0.72))
        return Ratings(
            name=name,
            off_ppa=off,
            def_ppa=deff,
            off_success=_num(row, f"{prefix}_off_success_std", 0.42),
            def_success=_num(row, f"{prefix}_def_success_std", 0.42),
            explosiveness=_num(row, f"{prefix}_off_expl_std", 1.15),
            havoc=_num(row, f"{prefix}_havoc_std", 0.12),
            rush_share=share,
            finish=float(np.clip(0.52 + 0.4 * off, 0.38, 0.72)),
        )

    return side("home", "home_team"), side("away", "away_team")


def simulate_game(
    home: Ratings,
    away: Ratings,
    *,
    seed: int | None = None,
    pred_margin: float = 0.0,
    total: float = 52.0,
    precip: float = 0.0,
    wind: float = 0.0,
    indoor: bool = False,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    engine = Engine(
        home=home,
        away=away,
        rng=rng,
        pred_margin=pred_margin,
        total=total,
        precip=precip,
        wind=wind,
        indoor=indoor,
        home_off_shock=float(rng.normal(0, 0.07)),
        away_off_shock=float(rng.normal(0, 0.07)),
    )
    return engine.play_game()


def simulate_matchup(row: pd.Series, n_sims: int = 200, seed: int = 0) -> dict[str, Any]:
    home, away = ratings_from_row(row)
    pred_margin = _num(row, "pred_margin", 0.0)
    total = _num(row, "close_total", _num(row, "over_under", 52.0))
    precip = _num(row, "precipitation", 0.0)
    wind = _num(row, "wind_speed", 0.0)
    indoor = bool(_num(row, "dome", 0.0) or _num(row, "indoor", 0.0))
    games = []
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 10_000_000, size=n_sims)
    for sim_seed in seeds:
        games.append(
            simulate_game(
                home,
                away,
                seed=int(sim_seed),
                pred_margin=pred_margin,
                total=total,
                precip=precip,
                wind=wind,
                indoor=indoor,
            )
        )
    home_scores = np.array([g["home"]["total"] for g in games], dtype=float)
    away_scores = np.array([g["away"]["total"] for g in games], dtype=float)
    margins = home_scores - away_scores
    sim_wp = float((margins > 0).mean() + 0.5 * (margins == 0).mean())
    from scipy.stats import norm
    from cfb_model import config

    card_wp = float(norm.cdf(pred_margin / (config.DEFAULT_MARGIN_SIGMA or 16.0)))
    win_prob = 0.55 * sim_wp + 0.45 * card_wp
    spread = _num(row, "close_spread", np.nan)
    cover = None
    cover_rate = None
    if np.isfinite(spread):
        home_covers = (margins + spread) > 0
        cover_rate = float(home_covers.mean())
        if abs(float(np.mean(margins + spread))) >= 0.25:
            cover = home.name if cover_rate >= 0.5 else away.name
    target_h = (total + pred_margin) / 2.0
    target_a = (total - pred_margin) / 2.0
    dist = (home_scores - target_h) ** 2 + (away_scores - target_a) ** 2
    gamebook = games[int(np.argmin(dist))]
    return {
        "home_team": home.name,
        "away_team": away.name,
        "n_sims": n_sims,
        "win_prob": round(win_prob, 4),
        "away_win_prob": round(1.0 - win_prob, 4),
        "mean_home": round(float(home_scores.mean()), 2),
        "mean_away": round(float(away_scores.mean()), 2),
        "mean_total": round(float((home_scores + away_scores).mean()), 2),
        "cover_side": cover,
        "cover_rate": None if cover_rate is None else round(cover_rate, 4),
        "pred_margin": pred_margin,
        "close_spread": None if not np.isfinite(spread) else spread,
        "spread_label": _spread_label(home.name, away.name, spread),
        "gamebook": gamebook,
    }


def _spread_label(home: str, away: str, spread: float) -> str | None:
    if spread is None or not np.isfinite(spread):
        return None
    if abs(spread) < 1e-9:
        return "pick'em"
    mag = abs(spread)
    mag_s = str(int(mag)) if mag == int(mag) else f"{mag:.2f}".rstrip("0").rstrip(".")
    if spread < 0:
        return f"{home} -{mag_s}"
    return f"{away} -{mag_s}"


def _game_dict(engine: Engine) -> dict[str, Any]:
    home = asdict(engine.home_box)
    away = asdict(engine.away_box)
    return {
        "home": home,
        "away": away,
        "plays": engine.plays,
        "drives": engine.drives,
        "quarters": {
            "home": [home["q1"], home["q2"], home["q3"], home["q4"], home["ot"]],
            "away": [away["q1"], away["q2"], away["q3"], away["q4"], away["ot"]],
        },
    }
