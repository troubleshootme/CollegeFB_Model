"""Score-by-quarter model built from scoring events instead of rounded averages.

Why not round ``total/2 +/- margin/2``: real scores are lumpy. Across 10,089 FBS games
a team never finished with 1 or 4 points, and 2 or 5 happened ~0.02% of the time;
quarter scores are dominated by 0, 7, 3, 14, 10, 21. Every drive ends in a touchdown
(6 + a 7th/8th point or a failed try), a field goal, a safety or nothing, so building a
quarter from drives yields only reachable scores with realistic frequencies.

Pieces
* ``shares``: how a team's points split across quarters. Q2 is the biggest quarter;
  favourites front-load blowouts (Q4 share falls ~25% -> 20%), underdogs score late.
* ``drive model``: per quarter, possessions N = 2 + Binomial(3, r); each drive is
  TD / FG / safety / nothing. Probabilities scale with the team's expected points.
  Parameters are fit by maximum likelihood on real quarter scores.
* ``simulate``: draws whole games (with game-level noise so margin/total spread matches
  reality), then returns the *typical* legal game: the sampled game, with the predicted
  winner ahead and no rare scores, closest to the model's expected points.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

QUARTERS = (1, 2, 3, 4)
MAX_Q = 48  # support of a single quarter's score pmf
BASE_DRIVES = 2
EXTRA_DRIVES = 3  # N = BASE + Binomial(EXTRA, r)  ->  2..5 possessions per team-quarter
MODEL_FILE = "quarter_model.json"
TOP_MATCHES = 12
_LINE_CACHE: dict[tuple, dict[str, list[int]]] = {}


# --------------------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------------------
@dataclass
class QuarterModel:
    # per quarter: r (possession-count tilt), then drive outcome weights td / fg / safety
    r: list[float] = field(default_factory=lambda: [0.45, 0.55, 0.45, 0.50])
    p_td: list[float] = field(default_factory=lambda: [0.20, 0.25, 0.20, 0.22])
    p_fg: list[float] = field(default_factory=lambda: [0.08, 0.09, 0.08, 0.08])
    p_safety: list[float] = field(default_factory=lambda: [0.002] * 4)
    conv: list[float] = field(default_factory=lambda: [0.04, 0.93, 0.03])  # TD ends 6 / 7 / 8 points
    # Field goals are less sensitive to offensive strength than touchdowns: p_fg scales as kappa**fg_gamma
    fg_gamma: float = 1.0
    # shares: share_q = a + b * |expected margin| for the favourite and for the underdog
    fav_a: list[float] = field(default_factory=lambda: [0.224, 0.297, 0.223, 0.257])
    fav_b: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    dog_a: list[float] = field(default_factory=lambda: [0.211, 0.299, 0.226, 0.265])
    dog_b: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    # game-level noise (points) so simulated margin / total spread matches reality
    sigma_margin: float = 8.0
    sigma_total: float = 8.0
    # empirical frequencies used to reject rare scores in the displayed game
    quarter_freq: dict[str, float] = field(default_factory=dict)
    final_freq: dict[str, float] = field(default_factory=dict)
    fit_info: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "QuarterModel":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / (MODEL_FILE + ".tmp")
        tmp.write_text(json.dumps(self.to_json(), indent=1))
        tmp.replace(directory / MODEL_FILE)

    @classmethod
    def load(cls, directory: Path) -> "QuarterModel | None":
        path = directory / MODEL_FILE
        if not path.exists():
            return None
        try:
            return cls.from_json(json.loads(path.read_text()))
        except (OSError, ValueError, TypeError):
            return None


# --------------------------------------------------------------------------------------
# shares
# --------------------------------------------------------------------------------------
def quarter_shares(model: QuarterModel, abs_margin: np.ndarray | float, favourite: bool) -> np.ndarray:
    """Share of a team's points in Q1..Q4 (rows sum to 1). Shape (..., 4)."""
    m = np.clip(np.asarray(abs_margin, dtype=float), 0.0, 45.0)[..., None]
    a = np.array(model.fav_a if favourite else model.dog_a)
    b = np.array(model.fav_b if favourite else model.dog_b)
    raw = np.clip(a + b * m, 0.05, None)
    return raw / raw.sum(axis=-1, keepdims=True)


# --------------------------------------------------------------------------------------
# drive model
# --------------------------------------------------------------------------------------
def _mean_drives(r: float) -> float:
    return BASE_DRIVES + EXTRA_DRIVES * r


def drive_probs(model: QuarterModel, q: int, kappa: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-drive probability of (touchdown, field goal) at strength ``kappa``, plus safety."""
    kappa = np.asarray(kappa, dtype=float)
    return model.p_td[q] * kappa, model.p_fg[q] * kappa**model.fg_gamma, model.p_safety[q]


def _points_per_drive(model: QuarterModel, q: int, kappa: np.ndarray) -> np.ndarray:
    c6, c7, c8 = model.conv
    td, fg, sf = drive_probs(model, q, kappa)
    return td * (6 * c6 + 7 * c7 + 8 * c8) + fg * 3.0 + sf * 2.0


def kappa_for_mean(model: QuarterModel, q: int, mu: np.ndarray) -> np.ndarray:
    """Strength ``kappa`` for which the quarter's expected points equal ``mu`` (monotone -> bisection)."""
    mu = np.asarray(mu, dtype=float)
    n = _mean_drives(model.r[q])
    lo = np.full(mu.shape, 0.02)
    # outcome probabilities (td + fg + safety) must stay below 0.97
    hi = np.full(mu.shape, 8.0)
    for _ in range(40):
        mid = (lo + hi) / 2.0
        td, fg, sf = drive_probs(model, q, mid)
        too_big = (td + fg + sf > 0.97) | (n * _points_per_drive(model, q, mid) > mu)
        hi = np.where(too_big, mid, hi)
        lo = np.where(too_big, lo, mid)
    return (lo + hi) / 2.0


def _drive_pmf(model: QuarterModel, q: int, kappa: float) -> np.ndarray:
    c6, c7, c8 = model.conv
    td, fg, sf = drive_probs(model, q, kappa)
    d = np.zeros(9)
    d[6], d[7], d[8] = td * c6, td * c7, td * c8
    d[3] = fg
    d[2] = sf
    d[0] = 1.0 - d[1:].sum()
    return d


def quarter_pmf(model: QuarterModel, q: int, mu: float) -> np.ndarray:
    """P(team scores k in quarter q), k = 0..MAX_Q, given expected points ``mu``."""
    kappa = float(kappa_for_mean(model, q, np.array([mu]))[0])
    d = _drive_pmf(model, q, kappa)
    r = model.r[q]
    out = np.zeros(MAX_Q + 1)
    power = np.zeros(1)
    power[0] = 1.0
    acc: dict[int, np.ndarray] = {}
    for n in range(1, BASE_DRIVES + EXTRA_DRIVES + 1):
        power = np.convolve(power, d)
        acc[n] = power
    from math import comb

    for extra in range(EXTRA_DRIVES + 1):
        w = comb(EXTRA_DRIVES, extra) * (r**extra) * ((1 - r) ** (EXTRA_DRIVES - extra))
        pm = acc[BASE_DRIVES + extra]
        k = min(len(pm), MAX_Q + 1)
        out[:k] += w * pm[:k]
    return out / out.sum()


# --------------------------------------------------------------------------------------
# expected points helpers
# --------------------------------------------------------------------------------------
def expected_team_points(total: np.ndarray, margin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = np.asarray(total, dtype=float)
    margin = np.asarray(margin, dtype=float)
    return total / 2.0 + margin / 2.0, total / 2.0 - margin / 2.0


def expected_quarter_means(
    model: QuarterModel, home_pts: np.ndarray, away_pts: np.ndarray, margin: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Expected points per quarter for each side, shape (n, 4)."""
    margin = np.asarray(margin, dtype=float)
    home_fav = margin >= 0
    amag = np.abs(margin)
    sh_fav = quarter_shares(model, amag, True)
    sh_dog = quarter_shares(model, amag, False)
    home_share = np.where(home_fav[:, None], sh_fav, sh_dog)
    away_share = np.where(home_fav[:, None], sh_dog, sh_fav)
    return np.asarray(home_pts, dtype=float)[:, None] * home_share, np.asarray(away_pts, dtype=float)[:, None] * away_share


# --------------------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------------------
def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - np.max(x))
    return z / z.sum()


def _unpack(theta: np.ndarray, base: QuarterModel) -> QuarterModel:
    m = QuarterModel.from_json(base.to_json())
    for q in range(4):
        r_logit, w_td, w_fg, w_sf = theta[4 * q : 4 * q + 4]
        m.r[q] = float(1.0 / (1.0 + np.exp(-r_logit)))
        p = _softmax(np.array([w_td, w_fg, w_sf, 0.0]))
        m.p_td[q], m.p_fg[q], m.p_safety[q] = float(p[0]), float(p[1]), float(p[2])
    c = _softmax(np.array([theta[16], theta[17], 0.0]))  # 6 / 8 / 7 points
    m.conv = [float(c[0]), float(1.0 - c[0] - c[1]), float(c[1])]
    m.fg_gamma = float(-0.5 + 2.0 / (1.0 + np.exp(-theta[18])))  # kept in (-0.5, 1.5)
    return m


def _pack(m: QuarterModel) -> np.ndarray:
    theta = []
    for q in range(4):
        r = np.clip(m.r[q], 1e-3, 1 - 1e-3)
        none = max(1.0 - m.p_td[q] - m.p_fg[q] - m.p_safety[q], 1e-3)
        theta += [np.log(r / (1 - r)), np.log(m.p_td[q] / none), np.log(m.p_fg[q] / none), np.log(m.p_safety[q] / none)]
    theta += [np.log(m.conv[0] / m.conv[1]), np.log(m.conv[2] / m.conv[1])]
    g = np.clip((m.fg_gamma + 0.5) / 2.0, 1e-3, 1 - 1e-3)
    theta.append(float(np.log(g / (1 - g))))
    return np.array(theta)


def fit_shares(model: QuarterModel, scores: pd.DataFrame) -> QuarterModel:
    """OLS of each team's quarter share on |expected margin|, separately for favourite and underdog."""
    m = QuarterModel.from_json(model.to_json())
    margin = scores["exp_margin"].to_numpy(dtype=float)
    home_fav = margin >= 0
    amag = np.abs(margin)
    for fav in (True, False):
        side_pts = []
        for side in ("home", "away"):
            side_pts.append(scores[[f"{side}_q{q}" for q in QUARTERS]].to_numpy(dtype=float))
        home, away = side_pts
        picked = np.where((home_fav if fav else ~home_fav)[:, None], home, away)
        total = picked.sum(axis=1)
        ok = total > 0
        share = picked[ok] / total[ok][:, None]
        x = np.c_[np.ones(ok.sum()), amag[ok]]
        coefs = np.linalg.lstsq(x, share, rcond=None)[0]  # (2, 4)
        a, b = coefs[0].tolist(), coefs[1].tolist()
        if fav:
            m.fav_a, m.fav_b = a, b
        else:
            m.dog_a, m.dog_b = a, b
    return m


def _grouped_counts(scores: pd.DataFrame, model: QuarterModel) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per quarter: (rounded mu bins, matrix of observed-score counts per bin)."""
    margin = scores["exp_margin"].to_numpy(dtype=float)
    home_pts, away_pts = expected_team_points(scores["exp_total"].to_numpy(dtype=float), margin)
    hq, aq = expected_quarter_means(model, home_pts, away_pts, margin)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for qi, q in enumerate(QUARTERS):
        mu = np.r_[hq[:, qi], aq[:, qi]]
        obs = np.r_[scores[f"home_q{q}"].to_numpy(), scores[f"away_q{q}"].to_numpy()].astype(int)
        obs = np.clip(obs, 0, MAX_Q)
        bins = np.round(mu * 2) / 2.0  # 0.5-point mu bins keep the pmf cache small
        frame = pd.DataFrame({"bin": bins, "obs": obs})
        ct = frame.groupby(["bin", "obs"]).size().unstack(fill_value=0)
        full = np.zeros((len(ct), MAX_Q + 1))
        full[:, ct.columns.to_numpy()] = ct.to_numpy()
        out[qi] = (ct.index.to_numpy(dtype=float), full)
    return out


def quarter_nll(model: QuarterModel, grouped: dict[int, tuple[np.ndarray, np.ndarray]]) -> tuple[float, int]:
    total, n = 0.0, 0
    for qi, (mus, counts) in grouped.items():
        for mu, row in zip(mus, counts):
            pmf = quarter_pmf(model, qi, max(mu, 0.5))
            total -= float((row * np.log(np.clip(pmf, 1e-12, None))).sum())
            n += int(row.sum())
    return total, n


def fit(scores: pd.DataFrame, max_iter: int = 60) -> QuarterModel:
    """Fit shares, drive-model parameters and game-noise from completed games.

    ``scores`` needs home_q1..4, away_q1..4 (regulation quarters), plus the pre-game
    expectations ``exp_margin`` (home minus away, e.g. -spread) and ``exp_total`` (over/under).
    """
    model = fit_shares(QuarterModel(), scores)
    grouped = _grouped_counts(scores, model)

    def objective(theta: np.ndarray) -> float:
        nll, n = quarter_nll(_unpack(theta, model), grouped)
        return nll / n

    start = _pack(model)
    res = minimize(objective, start, method="L-BFGS-B", options={"maxiter": max_iter, "maxfun": 4000})
    fitted = _unpack(res.x, model)
    fitted.fit_info = {"nll_per_team_quarter": float(res.fun), "n_games": int(len(scores)), "converged": bool(res.success)}
    fitted = _fit_noise(fitted, scores)
    return _fit_rarity(fitted, scores)


def _regulation_final(scores: pd.DataFrame) -> np.ndarray:
    return np.r_[
        scores[[f"home_q{q}" for q in QUARTERS]].sum(axis=1).to_numpy(),
        scores[[f"away_q{q}" for q in QUARTERS]].sum(axis=1).to_numpy(),
    ].astype(int)


def _fit_rarity(model: QuarterModel, scores: pd.DataFrame) -> QuarterModel:
    quarters = np.r_[
        scores[[f"home_q{q}" for q in QUARTERS]].to_numpy().ravel(), scores[[f"away_q{q}" for q in QUARTERS]].to_numpy().ravel()
    ].astype(int)
    qv, qc = np.unique(quarters, return_counts=True)
    fv, fc = np.unique(_regulation_final(scores), return_counts=True)
    model.quarter_freq = {str(int(k)): float(c / qc.sum()) for k, c in zip(qv, qc)}
    model.final_freq = {str(int(k)): float(c / fc.sum()) for k, c in zip(fv, fc)}
    return model


def _fit_noise(model: QuarterModel, scores: pd.DataFrame) -> QuarterModel:
    """Pick game-level latent sd so simulated margin/total spread equals the real residual spread."""
    margin_res = (
        scores[[f"home_q{q}" for q in QUARTERS]].sum(axis=1) - scores[[f"away_q{q}" for q in QUARTERS]].sum(axis=1) - scores["exp_margin"]
    ).to_numpy(dtype=float)
    total_res = (
        scores[[f"home_q{q}" for q in QUARTERS]].sum(axis=1) + scores[[f"away_q{q}" for q in QUARTERS]].sum(axis=1) - scores["exp_total"]
    ).to_numpy(dtype=float)
    target_m, target_t = float(np.nanstd(margin_res)), float(np.nanstd(total_res))
    model.sigma_margin = model.sigma_total = 0.0
    sample = scores.sample(min(len(scores), 1500), random_state=7)
    sim = simulate_regulation(
        model,
        sample["exp_total"].to_numpy(dtype=float),
        sample["exp_margin"].to_numpy(dtype=float),
        n_sims=40,
        seed=11,
    )
    home = sim["home"].sum(axis=2)
    away = sim["away"].sum(axis=2)
    # spread of the simulated result *around its own pre-game expectation* (not across games)
    intrinsic_m = float((home - away - sample["exp_margin"].to_numpy(dtype=float)[:, None]).std())
    intrinsic_t = float((home + away - sample["exp_total"].to_numpy(dtype=float)[:, None]).std())
    model.sigma_margin = float(np.sqrt(max(target_m**2 - intrinsic_m**2, 0.0)))
    model.sigma_total = float(np.sqrt(max(target_t**2 - intrinsic_t**2, 0.0)))
    model.fit_info.update(
        {
            "target_sd_margin": target_m,
            "target_sd_total": target_t,
            "intrinsic_sd_margin": intrinsic_m,
            "intrinsic_sd_total": intrinsic_t,
        }
    )
    return model


# --------------------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------------------
def _sample_side(model: QuarterModel, mu: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample quarter scores for expected means ``mu`` of shape (..., 4) -> same shape (ints)."""
    out = np.zeros(mu.shape, dtype=np.int16)
    c = np.array(model.conv)
    for qi in range(4):
        kappa = kappa_for_mean(model, qi, mu[..., qi])
        td, fg, sf = drive_probs(model, qi, kappa)
        n = BASE_DRIVES + rng.binomial(EXTRA_DRIVES, model.r[qi], size=kappa.shape)
        pts = np.zeros(kappa.shape, dtype=np.int16)
        for drive in range(BASE_DRIVES + EXTRA_DRIVES):
            active = drive < n
            u = rng.random(kappa.shape)
            is_td = u < td
            is_fg = (u >= td) & (u < td + fg)
            is_sf = (u >= td + fg) & (u < td + fg + sf)
            conv_u = rng.random(kappa.shape)
            td_pts = np.where(conv_u < c[0], 6, np.where(conv_u < c[0] + c[1], 7, 8))
            gain = np.where(is_td, td_pts, 0) + np.where(is_fg, 3, 0) + np.where(is_sf, 2, 0)
            pts += np.where(active, gain, 0).astype(np.int16)
        out[..., qi] = pts
    return out


def simulate_regulation(
    model: QuarterModel,
    total: np.ndarray,
    margin: np.ndarray,
    n_sims: int = 1500,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Simulate regulation for ``len(total)`` games. Arrays are (games, sims, 4)."""
    total = np.asarray(total, dtype=float)
    margin = np.asarray(margin, dtype=float)
    rng = np.random.default_rng(seed)
    g = len(total)
    d_margin = rng.normal(0.0, model.sigma_margin, size=(g, n_sims))
    d_total = rng.normal(0.0, model.sigma_total, size=(g, n_sims))
    home_pts = np.clip(total[:, None] / 2 + margin[:, None] / 2 + d_total / 2 + d_margin / 2, 2.0, None)
    away_pts = np.clip(total[:, None] / 2 - margin[:, None] / 2 + d_total / 2 - d_margin / 2, 2.0, None)
    # favourite / underdog identity and the margin that drives shares come from the pre-game view
    amag = np.abs(margin)[:, None] * np.ones((1, n_sims))
    home_fav = (margin >= 0)[:, None] * np.ones((1, n_sims), dtype=bool)
    sh_fav = quarter_shares(model, amag, True)
    sh_dog = quarter_shares(model, amag, False)
    home_share = np.where(home_fav[..., None], sh_fav, sh_dog)
    away_share = np.where(home_fav[..., None], sh_dog, sh_fav)
    home = _sample_side(model, home_pts[..., None] * home_share, rng)
    away = _sample_side(model, away_pts[..., None] * away_share, rng)
    return {"home": home, "away": away}


def _seed_for(home: str, away: str, margin: float, total: float) -> int:
    key = f"{home}|{away}|{margin:.2f}|{total:.2f}".encode()
    return zlib.crc32(key)


def typical_line_score(
    model: QuarterModel,
    home_team: str,
    away_team: str,
    total: float,
    margin: float,
    n_sims: int = 1500,
) -> dict[str, list[int]]:
    """The representative regulation game: predicted winner ahead, no rare scores, closest to expectation."""
    seed = _seed_for(home_team, away_team, margin, total)
    key = (id(model), seed, n_sims)
    if key in _LINE_CACHE:
        return _LINE_CACHE[key]
    sim = simulate_regulation(model, np.array([total]), np.array([margin]), n_sims=n_sims, seed=seed)
    home = sim["home"][0].astype(int)  # (sims, 4)
    away = sim["away"][0].astype(int)
    hf, af = home.sum(axis=1), away.sum(axis=1)
    home_pts, away_pts = expected_team_points(np.array([total]), np.array([margin]))
    # Match on the final score only. Matching quarter means too would pick 7-7-7-7 for everyone,
    # whereas real games are lumpy (27% of team-quarters are 0, 26% are 7, 11% are 3).
    dist = (hf - home_pts[0]) ** 2 + (af - away_pts[0]) ** 2
    winner_ok = (hf > af) if margin >= 0 else (af > hf)
    rare_q = {int(k) for k, v in model.quarter_freq.items() if v < 0.0025}
    rare_f = {int(k) for k, v in model.final_freq.items() if v < 0.0012} | {1, 2, 4, 5}
    ok = winner_ok.copy()
    if rare_q:
        rq = np.array(sorted(rare_q))
        ok &= ~(np.isin(home, rq).any(axis=1) | np.isin(away, rq).any(axis=1))
    ok &= ~(np.isin(hf, list(rare_f)) | np.isin(af, list(rare_f)))
    if not ok.any():  # relax rarity before ever returning a tie / wrong winner
        ok = winner_ok & ~(np.isin(hf, [1, 2, 4, 5]) | np.isin(af, [1, 2, 4, 5]))
    if not ok.any():
        ok = winner_ok
    if not ok.any():
        ok = hf != af
    if not ok.any():
        ok = np.ones_like(ok)
    scored = np.where(ok, dist, np.inf)
    # Among the closest matches take a seeded pick, so quarter patterns vary like real games
    # while the same matchup always shows the same line score.
    best = np.argsort(scored, kind="stable")[: min(TOP_MATCHES, int(ok.sum()))]
    pick = int(best[np.random.default_rng(seed + 1).integers(len(best))])
    result = {"home": home[pick].tolist(), "away": away[pick].tolist()}
    if len(_LINE_CACHE) > 5000:
        _LINE_CACHE.clear()
    _LINE_CACHE[key] = result
    return result
