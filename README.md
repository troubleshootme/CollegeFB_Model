# College Football Prediction Model

Leakage-safe, margin-first FBS predictions on the [CollegeFootballData](https://collegefootballdata.com/) API v2.

The model forecasts **home point margin**, then derives win probability, a home spread, and an edge versus the closing line. Closing spreads are used only for evaluation — they are never training features.

## Quick start

```bash
python -m pip install -r requirements.txt
cp .env.example .env   # then set CFBD_API_KEY (Pro / 75k calls per month)

# Recommended — full CFBD backfill (week-level PPA/havoc/box/Elo, WEPA, weather)
python -m cfb_model ingest --start-year 2016 --end-year 2026

# Fallback if you only have the bundled 2020–2024 archive
python -m cfb_model migrate

python -m cfb_model train
python -m cfb_model predict --year 2026 --week 5
python -m pytest tests
```

Get an API key at https://collegefootballdata.com/key. Auth uses the v2 SDK:

```python
import os, cfbd
configuration = cfbd.Configuration(access_token=os.environ["CFBD_API_KEY"])
```

Never commit keys. Legacy notebooks previously stored a token in source; that has been removed.

## Design

| Choice | Why |
|---|---|
| Predict margin, not winner | More signal; WP and ATS drop out of the residual scale |
| As-of features only | Same-game box scores and end-of-season SP+/FPI/CORE are excluded |
| Priors for weeks 0–4 | Talent, recruiting, returning production, prior-season ratings |
| James–Stein shrinkage | `w = n/(n+4)` blends in-season form toward last season |
| Closing line is the baseline | Opening spread is optional (`train --blend`); close is eval-only |
| Opponent-adjusted efficiency | PPA, success rate, explosiveness, havoc (after `ingest`); WEPA/CORE when the key’s tier allows |

```text
CFBD v2 ──► data/cfb.db ──► as-of features ──► Ridge + XGBoost
Open-Meteo ─┘                      │
                                   ├─► walk-forward vs Elo / close
                                   └─► python -m cfb_model predict
```

## Commands

| Command | Purpose |
|---|---|
| `python -m cfb_model ingest` | Full CFBD pull for a 75k/month Pro key (week-level havoc/box/Elo, WEPA, weather). `--lite` for year-level only; `--no-weather` to skip weather |
| `python -m cfb_model migrate` | Load `collegeFootball.db` into the normalized schema |
| `python -m cfb_model features` | Write `data/features.csv` |
| `python -m cfb_model train` | Walk-forward train/eval; write `models/`; update learning state |
| `python -m cfb_model evaluate` | Same as train, prints fold metrics |
| `python -m cfb_model learn` | Score finalized games; EWMA bias / sigma / injury scale |
| `python -m cfb_model injuries` | ESPN injury snapshot (no CFBD quota) |
| `python -m cfb_model profile --qb NAME --year YYYY` | Print a QB identity card |
| `python -m cfb_model predict --year 2026 --week N` | Weekly CSV of margin, WP, edge, confidence |

On ingest start the client calls `GET /info` and skips WEPA / CFBD weather when the key’s tier does not include those features.

## Warehouse

Normalized SQLite at `data/cfb.db` (gitignored). Schema: [`cfb_model/schema.sql`](cfb_model/schema.sql).

Important tables: `games` (**keeps `id`**), `lines` (one row per provider), `ppa_games`, `advanced_game_stats`, `talent`, `recruiting_teams`, `returning_production`, `core_ratings`, `sp_ratings`, `fpi_ratings` (prior-season only in features), `player_season_stats`, `player_ppa`, `transfer_portal`, `injury_reports` (as-of snapshots), `venues`, `teams`, `coaches_seasons`, `weather`.

`collegeFootball.db` is the old year-sharded archive (`Games2024`, `bets24`, …). It dropped `game_id` on games and flattened lines incorrectly. `migrate` reconstructs ids from the bets stream and parses text stats (`3-7`, `35:42`).

## Current holdout (migrated 2020–2024, no PPA yet)

Walk-forward on 3,140 FBS vs FBS completed games after `migrate` (Elo + talent + rolling scoring + venue). After `ingest` on a Pro key the same pipeline adds garbage-time-excluded PPA, success/explosiveness/havoc, returning production, CORE, and WEPA.

| Metric | Model | Pregame Elo | Closing spread |
|---|---:|---:|---:|
| Margin MAE | 13.44 | 13.47 | 12.20 |
| Moneyline accuracy | 70.1% | — | — |
| ATS hit rate | 48.5% | — | — |

Week 5+ MAE (13.19) is better than weeks 1–4 (14.00), matching CFBD’s advice that in-season identities stabilize later. The market still wins on MAE — expected until opponent-adjusted PPA is ingested. Artifacts: [`models/metrics.json`](models/metrics.json), `models/ridge.joblib`, `models/xgb.json`.

Plots and coefficients: [`modelling/evaluate.ipynb`](modelling/evaluate.ipynb).

## Project layout

```
cfb_model/          # package (client, ingest, features, models, CLI)
tests/              # leakage, parse, flatten, train smoke tests
data_collection_preparation/   # archived notebooks (use the CLI instead)
modelling/model_1.0            # unfinished linear POC
```

## Notes and later work

Injuries are snapshotted from ESPN (CFBD has no injury endpoint). Each ingest stores an as-of report and joins the latest snapshot *before kickoff*. Stale leftover rows are dropped. Quarterbacks are first-class profiles (like coaches): the model follows the passer across schools via the transfer portal, weights QB availability above every other position, and treats dual-threat QBs as a second offense that can punch above roster talent / FPI. `python -m cfb_model learn` scores published cards against finals and updates bias, residual scale, and the injury prior.

Flight arrival times, hedging, and a public dashboard are still out of scope.

Original research pointers:

- [Forecasting college football game outcomes using modern modeling techniques](https://www.researchgate.net/publication/338728435_Forecasting_college_football_game_outcomes_using_modern_modeling_techniques)
- [CFBD modeling tips](https://radsportsanalytics.com/blog/college-football-modeling-tips/)
- [Model Training Pack → predictions](https://blog.collegefootballdata.com/model-training-pack-how-to-make-predictions/)
