# CLAUDE.md

Guidance for working in this college football prediction repository.

## Overview

Python package `cfb_model` forecasts FBS home point margin from College Football Data API v2, then derives win probability and ATS edge. Closing spreads are evaluation-only (never training features).

## Commands

```bash
python -m pip install -r requirements.txt
python -m cfb_model ingest --start-year 2016 --end-year 2026   # Pro / 75k default
python -m cfb_model train
python -m cfb_model predict --year 2026 --week 5
python -m pytest tests
```

API auth (do not hardcode keys):

```python
import os, cfbd
configuration = cfbd.Configuration(access_token=os.environ["CFBD_API_KEY"])
```

On ingest, `GET /info` is used to detect tier features (`adjusted_metrics`, `weather`). WEPA and CFBD weather are skipped when unavailable. Open-Meteo is the weather fallback (`ingest --weather`).

## Data

- **Normalized warehouse:** `data/cfb.db` (gitignored), schema in `cfb_model/schema.sql`. `games.id` is the join key.
- **Legacy archive:** `collegeFootball.db` — year-sharded tables (`Games2024`, `bets24`, …). Games tables omitted `id`; `cfb_model.migrate` reconstructs it from the bets row stream.
- **Cache:** `data/cache/` JSON responses. A full 2016–2026 ingest is ~1–2k calls vs a 75k monthly Pro quota.

### Feature leakage rules

- Rolling PPA / success / havoc / scoring use only games with `start_date` strictly before kickoff (`shift(1)`).
- SP+, CORE, and WEPA snapshots are joined as **prior season** only.
- Pregame Elo on `/games` is safe; weekly `/ratings/elo` is ingested by default (disable with `ingest --lite`).
- Tests in `tests/test_leakage.py` guard the as-of join.

## Modeling

- Target: `home_margin = home_points - away_points`
- Models: RidgeCV (interpretable) + XGBoost. Champion is chosen from walk-forward MAE on folds with ≥1,000 training games.
- WP: `Φ(pred_margin / σ)` with σ from residuals.
- Evaluation vs pregame Elo (`elo_diff / 28`) and consensus closing spread.
- Artifacts: `models/ridge.joblib`, `models/xgb.json`, `models/metrics.json`.

## Layout

```
cfb_model/           # client, ingest, features, models, evaluate, CLI
tests/
data_collection_preparation/   # archived notebooks
modelling/evaluate.ipynb       # thin visual wrapper
```

Prefer adding Python modules over new notebooks. Keep notebooks as callers of `cfb_model`.
