# Weekly incremental training (no holdout)

## Goal

The live board (`src/` + `app/` + `web/`) must stop splitting training on a holdout season. The 2025 default and the Train-page holdout field go away. After a one-time catch-up fit on every completed FBS game, later weeks only train a correction layer on that week’s new results. The API polls for finals and runs that update when a board week is fully complete.

## Decisions (locked)

- One catch-up fit on **all completed FBS-vs-FBS games**, then freeze those trees.
- Each later week trains **only on that week**, not a full refit.
- A week is done when **every FBS-vs-FBS game** on that CFBD week has a final. Non-FBS leftovers do not block.
- Residual target is **actual home margin minus the frozen base prediction** (logged pre-game pick when present).
- Production score is `base_margin + week_layer`.

## Components

### Catch-up (`src.train.train`)

- No `holdout_season` argument, CLI flag, job body field, or UI input.
- Fit margin HGB, market HGB, ridge, win classifier, and ATS model on every completed FBS row.
- Walk-forward by season stays **evaluation-only** (does not change the saved trees).
- Record `models/weekly.json` `base_weeks`: fully complete weeks at catch-up time.
- Metrics JSON has no `holdout_season`. Dashboard uses train-game counts plus latest-week residual report when present.

### Week layer (`src.weekly`)

- `completed_weeks(frame)` → `(season, week)` pairs where all FBS-vs-FBS games are final.
- `fit_week_layer` fits a regularized ridge on that week’s residuals only. Minimum 8 games. Replaces the previous layer (does not stack).
- Idempotent: skip if `(season, week)` is already in `base_weeks` or `applied_weeks`.
- Persist `models/week_layer.joblib` and append the week report to `weekly.json`.
- `log_predictions` is append-only by `game_id` for unplayed games; learning prefers the logged `pred_margin`.

### Scoring (`src.simulate.score_slate`)

- Always compute frozen-base `pred_margin_raw`.
- Add the week layer when the artifact exists.
- Recompute points and spread edge from the adjusted margin.
- Log predictions for uncompleted games with a `game_id`.

### Auto tick

- FastAPI lifespan thread (off when `WEEKLY_AUTOTRAIN=0`).
- Interval: `WEEKLY_POLL_SECONDS` (default 900).
- Each tick: refresh live-season `/games` scores (CFBD), rebuild the feature frame, then:
  - If base artifacts are missing → start a `train` job (catch-up).
  - Else fit the week layer for the oldest newly completed week not already applied.
- Manual Collect / Train / pipeline remain. Train means catch-up rebuild only.

### UI

- Remove the holdout field and all “Holdout defaults to 2025” copy.
- Train page explains automatic weekly updates after FBS finals land.
- Metrics labels: latest-week (or walk-forward) accuracy/MAE, not “Holdout winner acc”.

## Error handling

- Score refresh failures do not abort learning on the current SQLite snapshot.
- JobConflict: skip starting a second catch-up.
- Incomplete week or `n < 8`: no layer write.
- Missing week layer: score with base only.

## Tests (seams)

- Week completeness (FBS-only rule).
- Layer trained on one week moves predictions toward that week’s residuals; second tick does not stack.
- Catch-up weeks are skipped by the layer.
- `train()` / jobs / API / Train page carry no holdout season.
- `score_slate` applies the layer when present and leaves `_models()` fixtures unchanged.
