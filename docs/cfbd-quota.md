# CFBD quota: train owns the key, serve does not

CollegeFootballData (CFBD) is a **shared monthly pool** (CFB + CBB, Tier 3 / ~75k). Every live call from the Matchup Desk API process steals from model training.

## Hard rule

| Process | May call CFBD? |
|---|---|
| API / uvicorn (`app.main`) | **No.** Board, predictions, matchups, `/api/teams` autocomplete, logos, replay. |
| `src.schedule_sync` (live scores) | **No.** Sibling [cfbschedule](http://127.0.0.1:47392) only. |
| `run_pipeline.py`, `python -m src.collect`, `python -m cfb_model ingest` | **Yes**, with `CFBD_ALLOW_CALLS=1`. |
| `/api/jobs/collect` and `/api/jobs/pipeline` | **Yes** — the job subprocess sets the flag. The parent API process does not. |

`src/cfbd_client.py` and `cfb_model.client.CfbdClient` refuse HTTP unless `CFBD_ALLOW_CALLS` is truthy (`1` / `true` / `yes` / `on`). Empty team/logo cache returns empty data or an error. It does **not** fall back to `/teams`.

## Teams and logos

Collect writes `fbs_teams` / `fbsTeams` in `collegeFootball.db` and a sidecar `data/fbs_teams_cache.json`. Autocomplete (`GET /api/teams`) and board logos read **only** that cache.

`refresh_if_empty` used to call CFBD `/teams/fbs` from the API lifespan and every `/api/teams` request whenever logos were missing. That path is gone on serve. A train job may still pass `refresh_if_empty=True` **and** `CFBD_ALLOW_CALLS=1`.

## Lines

Prefer CFBD `GET /lines` (integer `gameId`, joins `games.game_id`). cfbschedule betting lines use UUID ids and break those joins — they are a **train-time fallback** only, remapped onto numeric ids via season/week/home/away. Unmapped UUID rows are dropped.

## remainingCalls = 0

Train jobs probe `/info` (or the v2 SDK equivalent) before a pull. If `remainingCalls` is 0, collect/ingest abort instead of retrying 429s. Serve never needs this: it cannot call CFBD.

## Cron

See [`ops/crontab`](../ops/crontab). Cron silently skips a last line that is missing its trailing newline — that is why Monday retrains did not run. The file in this repo is newline-terminated.

Recommended split:

1. **Scores** every 15 minutes: `python -m src.schedule_sync` (cfbschedule, no CFBD).
2. **Retrain** Monday after the week is final: `CFBD_ALLOW_CALLS=1 python run_pipeline.py`.

Do not cron CFBD `/games` from the API host to refresh the board.

## lock_predictions

There is no `lock_predictions` route in this repository. If a sibling service adds one, it must score from local models + SQLite like `/api/predictions` and must not import `src.cfbd_client.get_json`.
