# Heat demand forecast backend

Read-only API serving precomputed forecasts from the modelling team. No training or inference here.

## Run
    make run          # from a fresh clone: builds and starts everything in Docker, waits until it answers, opens the presentation

Needs Docker with Compose v2 and make. The first run builds the image and takes about a minute. The presentation opens at http://localhost:8000/demo (API docs at /docs). `make down` stops it, `make logs` follows the output, `PORT=8001 make run` uses another port. It serves the committed demo data (`demo_data/`, see below); `DATA_SRC=./data make run` serves your own exports from `data/` instead.

Without Docker (needs uv):

    make install      # .venv + requirements
    make data         # synthetic stub data in data/ (the tests use it)
    DATA_DIR=demo_data make api-dev   # the demo data; without DATA_DIR it serves data/. Builds the page, uvicorn with reload, docs at /docs
    make test

`make help` lists every target.

## Presentation (`web/`)
    .venv/bin/python web/build.py      # from the repo root, same as `make web-build`

Inlines the files in `web/in/` (demo data, pipes at KTE, network image, backdrop) into `web/template.html` and writes `web/out/kaunas-heat-forecast-demo.html`, one self-contained file. The API serves it at `/demo`.

Served over http the page calls the API on its own origin. Opened from disk it calls `http://localhost:8000` (the default port), which needs `DEV_CORS=1` on the API (development only, set by `make api-dev`). If the API cannot be reached the page uses the inlined data; the pill next to the "Simulated Kaunas demand" badge says which source is in use. After changing `demo_data/`, refresh the inlined data with `python scripts/snapshot_demo.py` and rebuild the page.

## Data layout (`data/`, git-ignored, override with `DATA_DIR`)
- `sites.csv` - site_id, name, country, level, role
- `history/<site_id>.parquet` - timestamp_utc, heat_kwh
- `models/<model_id>/meta.json` + `forecasts.parquet` - site_id, issue_time_utc, horizon_h, target_time_utc, p10, p50, p90, p_peak

New model = new folder under `models/`. No code change.

## Demo data (`demo_data/`, committed)
Forecasts of three models for Kaunas, 10 daily forecasts (1–10 January 2022, 06:00 UTC, with a cold snap on 11–12 January), in the layout above. **The demand is simulated** (When2Heat Lithuania profile scaled to Kauno energija's 1,414 GWh a year); the weather is real (Open-Meteo archived forecasts); the forecasts are real model output.

| Model | Mean absolute error, % of mean demand | P10–P90 coverage (target 80 %) |
|---|---|---|
| `baseline_24h` – same hour of the latest known day | 8.4 % | – |
| `lgbm_v3` – v3 as trained on Danish and German networks, no Kaunas data | 17.4 % | 10.7 % |
| `lgbm_v3_kaunas` – the same model fitted on the simulated Kaunas series 2008–2020, calibrated on 2021 | 3.1 % | 80.3 % |

Read this carefully: v3 as trained abroad does **not** transfer to the simulated Kaunas series (over January–March 2022: 16.5 % vs 7.5 % for the simple rule; it over-predicts, mostly at night, because the simulated daily profile has a deep night dip that real networks do not). The Kaunas-fitted model is the plan's step 1 (retrain on local data) run on simulated data, so its error says nothing about real Kaunas. It needs Ignitis' real hourly demand. The page opens on `lgbm_v3_kaunas`; the model switch shows the other two. `p_peak` is the chance that an hour exceeds the 90th percentile of 2021 demand, from a normal distribution through the P10–P90 range.

Regenerate (needs clever-energy-data's output, lightgbm and joblib):

    python scripts/export_demo_data.py <clever-energy-data dir>   # writes demo_data/
    python scripts/validate_data.py demo_data
    python scripts/snapshot_demo.py                                # refreshes web/in/kaunas_demo_data.json

## Model training (`weights/`)
Copy the feature tables from clever-energy-data (`bronderslev`, `sonderborg`, `flensburg` .parquet) to `data/features/`, then from the repo root (needs lightgbm and joblib):

    python weights/preparemodels_v3.py

Writes `models/lightgbm_{p10,p50,p90}_v3.pkl` and `models/lightgbm_v3_meta.json` and prints the 2019 test results. The trained files are committed in `weights/` under the same names. v3 replaces v2 (`preparemodels.py`, `weights/lightgbm_*_tuned.pkl`): v2 scaled the target by site but read past demand in raw kWh, so on the unseen city it lost to "same hour yesterday" (18.1 % vs 11.9 % mean absolute error). v3 scales the target and past demand per site, drops inputs derived from the target, and calibrates P10–P90 on a separate year:

| 2019, mean absolute error | Flensburg (unseen city) | Sønderborg | Brønderslev A |
|---|---|---|---|
| Same hour yesterday | 11.9 % | 10.7 % | 12.2 % |
| v3 | 8.2 % | 5.0 % | 7.6 % |
| v3 P10–P90 coverage (target 80 %) | 64.9 % | 87.7 % | 72.8 % |

## Endpoints
`/sites`, `/models`, `/sites/{id}/history`, `/sites/{id}/issues?model=`, `/sites/{id}/forecast?model=&issue=`, `/sites/{id}/metrics?model=`, `/demo`

NaN is sent as `null` (`p10`, `p90`, `p_peak` of point baselines, gaps in history). `/metrics` compares only hours where both the forecast median and the actual value exist.

## Docker
    make up      # or: docker compose up --build. API on http://localhost:8000/docs, presentation at /demo
Mounts `./data`; if it has no `sites.csv`, stub data is generated on start.

See `docs/data-contract.md`. Validate an export: `python scripts/validate_data.py [data_dir]`.
