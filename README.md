# Heat demand forecast backend

Read-only API serving precomputed forecasts from the modelling team. No training or inference here.

## Run
    make run          # from a fresh clone: builds and starts everything in Docker, waits until it answers, opens the presentation

Needs Docker with Compose v2 and make. The first run builds the image and takes about a minute. The presentation opens at http://localhost:8000/demo (API docs at /docs). `make down` stops it, `make logs` follows the output, `PORT=8001 make run` uses another port.

Without Docker (needs uv):

    make install      # .venv + requirements
    make data         # synthetic data, same format as real
    make api-dev      # builds the page, uvicorn with reload, docs at /docs
    make test

`make help` lists every target.

## Presentation (`web/`)
    .venv/bin/python web/build.py      # from the repo root, same as `make web-build`

Inlines the files in `web/in/` (demo data, pipes at KTE, network image, backdrop) into `web/template.html` and writes `web/out/kaunas-heat-forecast-demo.html`, one self-contained file. The API serves it at `/demo`.

Served over http the page calls the API on its own origin. Opened from disk it calls `http://localhost:8000` (the default port), which needs `DEV_CORS=1` on the API (development only, set by `make api-dev`). If the API cannot be reached the page uses the inlined data; the pill next to the "Synthetic demo data" badge says which source is in use.

## Data layout (`data/`, git-ignored, override with `DATA_DIR`)
- `sites.csv` - site_id, name, country, level, role
- `history/<site_id>.parquet` - timestamp_utc, heat_kwh
- `models/<model_id>/meta.json` + `forecasts.parquet` - site_id, issue_time_utc, horizon_h, target_time_utc, p10, p50, p90, p_peak

New model = new folder under `models/`. No code change.

## Endpoints
`/sites`, `/models`, `/sites/{id}/history`, `/sites/{id}/issues?model=`, `/sites/{id}/forecast?model=&issue=`, `/sites/{id}/metrics?model=`, `/demo`

NaN is sent as `null` (`p10`, `p90`, `p_peak` of point baselines, gaps in history). `/metrics` compares only hours where both the forecast median and the actual value exist.

## Docker
    make up      # or: docker compose up --build. API on http://localhost:8000/docs, presentation at /demo
Mounts `./data`; if it has no `sites.csv`, stub data is generated on start.

Validate an export: `python scripts/validate_data.py [data_dir]`.
