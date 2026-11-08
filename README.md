# Heat demand forecast backend

Read-only API serving precomputed forecasts from the modelling team. No training or inference here.

## Run
    python -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python scripts/make_stub_data.py     # synthetic data, same format as real
    .venv/bin/uvicorn app.main:app --reload        # docs at /docs
    .venv/bin/pytest

## Data layout (`data/`, git-ignored, override with `DATA_DIR`)
- `sites.csv` - site_id, name, country, level, role
- `history/<site_id>.parquet` - timestamp_utc, heat_kwh
- `models/<model_id>/meta.json` + `forecasts.parquet` - site_id, issue_time_utc, horizon_h, target_time_utc, p10, p50, p90, p_peak

New model = new folder under `models/`. No code change.

## Endpoints
`/sites`, `/models`, `/sites/{id}/history`, `/sites/{id}/issues?model=`, `/sites/{id}/forecast?model=&issue=`, `/sites/{id}/metrics?model=`

## Docker
    docker compose up --build      # API on http://localhost:8000/docs
Mounts `./data`; if it has no `sites.csv`, stub data is generated on start.

See `docs/data-contract.md`. Validate an export: `python scripts/validate_data.py [data_dir]`.
