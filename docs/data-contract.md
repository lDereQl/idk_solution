# Data contract (modelling team -> backend)

All times UTC. Energy in kWh. Files go under `data/` (or `DATA_DIR`).

## `sites.csv`
`site_id, name, country, level, role`

## `history/<site_id>.parquet`
| column | type | note |
|---|---|---|
| timestamp_utc | datetime64[ns, UTC] | start of hour |
| heat_kwh | float | actual demand |

## `models/<model_id>/forecasts.parquet`
| column | type | note |
|---|---|---|
| site_id | str | must exist in sites.csv |
| issue_time_utc | datetime64[ns, UTC] | forecast time t |
| horizon_h | int | 1..72, all 72 present per (site, issue) |
| target_time_utc | datetime64[ns, UTC] | = issue_time_utc + (horizon_h - 1) h |
| p10, p50, p90 | float | kWh, p10 <= p50 <= p90 (p10/p90 may be NaN for point baselines) |
| p_peak | float | 0..1, may be NaN |

Predictions must be in physical kWh (already multiplied back by the site's C_peak).

## `models/<model_id>/meta.json`
`{"name": "...", "description": "..."}`

Check an export before use: `python scripts/validate_data.py [data_dir]`
