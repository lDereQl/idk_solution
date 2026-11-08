"""Read-only access to precomputed data. Layout:

data/sites.csv                         site_id, name, country, level, role
data/history/<site_id>.parquet         timestamp_utc, heat_kwh
data/models/<model_id>/meta.json       {"name": ..., "description": ...}
data/models/<model_id>/forecasts.parquet
    site_id, issue_time_utc, horizon_h, target_time_utc, p10, p50, p90, p_peak
A new model = a new folder. No code change.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))


@lru_cache
def sites() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sites.csv")


def model_ids() -> list[str]:
    return sorted(p.name for p in (DATA_DIR / "models").iterdir() if (p / "meta.json").exists())


def model_meta(model_id: str) -> dict:
    return {"id": model_id, **json.loads((DATA_DIR / "models" / model_id / "meta.json").read_text())}


@lru_cache
def forecasts(model_id: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "models" / model_id / "forecasts.parquet")


@lru_cache
def history(site_id: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "history" / f"{site_id}.parquet")
