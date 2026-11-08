import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from . import repository as repo

app = FastAPI(title="Heat demand forecast API")


def _site_or_404(site_id: str) -> None:
    if site_id not in set(repo.sites().site_id):
        raise HTTPException(404, f"unknown site {site_id}")


def _model_or_404(model_id: str) -> None:
    if model_id not in repo.model_ids():
        raise HTTPException(404, f"unknown model {model_id}")


def _ts(s: str, name: str) -> pd.Timestamp:
    try:
        t = pd.Timestamp(s)
    except ValueError:
        raise HTTPException(422, f"bad {name}")
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/sites")
def list_sites():
    return repo.sites().to_dict("records")


@app.get("/models")
def list_models():
    return [repo.model_meta(m) for m in repo.model_ids()]


@app.get("/sites/{site_id}/history")
def get_history(site_id: str, start: str | None = None, end: str | None = None):
    _site_or_404(site_id)
    df = repo.history(site_id)
    if start:
        df = df[df.timestamp_utc >= _ts(start, "start")]
    if end:
        df = df[df.timestamp_utc < _ts(end, "end")]
    return df.assign(timestamp_utc=df.timestamp_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ")).to_dict("records")


@app.get("/sites/{site_id}/issues")
def list_issues(site_id: str, model: str):
    """Available forecast issue times (for a dropdown in the UI)."""
    _site_or_404(site_id)
    _model_or_404(model)
    f = repo.forecasts(model)
    t = f[f.site_id == site_id].issue_time_utc.drop_duplicates().sort_values()
    return list(t.dt.strftime("%Y-%m-%dT%H:%M:%SZ"))


@app.get("/sites/{site_id}/forecast")
def get_forecast(site_id: str, model: str, issue: str | None = None):
    """72 rows (h=1..72) of p10/p50/p90/p_peak. Default: latest issue."""
    _site_or_404(site_id)
    _model_or_404(model)
    f = repo.forecasts(model)
    f = f[f.site_id == site_id]
    if f.empty:
        raise HTTPException(404, "no forecast for this site/model")
    t = _ts(issue, "issue") if issue else f.issue_time_utc.max()
    f = f[f.issue_time_utc == t].sort_values("horizon_h")
    if f.empty:
        raise HTTPException(404, "no forecast for this issue time")
    f = f.assign(
        issue_time_utc=f.issue_time_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        target_time_utc=f.target_time_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    return f.drop(columns="site_id").to_dict("records")


@app.get("/sites/{site_id}/metrics")
def get_metrics(site_id: str, model: str):
    """MAE, RMSE, top-5% peak MAE, P10-P90 coverage vs actuals. Computed on the fly."""
    _site_or_404(site_id)
    _model_or_404(model)
    f = repo.forecasts(model)
    f = f[f.site_id == site_id].merge(
        repo.history(site_id), left_on="target_time_utc", right_on="timestamp_utc"
    )
    if f.empty:
        raise HTTPException(404, "no overlap between forecast and history")
    err = (f.p50 - f.heat_kwh).abs()
    peak = f.heat_kwh >= f.heat_kwh.quantile(0.95)
    return {
        "model": model,
        "site_id": site_id,
        "n": int(len(f)),
        "mae_kwh": float(err.mean()),
        "rmse_kwh": float(np.sqrt(((f.p50 - f.heat_kwh) ** 2).mean())),
        "peak_top5_mae_kwh": float(err[peak].mean()),
        "p10_p90_coverage": float(((f.heat_kwh >= f.p10) & (f.heat_kwh <= f.p90)).mean()),
    }
