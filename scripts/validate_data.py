"""Validate a data dir against the data layout in README.md. Exit 1 on errors."""
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data")
errors: list[str] = []


def err(msg: str) -> None:
    errors.append(msg)


def utc(df: pd.DataFrame, col: str, where: str) -> None:
    if str(df[col].dtype) != "datetime64[ns, UTC]" and not str(df[col].dtype).startswith("datetime64[") or df[col].dt.tz is None:
        err(f"{where}: {col} must be tz-aware UTC")


sites = pd.read_csv(root / "sites.csv") if (root / "sites.csv").exists() else None
if sites is None:
    err("sites.csv missing")
else:
    missing = {"site_id", "name", "country", "level", "role"} - set(sites.columns)
    if missing:
        err(f"sites.csv: missing columns {sorted(missing)}")

for site in (sites.site_id if sites is not None and "site_id" in sites else []):
    p = root / "history" / f"{site}.parquet"
    if not p.exists():
        err(f"history/{site}.parquet missing")
        continue
    h = pd.read_parquet(p)
    if not {"timestamp_utc", "heat_kwh"} <= set(h.columns):
        err(f"{p.name}: need timestamp_utc, heat_kwh")
        continue
    utc(h, "timestamp_utc", p.name)
    if h.timestamp_utc.duplicated().any():
        err(f"{p.name}: duplicate timestamps")
    if (h.heat_kwh.dropna() < 0).any():
        err(f"{p.name}: negative heat_kwh")

models = [d for d in (root / "models").iterdir() if d.is_dir()] if (root / "models").exists() else []
if not models:
    err("no models/ folders")
COLS = ["site_id", "issue_time_utc", "horizon_h", "target_time_utc", "p10", "p50", "p90", "p_peak"]
for d in models:
    w = f"models/{d.name}"
    if not (d / "meta.json").exists():
        err(f"{w}: meta.json missing")
    else:
        try:
            json.loads((d / "meta.json").read_text())["name"]
        except Exception:
            err(f"{w}: meta.json must be JSON with 'name'")
    if not (d / "forecasts.parquet").exists():
        err(f"{w}: forecasts.parquet missing")
        continue
    f = pd.read_parquet(d / "forecasts.parquet")
    if set(COLS) - set(f.columns):
        err(f"{w}: missing columns {sorted(set(COLS) - set(f.columns))}")
        continue
    for c in ("issue_time_utc", "target_time_utc"):
        utc(f, c, w)
    if sites is not None and not set(f.site_id) <= set(sites.site_id):
        err(f"{w}: unknown site_id {sorted(set(f.site_id) - set(sites.site_id))}")
    if not f.horizon_h.between(1, 72).all():
        err(f"{w}: horizon_h outside 1..72")
    counts = f.groupby(["site_id", "issue_time_utc"]).horizon_h.nunique()
    if (counts != 72).any():
        err(f"{w}: {int((counts != 72).sum())} (site, issue) groups without all 72 horizons")
    if f.duplicated(["site_id", "issue_time_utc", "horizon_h"]).any():
        err(f"{w}: duplicate (site, issue, horizon)")
    if not (f.target_time_utc == f.issue_time_utc + pd.to_timedelta(f.horizon_h - 1, unit="h")).all():
        err(f"{w}: target_time_utc != issue_time_utc + (horizon_h - 1) h")
    if f.p50.isna().any():
        err(f"{w}: p50 has NaN")
    q = f.dropna(subset=["p10", "p90"])
    if not ((q.p10 <= q.p50) & (q.p50 <= q.p90)).all():
        err(f"{w}: quantile crossing (need p10 <= p50 <= p90)")
    if not f.p_peak.dropna().between(0, 1).all():
        err(f"{w}: p_peak outside 0..1")

if errors:
    print("\n".join(f"ERROR {e}" for e in errors))
    sys.exit(1)
print(f"OK: {root} valid ({len(models)} models)")
