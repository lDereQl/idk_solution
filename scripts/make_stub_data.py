"""Generate synthetic data in the real format, so the API works before the modelling
team delivers. Replace data/ contents with real exports later - same layout."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data"
rng = np.random.default_rng(0)

SITES = [  # site_id, name, country, level, role, mean kWh
    ("flensburg", "Flensburg", "DE", "network", "test", 60000),
    ("sonderborg_total", "Sønderborg", "DK", "network", "train", 20000),
    ("bronderslev_dma_a", "Brønderslev DMA A", "DK", "network area", "train", 1500),
    ("kaunas_synthetic", "Kaunas (synthetic)", "LT", "city (synthetic)", "kaunas_check", 160000),
]
# model_id, name, description, noise (relative), interval half-width (relative)
MODELS = [
    ("baseline_24h", "Seasonal naive 24h", "Same hour yesterday. No intervals.", 0.15, 0.0),
    ("lgbm_v1", "LightGBM quantile v1", "Iteration 1, raw quantiles.", 0.10, 0.05),
    ("lgbm_v2", "LightGBM v2 (tuned+conformal)", "Optuna, 5x peak weights, conformal.", 0.11, 0.18),
]

(OUT / "history").mkdir(parents=True, exist_ok=True)
idx = pd.date_range("2019-11-01", "2020-01-31 23:00", freq="h", tz="UTC")
hist = {}
for sid, *_, mean in SITES:
    h = idx.hour.to_numpy()
    seasonal = np.cos((idx.day_of_year.to_numpy() - 15) / 365 * 2 * np.pi)
    daily = 0.15 * np.sin((h - 6) / 24 * 2 * np.pi)
    y = mean * (1 + 0.5 * seasonal + daily) * (1 + rng.normal(0, 0.03, len(idx)))
    hist[sid] = pd.DataFrame({"timestamp_utc": idx, "heat_kwh": y.clip(min=0)})
    hist[sid].to_parquet(OUT / "history" / f"{sid}.parquet", index=False)

pd.DataFrame(SITES, columns=["site_id", "name", "country", "level", "role", "mean_kwh"]).drop(
    columns="mean_kwh"
).to_csv(OUT / "sites.csv", index=False)

issues = pd.date_range("2020-01-01 06:00", "2020-01-28 06:00", freq="D", tz="UTC")
for mid, name, desc, noise, width in MODELS:
    rows = []
    for sid, *_ in SITES:
        h = hist[sid].set_index("timestamp_utc").heat_kwh
        thr = h.quantile(0.9)
        for t in issues:
            for k in range(1, 73):
                tt = t + pd.Timedelta(hours=k - 1)
                p50 = h[tt] * (1 + rng.normal(0, noise))
                p_peak = float(np.clip((p50 - thr * 0.85) / (thr * 0.3), 0, 1)) if width else np.nan
                rows.append((sid, t, k, tt, p50 * (1 - width) if width else np.nan, p50,
                             p50 * (1 + width) if width else np.nan, p_peak))
    d = pd.DataFrame(rows, columns=["site_id", "issue_time_utc", "horizon_h", "target_time_utc",
                                    "p10", "p50", "p90", "p_peak"])
    (OUT / "models" / mid).mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "models" / mid / "forecasts.parquet", index=False)
    (OUT / "models" / mid / "meta.json").write_text(json.dumps({"name": name, "description": desc}))
print("stub data written to", OUT)
