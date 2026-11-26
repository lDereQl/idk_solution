"""Export the demo data: forecasts of three models for Kaunas, in the data-contract format (docs/data-contract.md).

    python scripts/export_demo_data.py <clever-energy-data dir> [out dir, default demo_data]

Needs the clever-energy-data folder (python pipeline.py all): features/when2heat.parquet and standard/heat/when2heat.parquet,
and the v3 weights in weights/ (python weights/preparemodels_v3.py). Needs lightgbm and joblib on top of requirements.txt.

The demand is the *simulated* Kaunas series (When2Heat Lithuania profile scaled to Kauno energija's 1,414 GWh a year);
the weather is real (Open-Meteo archived forecasts). The forecast issue times are daily at 06:00 UTC, 1-10 January 2022,
a window with a cold snap on 11-12 January. Models:
  baseline_24h    same hour of the latest known day; no intervals
  lgbm_v3         the v3 weights as trained on Brønderslev and Sønderborg 2016-2018, applied to Kaunas without any Kaunas data
  lgbm_v3_kaunas  the same model fitted on the simulated Kaunas series 2008-2020 and calibrated on 2021 (step 1 of the
                  implementation plan on simulated data; the accuracy says nothing about real Kaunas)
p_peak is the chance that the hour exceeds the 90th percentile of the site's demand in 2021, taken from a normal
distribution through the forecast P10-P90 range.
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("v3", ROOT / "weights" / "preparemodels_v3.py")
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

SITE = "kaunas_synthetic"
FIRST_ISSUE, LAST_ISSUE = pd.Timestamp("2022-01-01 06:00", tz="UTC"), pd.Timestamp("2022-01-10 06:00", tz="UTC")
REF_YEAR = 2021  # site scale, calibration and peak threshold come from the year before the forecasts
Z90 = 1.2816
COLS = ["site_id", "issue_time_utc", "horizon_h", "target_time_utc", "p10", "p50", "p90", "p_peak"]
MODELS = {
    "baseline_24h": ("Seasonal naive 24h", "Same hour of the latest known day. No intervals."),
    "lgbm_v3": ("LightGBM v3, trained abroad",
                "Quantile model trained on Danish and German networks 2016-2018, applied to Kaunas without Kaunas data."),
    "lgbm_v3_kaunas": ("LightGBM v3, retrained on Kaunas",
                       "Same model fitted on the simulated Kaunas series 2008-2020, range calibrated on 2021."),
}


def forecast_frame(df, p10, p50, p90, p_peak):
    f = pd.DataFrame({"site_id": SITE, "issue_time_utc": df["issue_time_utc"].to_numpy(),
                      "horizon_h": df["horizon_h"].to_numpy().astype("int64"),
                      "target_time_utc": df["target_time_utc"].to_numpy(), "p10": p10, "p50": p50, "p90": p90,
                      "p_peak": p_peak})
    for c in ("issue_time_utc", "target_time_utc"):
        f[c] = pd.to_datetime(f[c], utc=True).astype("datetime64[ns, UTC]")
    return f[COLS].sort_values(["issue_time_utc", "horizon_h"]).reset_index(drop=True)


def quantile_forecast(models, df, margin, threshold):
    """kWh forecast from scaled-input quantile models; P10/P90 widened by the conformal margin."""
    scale = df["scale"].to_numpy()
    q = np.stack([(models["p10"].predict(df[v3.INPUTS]) - margin) * scale,
                  models["p50"].predict(df[v3.INPUTS]) * scale,
                  (models["p90"].predict(df[v3.INPUTS]) + margin) * scale], axis=1)
    p10, p50, p90 = np.clip(np.sort(q, axis=1), 0, None).T  # no quantile crossing, no negative demand
    sigma = np.maximum((p90 - p10) / (2 * Z90), 1e-3 * scale)
    z = (threshold - p50) / sigma
    p_peak = 1 - 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
    return forecast_frame(df, p10, p50, p90, np.clip(p_peak, 0, 1))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ce = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "demo_data"

    feats = pd.read_parquet(ce / "features" / "when2heat.parquet")
    feats = feats[feats["site_id"] == SITE]
    hist = pd.read_parquet(ce / "standard" / "heat" / "when2heat.parquet")
    hist = hist[hist["site_id"] == SITE][["timestamp_utc", "heat_kwh"]].reset_index(drop=True)
    threshold = hist[hist["timestamp_utc"].dt.year == REF_YEAR]["heat_kwh"].quantile(0.9)

    scaled, scales = v3.add_scaled(feats, [REF_YEAR])
    year = scaled["issue_time_utc"].dt.year
    test = scaled[scaled["issue_time_utc"].between(FIRST_ISSUE, LAST_ISSUE)]
    print(f"{len(test) // 72} issues x 72 h; site scale {scales[SITE]:,.0f} kWh; peak threshold {threshold:,.0f} kWh")

    meta = json.loads((ROOT / "weights" / "lightgbm_v3_meta.json").read_text())
    abroad = {q: joblib.load(ROOT / "weights" / f"lightgbm_{q}_v3.pkl") for q in ("p10", "p50", "p90")}
    print("fitting the Kaunas model on 2008-2020 ...")
    local = v3.fit(scaled[year <= 2020])
    local_margin = v3.interval_margin(local, scaled[year == REF_YEAR])

    naive = test["heat_lag_24h"].to_numpy()
    forecasts = {
        "baseline_24h": forecast_frame(test, np.nan, naive, np.nan, np.nan),
        "lgbm_v3": quantile_forecast(abroad, test, meta["interval_margin"], threshold),
        "lgbm_v3_kaunas": quantile_forecast(local, test, local_margin, threshold),
    }

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"site_id": SITE, "name": "Kaunas (synthetic)", "country": "LT", "level": "city (synthetic)",
                   "role": "kaunas_check"}]).to_csv(out / "sites.csv", index=False)
    start, end = FIRST_ISSUE - pd.Timedelta(days=8), LAST_ISSUE + pd.Timedelta(hours=72)
    (out / "history").mkdir(exist_ok=True)
    hist[hist["timestamp_utc"].between(start, end)].to_parquet(out / f"history/{SITE}.parquet", index=False)
    for mid, f in forecasts.items():
        (out / "models" / mid).mkdir(parents=True, exist_ok=True)
        f.to_parquet(out / "models" / mid / "forecasts.parquet", index=False)
        (out / "models" / mid / "meta.json").write_text(json.dumps(dict(zip(("name", "description"), MODELS[mid]))))

    # what the demo will show
    h = hist.set_index("timestamp_utc")["heat_kwh"]
    for mid, f in forecasts.items():
        y = h.reindex(f["target_time_utc"]).to_numpy()
        mae = 100 * np.abs(f["p50"].to_numpy() - y).mean() / y.mean()
        inside = ((y >= f["p10"]) & (y <= f["p90"]))[f["p10"].notna().to_numpy()]
        print(f"{mid:15s} MAE {mae:5.1f} % of mean demand" + (f" | P10-P90 coverage {100 * inside.mean():.1f} %" if len(inside) else ""))
    print("written to", out)


if __name__ == "__main__":
    main()
