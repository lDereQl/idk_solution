"""LightGBM v3: the corrected version of preparemodels.py.

What changed from v2, and why (found when testing v2 on data it never saw):
- v2 divided the target by the site's maximum but kept past demand (heat_lag_*, heat_roll24_*) in raw kWh,
  so the model learned the size of the training networks: on Flensburg 2019 it was worse than
  "same hour yesterday" (18.1 % vs 11.9 % mean absolute error) and on Brønderslev area A it failed (60.1 %).
  v3 divides the target AND every past-demand input by the site's mean demand in the years before the test year.
- v2 used inputs derived from the target (peak_event, peak_threshold_kwh, target_interpolated) and
  site-specific columns (active_meters, supply/return temperatures, price). v3 leaves them out.
- v2 tuned and calibrated the P10-P90 range on the test year. v3 calibrates on a separate year:
  fit on 2016-2017, calibrate on 2018, then refit on 2016-2018 and test on 2019.

Input: the feature tables from clever-energy-data (python pipeline.py all) copied to data/features/.
Output: models/lightgbm_{p10,p50,p90}_v3.pkl and models/lightgbm_v3_meta.json (inputs, scales, interval margin).
Prediction in kWh = model.predict(X with past demand divided by scale) * scale, where scale is the site's mean
demand in an earlier period; P10/P90 are widened by the margin (also in units of scale).

Needs: pandas, numpy, pyarrow, lightgbm, joblib.
"""
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

FEATURES = Path("data/features")
MODELS = Path("models")
TRAIN_FILES = ["bronderslev.parquet", "sonderborg.parquet"]
TRAIN_YEARS, FIT_YEARS, CAL_YEAR, TEST_YEAR = [2016, 2017, 2018], [2016, 2017], 2018, 2019
QUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}
TARGET_COVERAGE = 0.80

# past demand: scaled per site like the target
HEAT = ["heat_lag_1h", "heat_lag_24h", "heat_lag_168h", "heat_roll24_mean", "heat_roll24_max"]
OTHER = ["horizon_h", "temp_c", "apparent_temp_c", "wind_ms", "shortwave_wm2", "cloud_pct", "rh_pct", "snow_depth_m",
         "hdh", "temp_ema_24h", "temp_ema_48h", "temp_ema_72h", "cold_snap", "hour_sin", "hour_cos", "dow_sin",
         "dow_cos", "is_holiday", "month", "heating_season"]
INPUTS = OTHER + [c + "_s" for c in HEAT]
PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=-1)


def add_scaled(df, ref_years):
    """Divide target and past demand by each site's mean demand in ref_years (known before the test year)."""
    parts, scales = [], {}
    for site, g in df.groupby("site_id"):
        ref = g[g["issue_time_utc"].dt.year.isin(ref_years) & (g["horizon_h"] == 1)]["heat_kwh"].mean()
        g = g.copy()
        g["scale"] = ref
        for c in HEAT + ["heat_kwh"]:
            g[c + "_s"] = g[c] / ref
        parts.append(g)
        scales[site] = float(ref)
    return pd.concat(parts), scales


def fit(df):
    return {q: lgb.LGBMRegressor(objective="quantile", alpha=a, **PARAMS).fit(df[INPUTS], df["heat_kwh_s"])
            for q, a in QUANTILES.items()}


def interval_margin(models, cal):
    """Conformal margin: how much to widen P10-P90 so TARGET_COVERAGE of the calibration year falls inside."""
    y = cal["heat_kwh_s"].to_numpy()
    lo, hi = models["p10"].predict(cal[INPUTS]), models["p90"].predict(cal[INPUTS])
    return max(0.0, float(np.quantile(np.maximum(lo - y, y - hi), TARGET_COVERAGE)))


def evaluate(name, models, df, margin):
    y, scale = df["heat_kwh"].to_numpy(), df["scale"].to_numpy()
    p50 = models["p50"].predict(df[INPUTS]) * scale
    lo = (models["p10"].predict(df[INPUTS]) - margin) * scale
    hi = (models["p90"].predict(df[INPUTS]) + margin) * scale
    mae = np.abs(y - p50).mean() / y.mean() * 100
    naive = df["heat_lag_24h"].notna().to_numpy()
    naive_mae = np.abs(y[naive] - df["heat_lag_24h"].to_numpy()[naive]).mean() / y[naive].mean() * 100
    days = [np.abs(y - p50)[df["horizon_h"].between(a, b).to_numpy()].mean() / y.mean() * 100
            for a, b in [(1, 24), (25, 48), (49, 72)]]
    coverage = np.mean((y >= lo) & (y <= hi)) * 100
    print(f"{name:32s} MAE {mae:5.1f} % | same hour yesterday {naive_mae:5.1f} % | "
          f"better by {100 * (1 - mae / naive_mae):3.0f} % | day 1/2/3 {days[0]:.1f}/{days[1]:.1f}/{days[2]:.1f} % | "
          f"P10-P90 coverage {coverage:.1f} %")


def main():
    # --- 1. LOAD AND SCALE PER SITE ---
    print("--- STEP 1: LOADING TRAINING SITES AND SCALING DEMAND PER SITE ---")
    train_all, scales = add_scaled(pd.concat([pd.read_parquet(FEATURES / f) for f in TRAIN_FILES]), TRAIN_YEARS)
    years = train_all["issue_time_utc"].dt.year

    # --- 2. CALIBRATE THE INTERVAL ON A SEPARATE YEAR ---
    print(f"\n--- STEP 2: FIT {FIT_YEARS}, CALIBRATE P10-P90 ON {CAL_YEAR} ---")
    margin = interval_margin(fit(train_all[years.isin(FIT_YEARS)]), train_all[years == CAL_YEAR])
    print(f"Interval margin: {margin:.3f} x site mean demand")

    # --- 3. FINAL MODELS ON ALL TRAINING YEARS ---
    print(f"\n--- STEP 3: FINAL MODELS ON {TRAIN_YEARS} ---")
    models = fit(train_all[years.isin(TRAIN_YEARS)])

    # --- 4. TEST ON DATA THE MODEL NEVER SAW ---
    print(f"\n--- STEP 4: TEST ON {TEST_YEAR} (unseen city and unseen year) ---")
    fl, _ = add_scaled(pd.read_parquet(FEATURES / "flensburg.parquet"), [TEST_YEAR - 1])
    evaluate("Flensburg (unseen city)", models, fl[fl["issue_time_utc"].dt.year == TEST_YEAR], margin)
    test_sites = train_all[years == TEST_YEAR]
    for site in ["sonderborg_total", "bronderslev_dma_a"]:
        evaluate(f"{site} (unseen year)", models, test_sites[test_sites["site_id"] == site], margin)

    # --- 5. EXPORT ---
    MODELS.mkdir(parents=True, exist_ok=True)
    for q, m in models.items():
        joblib.dump(m, MODELS / f"lightgbm_{q}_v3.pkl")
    meta = {"inputs": INPUTS, "scaled_inputs": HEAT, "interval_margin": margin, "train_years": TRAIN_YEARS,
            "training_site_scales_kwh": scales,
            "usage": "divide heat_lag_*/heat_roll24_* by the site's mean demand (scale), predict, multiply by scale; "
                     "widen P10/P90 by interval_margin * scale"}
    (MODELS / "lightgbm_v3_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"\nModel weights exported to /{MODELS}/ directory!")


if __name__ == "__main__":
    main()
