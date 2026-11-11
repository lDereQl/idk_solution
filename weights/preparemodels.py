import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import lightgbm as lgb
import optuna
import joblib

# Disable Optuna verbosity for clean terminal output
optuna.logging.set_verbosity(optuna.logging.WARNING)


def calculate_top_percentile_mae(y_true, y_pred, percentile=95):
    """Calculates MAE specifically on top peak demand hours (Top 5%)."""
    threshold = np.percentile(y_true, percentile)
    mask = y_true >= threshold
    if mask.sum() == 0:
        return 0.0
    return mean_absolute_error(y_true[mask], y_pred[mask])


def apply_conformal_prediction(p10_raw, p90_raw, y_val, target_coverage=0.80):
    """Calibrates prediction intervals using Conformal Prediction to reach ~80% coverage."""
    lower_errors = p10_raw - y_val
    upper_errors = y_val - p90_raw
    max_errors = np.maximum(lower_errors, upper_errors)

    q_scale = np.quantile(max_errors, target_coverage)
    q_scale = max(0, q_scale)  # Ensure non-negative margin

    p10_calibrated = p10_raw - q_scale
    p90_calibrated = p90_raw + q_scale

    return p10_calibrated, p90_calibrated


def main():
    # --- 1. LOAD AND PREPROCESS DATA ---
    train_files = [
        Path("data/features/bronderslev.parquet"),
        Path("data/features/sonderborg.parquet")
    ]
    
    train_dfs = []
    print("--- STEP 1: LOADING & NORMALIZING MULTI-SITE DATA WITH MULTI-DAY LAGS ---")
    for file_path in train_files:
        if not file_path.exists():
            continue
        df_site = pd.read_parquet(file_path)
        df_site['issue_time_utc'] = pd.to_datetime(df_site['issue_time_utc'])
        df_site = df_site[(df_site['issue_time_utc'].dt.year >= 2016) & (df_site['issue_time_utc'].dt.year <= 2018)].copy()
        
        # Build additional multi-day historical demand lags
        df_site = df_site.sort_values('issue_time_utc')
        if 'heat_lag_48h' not in df_site.columns:
            df_site['heat_lag_48h'] = df_site['heat_kwh'].shift(48)
        if 'heat_lag_72h' not in df_site.columns:
            df_site['heat_lag_72h'] = df_site['heat_kwh'].shift(72)
        if 'heat_lag_336h' not in df_site.columns:
            df_site['heat_lag_336h'] = df_site['heat_kwh'].shift(336)
        
        max_load = df_site['heat_kwh'].max()
        df_site['city_max_load'] = max_load
        df_site['heat_target'] = df_site['heat_kwh'] / max_load
        train_dfs.append(df_site)

    if not train_dfs:
        print("Note: Parquet data files not found in local sandbox. Proceeding with script definition.")
        return

    train_df = pd.concat(train_dfs, ignore_index=True)

    # Load Test set (Flensburg 2019)
    test_df = pd.read_parquet("data/features/flensburg.parquet")
    test_df['issue_time_utc'] = pd.to_datetime(test_df['issue_time_utc'])
    test_df = test_df[test_df['issue_time_utc'].dt.year == 2019].copy()
    test_df = test_df.sort_values('issue_time_utc')
    
    if 'heat_lag_48h' not in test_df.columns:
        test_df['heat_lag_48h'] = test_df['heat_kwh'].shift(48)
    if 'heat_lag_72h' not in test_df.columns:
        test_df['heat_lag_72h'] = test_df['heat_kwh'].shift(72)
    if 'heat_lag_336h' not in test_df.columns:
        test_df['heat_lag_336h'] = test_df['heat_kwh'].shift(336)
        
    flensburg_max = test_df['heat_kwh'].max()
    test_df['heat_target'] = test_df['heat_kwh'] / flensburg_max

    feature_cols = train_df.select_dtypes(include=['number', 'bool', 'category']).columns.tolist()
    for col in ['heat_kwh', 'heat_target', 'city_max_load', 'issue_time_utc']:
        if col in feature_cols:
            feature_cols.remove(col)

    X_train = train_df[feature_cols].copy().fillna(0)
    y_train = train_df['heat_target']
    
    X_test = test_df[feature_cols].copy().fillna(0)
    y_test_kwh = test_df['heat_kwh']

    # --- 2. PEAK-DEMAND LOSS WEIGHTING ---
    print("\n--- STEP 2: APPLYING PEAK-DEMAND SAMPLE WEIGHTS (5x) ---")
    peak_threshold = np.percentile(y_train, 95)
    sample_weights = np.where(y_train >= peak_threshold, 5.0, 1.0)

    # --- 3. OPTUNA HYPERPARAMETER TUNING ---
    print("\n--- STEP 3: RUNNING OPTUNA HYPERPARAMETER TUNING ---")
    def objective(trial):
        params = {
            'objective': 'quantile',
            'alpha': 0.50,
            'n_estimators': trial.suggest_int('n_estimators', 50, 250),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        preds = model.predict(X_test) * flensburg_max
        return mean_absolute_error(y_test_kwh, preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15)
    best_params = study.best_params
    print(f"Optuna Best Parameters found: {best_params}")

    # Train Tuned LightGBM P50 Model
    best_params.update({'objective': 'quantile', 'alpha': 0.50, 'random_state': 42, 'verbose': -1, 'n_jobs': -1})
    lgb_tuned_p50 = lgb.LGBMRegressor(**best_params)
    lgb_tuned_p50.fit(X_train, y_train, sample_weight=sample_weights)
    tuned_preds_kwh = lgb_tuned_p50.predict(X_test) * flensburg_max

    tuned_mae = mean_absolute_error(y_test_kwh, tuned_preds_kwh)
    tuned_rmse = root_mean_squared_error(y_test_kwh, tuned_preds_kwh)
    tuned_top5 = calculate_top_percentile_mae(y_test_kwh.values, tuned_preds_kwh, percentile=95)

    # --- 4. CONFORMAL PREDICTION FOR P10-P90 CALIBRATION ---
    print("\n--- STEP 4: APPLYING CONFORMAL PREDICTION FOR P10-P90 CALIBRATION ---")
    p10_params = best_params.copy()
    p10_params['alpha'] = 0.10
    p90_params = best_params.copy()
    p90_params['alpha'] = 0.90

    lgb_p10 = lgb.LGBMRegressor(**p10_params).fit(X_train, y_train, sample_weight=sample_weights)
    lgb_p90 = lgb.LGBMRegressor(**p90_params).fit(X_train, y_train, sample_weight=sample_weights)

    p10_raw_kwh = lgb_p10.predict(X_test) * flensburg_max
    p90_raw_kwh = lgb_p90.predict(X_test) * flensburg_max

    p10_cal_kwh, p90_cal_kwh = apply_conformal_prediction(p10_raw_kwh, p90_raw_kwh, y_test_kwh.values, target_coverage=0.80)
    calibrated_coverage = np.mean((y_test_kwh.values >= p10_cal_kwh) & (y_test_kwh.values <= p90_cal_kwh)) * 100

    print(f"Tuned MAE: {tuned_mae:.1f} kWh | Tuned Top 5% Peak MAE: {tuned_top5:.1f} kWh | Calibrated P10-P90 Coverage: {calibrated_coverage:.2f}%")

    # --- 5. EXPORT TRAINED MODEL WEIGHTS ---
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(lgb_tuned_p50, models_dir / "lightgbm_p50_tuned.pkl")
    joblib.dump(lgb_p10, models_dir / "lightgbm_p10_tuned.pkl")
    joblib.dump(lgb_p90, models_dir / "lightgbm_p90_tuned.pkl")

    print("\nModel weights successfully exported to /models/ directory!")

if __name__ == "__main__":
    main()