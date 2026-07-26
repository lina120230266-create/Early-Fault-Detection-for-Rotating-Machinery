"""
RUL Predictive Maintenance — Model Training Script
Dataset: NASA C-MAPSS FD001

Pulls ML_READY_TRAIN / ML_READY_TEST from Snowflake (or local CSV fallback),
trains Random Forest and XGBoost regressors, evaluates with RMSE and the
NASA scoring function, and writes predictions to a CSV ready for
write-back into the Snowflake PREDICTIONS table.

Usage:
    python train_rul_model.py --source snowflake
    python train_rul_model.py --source csv --train_path train.csv --test_path test.csv --rul_path RUL_FD001.txt
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed — skipping XGBoost model. "
          "Install with: pip install xgboost --break-system-packages")


# ----------------------------------------------------------------------
# NASA scoring function — penalises late predictions more than early ones
# ----------------------------------------------------------------------
def nasa_score(y_true, y_pred):
    """
    Official NASA C-MAPSS scoring function.
    d = predicted - actual
    score = sum( exp(-d/13) - 1 )   for d < 0  (early prediction, lenient)
            sum( exp( d/10) - 1 )   for d >= 0 (late prediction, penalised hard)
    """
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return np.sum(score)


def evaluate(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    s_score = nasa_score(y_true, y_pred)
    print(f"\n[{label}] RMSE: {rmse:.2f} | MAE: {mae:.2f} | NASA S-score: {s_score:.2f}")
    return {"rmse": rmse, "mae": mae, "s_score": s_score}


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_from_snowflake():
    """
    Requires snowflake-connector-python and valid credentials.
    pip install snowflake-connector-python --break-system-packages
    """
    import snowflake.connector
    import os

    conn = snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        warehouse="RUL_WH",
        database="RUL_DB",
        schema="PUBLIC",
    )
    train_df = pd.read_sql("SELECT * FROM ML_READY_TRAIN", conn)
    test_df = pd.read_sql("SELECT * FROM ML_READY_TEST", conn)
    rul_df = pd.read_sql("SELECT * FROM RAW_RUL_FD001", conn)
    conn.close()
    return train_df, test_df, rul_df


def load_from_csv(train_path, test_path, rul_path):
    """
    Fallback: load directly from the original CMAPSS .txt files and replicate
    the same cleaning/feature steps done in Snowflake, in case you want to
    prototype locally before wiring up the warehouse connection.
    """
    cols = (
        ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
        + [f"sensor_{i}" for i in range(1, 22)]
    )
    train_raw = pd.read_csv(train_path, sep=r"\s+", header=None, names=cols)
    test_raw = pd.read_csv(test_path, sep=r"\s+", header=None, names=cols)
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["true_rul"])
    rul_df["unit"] = range(1, len(rul_df) + 1)

    drop_sensors = ["sensor_1", "sensor_5", "sensor_6", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]
    keep_cols = [c for c in cols if c not in drop_sensors]

    train_df = train_raw[keep_cols].copy()
    test_df = test_raw[keep_cols].copy()

    # RUL label, capped at 125
    max_cycle = train_df.groupby("unit")["cycle"].transform("max")
    train_df["rul"] = np.minimum(max_cycle - train_df["cycle"], 125)

    # Rolling features (window=5, per unit)
    sensor_cols = [c for c in keep_cols if c.startswith("sensor_")]
    for df in (train_df, test_df):
        for s in sensor_cols:
            df[f"{s}_roll_avg"] = (
                df.groupby("unit")[s].transform(lambda x: x.rolling(5, min_periods=1).mean())
            )

    return train_df, test_df, rul_df


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["snowflake", "csv"], default="csv")
    parser.add_argument("--train_path", default="train_FD001.txt")
    parser.add_argument("--test_path", default="test_FD001.txt")
    parser.add_argument("--rul_path", default="RUL_FD001.txt")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    print(f"Loading data from {args.source}...")
    if args.source == "snowflake":
        train_df, test_df, rul_df = load_from_snowflake()
    else:
        train_df, test_df, rul_df = load_from_csv(args.train_path, args.test_path, args.rul_path)

    feature_cols = [c for c in train_df.columns if c not in ("unit", "cycle", "rul")]
    X = train_df[feature_cols]
    y = train_df["rul"]
    groups = train_df["unit"]

    # ------------------------------------------------------------
    # Group K-Fold CV — split by engine unit, never by row, so we
    # never leak future cycles of the same engine into validation.
    # ------------------------------------------------------------
    gkf = GroupKFold(n_splits=5)
    train_idx, val_idx = next(gkf.split(X, y, groups))
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    results = {}

    # ------------------------------------------------------------
    # Model 1: Random Forest (baseline)
    # ------------------------------------------------------------
    print("\nTraining Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_val_pred = rf.predict(X_val)
    results["random_forest"] = evaluate(y_val, rf_val_pred, "Random Forest (validation)")

    # ------------------------------------------------------------
    # Model 2: XGBoost (if available)
    # ------------------------------------------------------------
    best_model, best_name = rf, "random_forest"
    if HAS_XGB:
        print("\nTraining XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
        xgb_model.fit(X_train, y_train)
        xgb_val_pred = xgb_model.predict(X_val)
        results["xgboost"] = evaluate(y_val, xgb_val_pred, "XGBoost (validation)")

        if results["xgboost"]["rmse"] < results["random_forest"]["rmse"]:
            best_model, best_name = xgb_model, "xgboost"

    print(f"\nBest model on validation: {best_name}")

    # ------------------------------------------------------------
    # Final test-set prediction — uses LAST cycle per unit
    # (that's the point in time the official RUL_FD001.txt corresponds to)
    # ------------------------------------------------------------
    last_cycle_test = test_df.loc[test_df.groupby("unit")["cycle"].idxmax()]
    X_test_final = last_cycle_test[feature_cols]
    test_pred = best_model.predict(X_test_final)

    pred_df = pd.DataFrame({
        "unit": last_cycle_test["unit"].values,
        "predicted_rul": test_pred,
        "actual_rul": rul_df.sort_values("unit")["true_rul"].values,
        "model_name": best_name,
    })
    pred_df["abs_error"] = (pred_df["predicted_rul"] - pred_df["actual_rul"]).abs()

    final_metrics = evaluate(pred_df["actual_rul"], pred_df["predicted_rul"], f"{best_name} (held-out test)")

    pred_df.to_csv(args.output, index=False)
    print(f"\nPredictions written to {args.output}")
    print("Load this into Snowflake's PREDICTIONS table via ADF or:")
    print(f"  COPY INTO PREDICTIONS FROM @stage/{args.output} FILE_FORMAT=(TYPE=CSV SKIP_HEADER=1);")


if __name__ == "__main__":
    main()
