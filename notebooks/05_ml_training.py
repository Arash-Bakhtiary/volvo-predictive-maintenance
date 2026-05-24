# Databricks notebook source
# MAGIC %md
# MAGIC # ML Training — XGBoost Failure Predictor
# MAGIC Trains on gold.bus_features, logs to MLflow, registers model.

# COMMAND ----------
dbutils.widgets.text("catalog", "volvo_poc")
dbutils.widgets.text("experiment_name", "/volvo-poc/failure-predictor")
catalog         = dbutils.widgets.get("catalog")
experiment_name = dbutils.widgets.get("experiment_name")

# COMMAND ----------
import mlflow, mlflow.xgboost
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score

mlflow.set_experiment(experiment_name)
mlflow.set_registry_uri("databricks-uc")

df = spark.table(f"{catalog}.gold.bus_features").toPandas()

FEATURE_COLS = [
    "odometer_km", "speed_kph", "oil_pressure_bar", "battery_voltage_v",
    "vibration_ms2", "ambient_temp_c", "fuel_rate_lph",
    "brake_wear_pct", "dpf_pressure_kpa", "error_code_count", "engine_temp_c",
    "avg_speed_7d", "avg_engine_temp_7d", "max_vibration_7d",
    "avg_dpf_pressure_7d", "error_code_sum_7d", "anomaly_count_30d",
    "brake_wear_index", "composite_risk_score",
]
TARGET = "next_14_days_failure"

# Encode bus_model as integer
df["bus_model_enc"] = pd.Categorical(df["bus_model"]).codes
FEATURE_COLS += ["bus_model_enc"]

X = df[FEATURE_COLS].fillna(0)
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
spw = neg / pos if pos > 0 else 1.0
print(f"Train: {len(X_train)} | Test: {len(X_test)} | scale_pos_weight: {spw:.2f}")

with mlflow.start_run(run_name="xgboost_baseline") as run:
    params = dict(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=round(spw, 2),
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, gamma=0.1,
        use_label_encoder=False, eval_metric="auc",
        random_state=42, n_jobs=-1,
    )
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "auc_roc":         round(roc_auc_score(y_test, probs), 4),
        "f1_score":        round(f1_score(y_test, preds, zero_division=0), 4),
        "precision":       round(precision_score(y_test, preds, zero_division=0), 4),
        "recall":          round(recall_score(y_test, preds, zero_division=0), 4),
        "avg_precision":   round(average_precision_score(y_test, probs), 4),
        "train_size":      len(X_train),
        "test_size":       len(X_test),
    }
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    signature = mlflow.models.infer_signature(X_train, probs)
    mlflow.xgboost.log_model(
        model, "model", signature=signature,
        registered_model_name=f"{catalog}.ml.bus_failure_predictor",
        input_example=X_train.iloc[:5],
    )

    for k, v in metrics.items():
        print(f"  {k}: {v}")

    assert metrics["auc_roc"] >= 0.60, f"AUC-ROC {metrics['auc_roc']} below threshold"
    print(f"\nRun ID: {run.info.run_id}")
    print("Model registered: " + f"{catalog}.ml.bus_failure_predictor")
    print("ML training complete.")
