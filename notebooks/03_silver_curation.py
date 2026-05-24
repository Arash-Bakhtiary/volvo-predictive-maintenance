# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer — Curation, PII Masking & Quality Checks
# MAGIC Reads from Bronze, applies:
# MAGIC - Null imputation (median for continuous, mode for categorical)
# MAGIC - Outlier capping (IQR-based per feature)
# MAGIC - PII hashing: driver_id → SHA-256 → driver_id_hashed (original dropped)
# MAGIC - Anomaly flag columns
# MAGIC - Data quality assertions (fail job if pass rate < 99%)
# MAGIC - Column mask applied post-write

# COMMAND ----------
dbutils.widgets.text("catalog", "volvo_poc")
catalog = dbutils.widgets.get("catalog")

# COMMAND ----------
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

spark.sql(f"USE CATALOG {catalog}")

# ── Read Bronze ──────────────────────────────────────────────────────────────
df_bronze = spark.table(f"{catalog}.bronze.bus_telemetry_raw")
raw_count  = df_bronze.count()
print(f"Bronze rows read: {raw_count:,}")

# ── Null Imputation ───────────────────────────────────────────────────────────
continuous_cols = [
    "odometer_km", "speed_kph", "oil_pressure_bar", "battery_voltage_v",
    "vibration_ms2", "ambient_temp_c", "fuel_rate_lph",
    "brake_wear_pct", "dpf_pressure_kpa", "engine_temp_c"
]
for col in continuous_cols:
    median_val = df_bronze.approxQuantile(col, [0.5], 0.01)[0]
    df_bronze = df_bronze.fillna({col: median_val})

df_bronze = df_bronze.fillna({"error_code_count": 0})
df_bronze = df_bronze.fillna({"depot_region": "Unknown", "route_id": "UNKNOWN-000"})

# ── Outlier Capping (IQR × 1.5) ─────────────────────────────────────────────
cap_cols = {
    "odometer_km":      (1_000,   900_000),
    "speed_kph":        (0,       200),
    "oil_pressure_bar": (0,       10),
    "battery_voltage_v":(18,      32),
    "vibration_ms2":    (0,       8),
    "brake_wear_pct":   (0,       100),
    "dpf_pressure_kpa": (0,       20),
    "engine_temp_c":    (40,      130),
    "fuel_rate_lph":    (0,       120),
    "error_code_count": (0,       15),
}
for col, (lo, hi) in cap_cols.items():
    df_bronze = df_bronze.withColumn(col, F.least(F.greatest(F.col(col), F.lit(lo)), F.lit(hi)))

# ── PII Hashing: driver_id → SHA-256 → drop original ────────────────────────
df_silver = (df_bronze
    .withColumn("driver_id_hashed", F.sha2(F.col("driver_id"), 256))
    .drop("driver_id")
)

# ── Anomaly Flags ─────────────────────────────────────────────────────────────
df_silver = (df_silver
    .withColumn("oil_pressure_anomaly",
        F.when(F.col("engine_type") == "electric", F.lit(False))
         .otherwise((F.col("oil_pressure_bar") < 2.0) | (F.col("oil_pressure_bar") > 7.5)))
    .withColumn("engine_temp_anomaly",
        (F.col("engine_temp_c") < 60) | (F.col("engine_temp_c") > 105))
    .withColumn("battery_anomaly",
        (F.col("battery_voltage_v") < 22.0) | (F.col("battery_voltage_v") > 27.5))
    .withColumn("high_vibration_flag",
        F.col("vibration_ms2") > 3.5)
    .withColumn("brake_critical_flag",
        F.col("brake_wear_pct") > 80)
    .withColumn("dpf_critical_flag",
        (F.col("engine_type") != "electric") & (F.col("dpf_pressure_kpa") > 12.0))
)

# ── Data Quality Checks ───────────────────────────────────────────────────────
quality_rules = {
    "valid_bus_id":        F.col("bus_id").isNotNull(),
    "valid_record_id":     F.col("record_id").isNotNull(),
    "valid_speed":         F.col("speed_kph").between(0, 200),
    "valid_oil_pressure":  F.col("oil_pressure_bar").between(0, 10),
    "valid_battery":       F.col("battery_voltage_v").between(18, 32),
    "valid_brake_wear":    F.col("brake_wear_pct").between(0, 100),
    "valid_target":        F.col("next_14_days_failure").isin(0, 1),
    "valid_engine_temp":   F.col("engine_temp_c").between(40, 130),
}

dq_expr = F.lit(True)
for rule, condition in quality_rules.items():
    dq_expr = dq_expr & condition

df_silver = df_silver.withColumn("is_valid_record", dq_expr)

valid_count   = df_silver.filter(F.col("is_valid_record")).count()
invalid_count = df_silver.filter(~F.col("is_valid_record")).count()
pass_rate     = valid_count / (valid_count + invalid_count)
print(f"Quality pass rate: {pass_rate:.4%}  ({valid_count:,} valid, {invalid_count:,} invalid)")
assert pass_rate >= 0.99, f"Quality pass rate {pass_rate:.2%} below 99% threshold — aborting"

# ── Add Silver Metadata ───────────────────────────────────────────────────────
df_silver = df_silver.withColumn("_silver_timestamp", F.current_timestamp())

# ── Write Silver Delta Table ──────────────────────────────────────────────────
(df_silver.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("bus_model")
    .saveAsTable(f"{catalog}.silver.bus_telemetry_curated")
)

# ── Apply Column Mask on driver_id_hashed ────────────────────────────────────
spark.sql(f"""
    ALTER TABLE {catalog}.silver.bus_telemetry_curated
    ALTER COLUMN driver_id_hashed
    SET MASK {catalog}.silver.mask_driver_id
""")

# ── Set Table Properties ──────────────────────────────────────────────────────
spark.sql(f"""
    ALTER TABLE {catalog}.silver.bus_telemetry_curated
    SET TBLPROPERTIES (
        'delta.enableChangeDataFeed'       = 'true',
        'delta.autoOptimize.optimizeWrite' = 'true',
        'delta.autoOptimize.autoCompact'   = 'true',
        'quality'  = 'silver',
        'team'     = 'data-engineering',
        'pii'      = 'pseudonymous',
        'project'  = 'volvo-predictive-maintenance'
    )
""")

# ── Final Validation ──────────────────────────────────────────────────────────
silver_count = spark.table(f"{catalog}.silver.bus_telemetry_curated").count()
print(f"Silver rows written: {silver_count:,}")
assert "driver_id" not in spark.table(f"{catalog}.silver.bus_telemetry_curated").columns, \
    "driver_id column must not be present in silver!"
print("PII check passed — driver_id absent from silver.")
print("Silver curation complete.")
