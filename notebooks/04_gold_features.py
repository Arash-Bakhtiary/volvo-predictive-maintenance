# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer — Feature Engineering
# MAGIC Computes rolling statistics, wear indices, anomaly counts, and composite
# MAGIC risk score from Silver. Writes volvo_poc.gold.bus_features.

# COMMAND ----------
dbutils.widgets.text("catalog", "volvo_poc")
catalog = dbutils.widgets.get("catalog")

# COMMAND ----------
from pyspark.sql import functions as F, Window

spark.sql(f"USE CATALOG {catalog}")

df = spark.table(f"{catalog}.silver.bus_telemetry_curated").filter("is_valid_record = true")

# ── Latest snapshot per bus (most recent reading) ─────────────────────────────
w_bus = Window.partitionBy("bus_id").orderBy(F.col("event_timestamp").desc())
df_latest = (df
    .withColumn("_rn", F.row_number().over(w_bus))
    .filter(F.col("_rn") == 1)
    .drop("_rn")
)

# ── 7-day rolling window stats ─────────────────────────────────────────────────
w7 = (Window.partitionBy("bus_id")
      .orderBy(F.unix_timestamp("event_timestamp"))
      .rangeBetween(-7 * 86400, 0))

w30 = (Window.partitionBy("bus_id")
       .orderBy(F.unix_timestamp("event_timestamp"))
       .rangeBetween(-30 * 86400, 0))

df_windowed = (df
    .withColumn("avg_speed_7d",        F.avg("speed_kph").over(w7))
    .withColumn("avg_engine_temp_7d",  F.avg("engine_temp_c").over(w7))
    .withColumn("max_vibration_7d",    F.max("vibration_ms2").over(w7))
    .withColumn("avg_dpf_pressure_7d", F.avg("dpf_pressure_kpa").over(w7))
    .withColumn("error_code_sum_7d",   F.sum("error_code_count").over(w7))
    .withColumn("anomaly_count_30d",
        (F.sum(F.col("oil_pressure_anomaly").cast("int")).over(w30) +
         F.sum(F.col("engine_temp_anomaly").cast("int")).over(w30) +
         F.sum(F.col("battery_anomaly").cast("int")).over(w30)))
)

# Take the latest windowed snapshot per bus
df_feat = (df_windowed
    .withColumn("_rn", F.row_number().over(w_bus))
    .filter(F.col("_rn") == 1)
    .drop("_rn")
)

# ── Derived features ───────────────────────────────────────────────────────────
df_feat = (df_feat
    .withColumn("brake_wear_index",  F.col("brake_wear_pct") / 100.0)
    .withColumn("composite_risk_score",
        F.least(F.lit(1.0),
            F.col("brake_wear_index") * 0.35 +
            F.least(F.col("error_code_sum_7d") / 20.0, F.lit(1.0)) * 0.25 +
            F.least(F.col("avg_dpf_pressure_7d") / 15.0, F.lit(1.0)) * 0.20 +
            F.least(F.col("odometer_km") / 800000.0, F.lit(1.0)) * 0.10 +
            F.least(F.col("anomaly_count_30d") / 30.0, F.lit(1.0)) * 0.10
        ))
    .withColumn("snapshot_date", F.to_date("event_timestamp"))
)

# ── Select final feature columns ───────────────────────────────────────────────
feature_cols = [
    "bus_id", "bus_model", "engine_type", "depot_region", "snapshot_date",
    "odometer_km", "speed_kph", "oil_pressure_bar", "battery_voltage_v",
    "vibration_ms2", "ambient_temp_c", "fuel_rate_lph",
    "brake_wear_pct", "dpf_pressure_kpa", "error_code_count", "engine_temp_c",
    "avg_speed_7d", "avg_engine_temp_7d", "max_vibration_7d",
    "avg_dpf_pressure_7d", "error_code_sum_7d", "anomaly_count_30d",
    "brake_wear_index", "composite_risk_score",
    "oil_pressure_anomaly", "engine_temp_anomaly", "battery_anomaly",
    "high_vibration_flag", "brake_critical_flag", "dpf_critical_flag",
    "next_14_days_failure",
]

df_gold = df_feat.select(*feature_cols)

# ── Write Gold table ───────────────────────────────────────────────────────────
(df_gold.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("bus_model")
    .saveAsTable(f"{catalog}.gold.bus_features")
)

spark.sql(f"""
ALTER TABLE {catalog}.gold.bus_features
SET TBLPROPERTIES (
    'delta.enableChangeDataFeed'       = 'true',
    'delta.autoOptimize.optimizeWrite' = 'true',
    'quality' = 'gold',
    'project' = 'volvo-predictive-maintenance'
)
""")

# Apply row filter
spark.sql(f"""
ALTER TABLE {catalog}.gold.bus_features
SET ROW FILTER {catalog}.gold.filter_by_region ON (depot_region)
""")

count = spark.table(f"{catalog}.gold.bus_features").count()
print(f"Gold bus_features rows: {count:,}")
print("Gold feature engineering complete.")
