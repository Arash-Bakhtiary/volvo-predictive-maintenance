# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer — Raw Ingestion
# MAGIC Loads all 10 Volvo bus telemetry CSV files from DBFS into a Delta table
# MAGIC using COPY INTO (idempotent, incremental-safe).

# COMMAND ----------
dbutils.widgets.text("catalog", "volvo_poc")
catalog = dbutils.widgets.get("catalog")
raw_path = "/Volumes/volvo_poc/bronze/raw_uploads/"

# COMMAND ----------
# MAGIC %md ## 1. Create Bronze Delta Table

# COMMAND ----------
spark.sql(f"USE CATALOG {catalog}")
spark.sql("USE SCHEMA bronze")

spark.sql("""
CREATE TABLE IF NOT EXISTS bus_telemetry_raw (
    record_id            STRING        NOT NULL,
    bus_id               STRING        NOT NULL,
    bus_model            STRING        NOT NULL,
    engine_type          STRING,
    event_timestamp      TIMESTAMP,
    odometer_km          DOUBLE,
    speed_kph            DOUBLE,
    oil_pressure_bar     DOUBLE,
    battery_voltage_v    DOUBLE,
    vibration_ms2        DOUBLE,
    ambient_temp_c       DOUBLE,
    fuel_rate_lph        DOUBLE,
    brake_wear_pct       DOUBLE,
    dpf_pressure_kpa     DOUBLE,
    error_code_count     INT,
    engine_temp_c        DOUBLE,
    driver_id            STRING,
    route_id             STRING,
    depot_region         STRING,
    next_14_days_failure INT,
    _ingest_timestamp    TIMESTAMP,
    _source_file         STRING
)
USING DELTA
PARTITIONED BY (bus_model)
TBLPROPERTIES (
    'delta.enableChangeDataFeed'          = 'true',
    'delta.autoOptimize.optimizeWrite'    = 'true',
    'delta.autoOptimize.autoCompact'      = 'true',
    'quality'                             = 'bronze',
    'team'                                = 'data-engineering',
    'pii'                                 = 'true',
    'project'                             = 'volvo-predictive-maintenance'
)
COMMENT 'Raw bus telemetry — full fidelity, PII present, no transformations'
""")

print("Table created (or already exists).")

# COMMAND ----------
# MAGIC %md ## 2. Load CSVs via COPY INTO

# COMMAND ----------
# COPY INTO is idempotent — safe to re-run; only loads new files each time
result = spark.sql(f"""
COPY INTO {catalog}.bronze.bus_telemetry_raw
FROM (
    SELECT
        record_id,
        bus_id,
        bus_model,
        engine_type,
        CAST(event_timestamp AS TIMESTAMP) AS event_timestamp,
        CAST(odometer_km          AS DOUBLE)  AS odometer_km,
        CAST(speed_kph            AS DOUBLE)  AS speed_kph,
        CAST(oil_pressure_bar     AS DOUBLE)  AS oil_pressure_bar,
        CAST(battery_voltage_v    AS DOUBLE)  AS battery_voltage_v,
        CAST(vibration_ms2        AS DOUBLE)  AS vibration_ms2,
        CAST(ambient_temp_c       AS DOUBLE)  AS ambient_temp_c,
        CAST(fuel_rate_lph        AS DOUBLE)  AS fuel_rate_lph,
        CAST(brake_wear_pct       AS DOUBLE)  AS brake_wear_pct,
        CAST(dpf_pressure_kpa     AS DOUBLE)  AS dpf_pressure_kpa,
        CAST(error_code_count     AS INT)     AS error_code_count,
        CAST(engine_temp_c        AS DOUBLE)  AS engine_temp_c,
        driver_id,
        route_id,
        depot_region,
        CAST(next_14_days_failure AS INT)     AS next_14_days_failure,
        current_timestamp()                   AS _ingest_timestamp,
        _metadata.file_path                   AS _source_file
    FROM '{raw_path}'
)
FILEFORMAT = CSV
FORMAT_OPTIONS (
    'header'        = 'true',
    'inferSchema'   = 'false',
    'mergeSchema'   = 'false'
)
COPY_OPTIONS ('force' = 'false')
""")
display(result)

# COMMAND ----------
# MAGIC %md ## 3. Validate

# COMMAND ----------
count = spark.sql(f"SELECT COUNT(*) AS total FROM {catalog}.bronze.bus_telemetry_raw").collect()[0]["total"]
print(f"Total rows ingested: {count:,}")
assert count == 1_000_000, f"Expected 1,000,000 rows, got {count:,}"

spark.sql(f"""
SELECT bus_model, COUNT(*) AS rows, ROUND(AVG(next_14_days_failure)*100,2) AS failure_pct
FROM {catalog}.bronze.bus_telemetry_raw
GROUP BY bus_model
ORDER BY bus_model
""").show(truncate=False)

print("Bronze ingestion complete.")
