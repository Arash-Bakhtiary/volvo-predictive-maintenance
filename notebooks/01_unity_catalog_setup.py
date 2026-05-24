# Databricks notebook source
# MAGIC %md
# MAGIC # Step 1 — Unity Catalog Setup
# MAGIC Creates catalog, schemas, PII column mask, row filter, and grants.

# COMMAND ----------
dbutils.widgets.text("catalog", "volvo_poc")
catalog = dbutils.widgets.get("catalog")

# COMMAND ----------
# MAGIC %md ## Create Catalog and Schemas

# COMMAND ----------
# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS volvo_poc
# MAGIC   COMMENT 'Volvo Bus Predictive Maintenance POC';
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS volvo_poc.bronze
# MAGIC   COMMENT 'Raw ingestion layer — full fidelity, PII present';
# MAGIC CREATE SCHEMA IF NOT EXISTS volvo_poc.silver
# MAGIC   COMMENT 'Curated layer — cleansed, PII masked';
# MAGIC CREATE SCHEMA IF NOT EXISTS volvo_poc.gold
# MAGIC   COMMENT 'Feature and prediction layer — business-ready';
# MAGIC CREATE SCHEMA IF NOT EXISTS volvo_poc.ml
# MAGIC   COMMENT 'MLflow experiments and registered models';

# COMMAND ----------
# MAGIC %md ## Create UC Volume

# COMMAND ----------
# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS volvo_poc.bronze.raw_uploads
# MAGIC   COMMENT 'Landing zone for raw telemetry CSV files';

# COMMAND ----------
# MAGIC %md ## Column Mask & Row Filter

# COMMAND ----------
# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION volvo_poc.silver.mask_driver_id(driver_id_hashed STRING)
# MAGIC   RETURN CASE WHEN is_member('admins') THEN driver_id_hashed ELSE '*** MASKED ***' END;
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION volvo_poc.gold.filter_by_region(depot_region STRING)
# MAGIC   RETURN is_member('admins') OR is_member('data_engineers');

# COMMAND ----------
# MAGIC %md ## Grants

# COMMAND ----------
# MAGIC %sql
# MAGIC GRANT USE CATALOG ON CATALOG volvo_poc TO `account users`;
# MAGIC GRANT USE SCHEMA ON SCHEMA volvo_poc.bronze TO `account users`;
# MAGIC GRANT USE SCHEMA, SELECT ON SCHEMA volvo_poc.silver TO `account users`;
# MAGIC GRANT EXECUTE ON FUNCTION volvo_poc.silver.mask_driver_id TO `account users`;
# MAGIC GRANT USE SCHEMA, SELECT ON SCHEMA volvo_poc.gold TO `account users`;
# MAGIC GRANT EXECUTE ON FUNCTION volvo_poc.gold.filter_by_region TO `account users`;
# MAGIC GRANT USE SCHEMA, SELECT ON SCHEMA volvo_poc.ml TO `account users`;

# COMMAND ----------
print("Unity Catalog setup complete.")
