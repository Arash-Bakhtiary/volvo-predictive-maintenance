-- ============================================================
-- Unity Catalog Setup — Volvo POC
-- ============================================================

-- Catalog
CREATE CATALOG IF NOT EXISTS volvo_poc
  COMMENT 'Volvo Bus Predictive Maintenance POC — Databricks Free Edition';

USE CATALOG volvo_poc;

-- Schemas (Medallion layers)
CREATE SCHEMA IF NOT EXISTS volvo_poc.bronze
  COMMENT 'Raw ingestion layer — full fidelity, no transformations, PII present';

CREATE SCHEMA IF NOT EXISTS volvo_poc.silver
  COMMENT 'Curated layer — cleansed, validated, PII hashed and masked';

CREATE SCHEMA IF NOT EXISTS volvo_poc.gold
  COMMENT 'Feature and prediction layer — business-ready, row-filtered';

CREATE SCHEMA IF NOT EXISTS volvo_poc.ml
  COMMENT 'MLflow experiments and registered model versions';
