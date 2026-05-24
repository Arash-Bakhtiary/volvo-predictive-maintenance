-- ============================================================
-- Grants — Least-Privilege Role Model
-- ============================================================

-- Catalog-level
GRANT USE CATALOG ON CATALOG volvo_poc TO `account users`;

-- Bronze: data engineers only (read); no analyst access (PII present)
GRANT USE SCHEMA ON SCHEMA volvo_poc.bronze TO `account users`;

-- Silver: read for all, but PII masked via column mask function
GRANT USE SCHEMA, SELECT ON SCHEMA volvo_poc.silver TO `account users`;
GRANT EXECUTE ON FUNCTION volvo_poc.silver.mask_driver_id TO `account users`;

-- Gold: read for all, row-filtered by region function
GRANT USE SCHEMA, SELECT ON SCHEMA volvo_poc.gold TO `account users`;
GRANT EXECUTE ON FUNCTION volvo_poc.gold.filter_by_region TO `account users`;

-- ML schema: read for all; write only for data engineers/admins
GRANT USE SCHEMA, SELECT ON SCHEMA volvo_poc.ml TO `account users`;
