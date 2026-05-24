-- ============================================================
-- Column Masking Functions — PII Protection
-- ============================================================
-- driver_id_hashed is masked for all users except data_stewards group.
-- Applied to: volvo_poc.silver.bus_telemetry_curated

USE CATALOG volvo_poc;

CREATE OR REPLACE FUNCTION volvo_poc.silver.mask_driver_id(driver_id_hashed STRING)
  RETURN CASE
    WHEN is_member('data_stewards') OR is_member('admins') THEN driver_id_hashed
    ELSE '*** MASKED ***'
  END;

-- failure_probability visible to all; bus_id never masked (not personal data)
