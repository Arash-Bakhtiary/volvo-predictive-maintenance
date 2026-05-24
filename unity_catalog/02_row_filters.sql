-- ============================================================
-- Row-Level Security Filters
-- ============================================================
-- Analysts only see buses from their assigned depot_region.
-- Admins see all rows.
-- Applied to: volvo_poc.gold.fleet_failure_scores

USE CATALOG volvo_poc;

CREATE OR REPLACE FUNCTION volvo_poc.gold.filter_by_region(depot_region STRING)
  RETURN is_member('admins')
      OR is_member('data_engineers')
      OR depot_region = CURRENT_USER();
