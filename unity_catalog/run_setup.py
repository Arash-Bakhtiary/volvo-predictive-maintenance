"""Execute Unity Catalog governance setup against Databricks SQL warehouse."""
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

WAREHOUSE_ID = "5b1ca4f1d21b522a"

# All statements in execution order
STATEMENTS = [
    # --- Catalog & Schemas ---
    "CREATE CATALOG IF NOT EXISTS volvo_poc COMMENT 'Volvo Bus Predictive Maintenance POC'",
    "CREATE SCHEMA IF NOT EXISTS volvo_poc.bronze COMMENT 'Raw ingestion layer — full fidelity, PII present'",
    "CREATE SCHEMA IF NOT EXISTS volvo_poc.silver COMMENT 'Curated layer — cleansed, PII masked'",
    "CREATE SCHEMA IF NOT EXISTS volvo_poc.gold   COMMENT 'Feature and prediction layer — business-ready'",
    "CREATE SCHEMA IF NOT EXISTS volvo_poc.ml     COMMENT 'MLflow experiments and registered models'",

    # --- Column Mask ---
    """CREATE OR REPLACE FUNCTION volvo_poc.silver.mask_driver_id(driver_id_hashed STRING)
       RETURN CASE
         WHEN is_member('admins') THEN driver_id_hashed
         ELSE '*** MASKED ***'
       END""",

    # --- Row Filter ---
    """CREATE OR REPLACE FUNCTION volvo_poc.gold.filter_by_region(depot_region STRING)
       RETURN is_member('admins') OR is_member('data_engineers')""",

    # --- Grants ---
    "GRANT USE CATALOG ON CATALOG volvo_poc TO `account users`",
    "GRANT USE SCHEMA ON SCHEMA volvo_poc.bronze TO `account users`",
    "GRANT USE SCHEMA, SELECT ON SCHEMA volvo_poc.silver TO `account users`",
    "GRANT EXECUTE ON FUNCTION volvo_poc.silver.mask_driver_id TO `account users`",
    "GRANT USE SCHEMA, SELECT ON SCHEMA volvo_poc.gold TO `account users`",
    "GRANT EXECUTE ON FUNCTION volvo_poc.gold.filter_by_region TO `account users`",
    "GRANT USE SCHEMA, SELECT ON SCHEMA volvo_poc.ml TO `account users`",
]


def run_statement(client: WorkspaceClient, sql: str, label: str) -> bool:
    short = sql.strip().split("\n")[0][:80]
    print(f"  → {short}...", end=" ", flush=True)
    response = client.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=sql,
        wait_timeout="30s",
    )
    state = response.status.state
    if state == StatementState.SUCCEEDED:
        print("OK")
        return True
    elif state in (StatementState.PENDING, StatementState.RUNNING):
        # Poll until done
        stmt_id = response.statement_id
        for _ in range(30):
            time.sleep(2)
            result = client.statement_execution.get_statement(stmt_id)
            state = result.status.state
            if state == StatementState.SUCCEEDED:
                print("OK")
                return True
            if state == StatementState.FAILED:
                err = result.status.error
                print(f"FAILED — {err.message}")
                return False
        print("TIMEOUT")
        return False
    else:
        err = response.status.error
        print(f"FAILED — {err.message if err else state}")
        return False


def main():
    client = WorkspaceClient()
    print(f"\nUnity Catalog Setup — {len(STATEMENTS)} statements\n")
    passed, failed = 0, 0
    for i, sql in enumerate(STATEMENTS, 1):
        label = f"[{i}/{len(STATEMENTS)}]"
        ok = run_statement(client, sql, label)
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*50}")
    print(f"Done: {passed} succeeded, {failed} failed")
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
