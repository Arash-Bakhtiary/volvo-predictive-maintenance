# Databricks notebook source
# MAGIC %md
# MAGIC # Genie Space Setup — Volvo Fleet Maintenance Q&A
# MAGIC
# MAGIC Run this notebook once in your Databricks workspace to create the Genie space.
# MAGIC Genie spaces cannot be created fully via API from scratch — use the UI wizard
# MAGIC with the queries and instructions below.
# MAGIC
# MAGIC ## Steps
# MAGIC 1. Open your workspace → **New** → **Genie Space**
# MAGIC 2. Set title: **Volvo Fleet Maintenance Q&A**
# MAGIC 3. Select warehouse: **Serverless Starter Warehouse**
# MAGIC 4. Add the datasets and instructions below

# COMMAND ----------
# MAGIC %md
# MAGIC ## Datasets to Add
# MAGIC
# MAGIC **Dataset 1: Fleet Failure Scores**
# MAGIC ```sql
# MAGIC SELECT bus_id, bus_model, depot_region, failure_probability,
# MAGIC        risk_tier, recommended_action, scored_at
# MAGIC FROM volvo_poc.gold.fleet_failure_scores
# MAGIC ORDER BY failure_probability DESC
# MAGIC ```
# MAGIC
# MAGIC **Dataset 2: Maintenance Candidates**
# MAGIC ```sql
# MAGIC SELECT * FROM volvo_poc.gold.maintenance_candidates
# MAGIC ORDER BY failure_probability DESC
# MAGIC ```
# MAGIC
# MAGIC **Dataset 3: Bus Features**
# MAGIC ```sql
# MAGIC SELECT bus_id, bus_model, depot_region,
# MAGIC        odometer_km, brake_wear_pct, dpf_pressure_kpa,
# MAGIC        error_code_count, engine_temp_c, composite_risk_score,
# MAGIC        next_14_days_failure
# MAGIC FROM volvo_poc.gold.bus_features
# MAGIC ```

# COMMAND ----------
# MAGIC %md
# MAGIC ## Instructions for Genie
# MAGIC
# MAGIC Paste this into the Genie space **Instructions** field:
# MAGIC
# MAGIC ```
# MAGIC You are a Volvo bus fleet maintenance analyst assistant.
# MAGIC
# MAGIC Key definitions:
# MAGIC - failure_probability: ML-predicted probability (0.0–1.0) of a bus failing within 14 days
# MAGIC - risk_tier: HIGH (≥0.65), MEDIUM (0.40–0.65), LOW (<0.40)
# MAGIC - recommended_action: IMMEDIATE (HIGH risk), SCHEDULE (MEDIUM), MONITOR (LOW)
# MAGIC - composite_risk_score: weighted score combining brake wear, DPF pressure, error codes, odometer
# MAGIC
# MAGIC Always:
# MAGIC - Order results by failure_probability DESC when ranking buses
# MAGIC - Include bus_model and depot_region in results
# MAGIC - Show failure_probability as a percentage when possible
# MAGIC ```

# COMMAND ----------
# MAGIC %md
# MAGIC ## Sample Questions to Seed
# MAGIC
# MAGIC Add these as sample questions in the Genie space:
# MAGIC
# MAGIC 1. Which buses have the highest failure probability today?
# MAGIC 2. How many HIGH risk buses are in each depot region?
# MAGIC 3. Show all Volvo B12B buses needing immediate maintenance
# MAGIC 4. What is the average failure probability by bus model?
# MAGIC 5. List buses with brake_wear_pct above 80 and HIGH risk tier
# MAGIC 6. Which depot region has the most at-risk buses?
# MAGIC 7. How does failure probability correlate with odometer reading?

# COMMAND ----------
# Verify the tables are accessible from this notebook
display(spark.sql("SELECT risk_tier, COUNT(*) as cnt, ROUND(AVG(failure_probability),3) as avg_prob FROM volvo_poc.gold.fleet_failure_scores GROUP BY risk_tier ORDER BY avg_prob DESC"))
