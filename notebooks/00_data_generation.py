# Databricks notebook source
# MAGIC %md
# MAGIC # Step 0 — Synthetic Data Generation
# MAGIC Generates 1,000,000 rows of bus telemetry (10 Volvo models × 100,000).
# MAGIC Uploads CSVs to the UC Volume for Bronze ingestion.

# COMMAND ----------
dbutils.widgets.text("catalog", "volvo_poc")
catalog = dbutils.widgets.get("catalog")

# COMMAND ----------
# MAGIC %md ## Install dependencies and generate data

# COMMAND ----------
# MAGIC %pip install faker==40.19.1

# COMMAND ----------
import math, random, uuid, os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
rng  = np.random.default_rng(42)

ROWS_PER_MODEL = 100_000
VOLUME_PATH    = f"/Volumes/{catalog}/bronze/raw_uploads"

DEPOT_REGIONS   = ["Gothenburg","Stockholm","Malmö","Oslo","Copenhagen","Helsinki","Tampere","Bergen","Aarhus","Turku"]
ROUTE_PREFIXES  = ["GT","ST","ML","OS","CP","HE","TP","BG","AR","TK"]

BUS_MODELS = [
    ("Volvo_7900_Electric","electric",  80_000, 0.00,-0.3, 80, False),
    ("Volvo_7700",         "diesel",   150_000, 1.00, 0.0, 90, True),
    ("Volvo_8900",         "diesel",   200_000, 1.05, 0.1,100, True),
    ("Volvo_9700",         "diesel",   250_000, 1.08, 0.2,120, True),
    ("Volvo_9900",         "diesel",   300_000, 1.12,0.15,120, True),
    ("Volvo_B5LH",         "hybrid",   180_000, 0.70, 0.0, 90, True),
    ("Volvo_B7R",          "diesel",   220_000, 1.03, 0.1,110, True),
    ("Volvo_B8R",          "diesel",   240_000, 1.06,0.15,110, True),
    ("Volvo_B11R",         "diesel",   400_000, 1.15,0.25,120, True),
    ("Volvo_B12B",         "diesel",   500_000, 1.20, 0.3,120, True),
]

def logistic(x): return 1/(1+math.exp(-x))

def failure_prob(row):
    s  = 2.5*(row["brake_wear_pct"]/100)
    s += 1.8*min(row["error_code_count"]/5,1)
    s += 1.2*min(row["dpf_pressure_kpa"]/12,1)
    s += 1.0*min(row["odometer_km"]/800000,1)
    s += 0.8*max(0,(row["engine_temp_c"]-100)/20)
    s += 0.6*max(0,(row["vibration_ms2"]-2)/3)
    s -= 0.5*min(row["oil_pressure_bar"]/7,1)
    s -= 0.3*min(row["battery_voltage_v"]/27.5,1)
    return logistic(s-4.4)

all_dfs = []
for name,etype,odo_med,fuel_off,vib_off,max_spd,dpf in BUS_MODELS:
    n  = ROWS_PER_MODEL
    ids = [f"VOL-{name.replace('Volvo_','')[:6]}-{i:04d}" for i in range(1,41)]
    bus_id = rng.choice(ids, n)
    start  = datetime(2024,1,1)
    ts     = [start+timedelta(seconds=int(s)) for s in rng.integers(0,2*365*86400,n)]
    odo    = np.clip(rng.lognormal(math.log(odo_med),0.55,n)*rng.uniform(0.7,1.3,n),1000,900000)
    spd    = np.clip(rng.beta(2,3,n)*max_spd,0,max_spd)
    oil    = np.zeros(n) if etype=="electric" else np.clip(rng.normal(4.2,0.6,n),1.5,8.5)
    bat    = np.clip(rng.normal(24.5,0.8,n),20,29)
    vib    = np.clip(rng.lognormal(0.5+vib_off,0.4,n),0,6)
    month  = np.array([t.month for t in ts])
    tmp    = np.clip(rng.normal(0,1,n)*8+(-5+20*np.sin((month-3)*math.pi/6)),-28,42)
    fuel   = np.zeros(n) if etype=="electric" else np.clip(rng.lognormal(2.8,0.3,n)*fuel_off,5,80)
    wear   = np.clip(rng.beta(2,5,n)*100+np.clip(odo/500000,0,1)*40,0,100)
    dpf_p  = np.zeros(n) if not dpf else np.clip(rng.lognormal(1.2,0.5,n),0.1,18)
    lam    = np.clip(0.8+(wear/100)*1.5+(dpf_p/18)*1.2,0.1,8)
    errs   = rng.poisson(lam).astype(int)
    etemp  = np.clip(rng.normal(55,8,n),30,90) if etype=="electric" else np.clip(rng.normal(0,6,n)+87+(spd/max_spd)*8,55,125)
    labels = [int(random.random()<failure_prob({"brake_wear_pct":wear[i],"error_code_count":int(errs[i]),"dpf_pressure_kpa":dpf_p[i],"odometer_km":odo[i],"engine_temp_c":etemp[i],"vibration_ms2":vib[i],"oil_pressure_bar":oil[i],"battery_voltage_v":bat[i]})) for i in range(n)]
    df = pd.DataFrame({
        "record_id":[str(uuid.uuid4()) for _ in range(n)],
        "bus_id":bus_id,"bus_model":name,"engine_type":etype,
        "event_timestamp":ts,
        "odometer_km":np.round(odo,1),"speed_kph":np.round(spd,2),
        "oil_pressure_bar":np.round(oil,3),"battery_voltage_v":np.round(bat,3),
        "vibration_ms2":np.round(vib,4),"ambient_temp_c":np.round(tmp,2),
        "fuel_rate_lph":np.round(fuel,3),"brake_wear_pct":np.round(wear,2),
        "dpf_pressure_kpa":np.round(dpf_p,4),"error_code_count":errs,
        "engine_temp_c":np.round(etemp,2),
        "driver_id":[f"DRV-{fake.bothify('??###??').upper()}" for _ in range(n)],
        "route_id":[f"{rng.choice(ROUTE_PREFIXES)}-{rng.integers(1,200):03d}" for _ in range(n)],
        "depot_region":rng.choice(DEPOT_REGIONS,n),
        "next_14_days_failure":labels,
    })
    rate = sum(labels)/n
    print(f"  {name}: {n:,} rows | failure rate: {rate:.2%}")
    all_dfs.append(df)

full_df = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal: {len(full_df):,} rows | Fleet failure rate: {full_df['next_14_days_failure'].mean():.2%}")

# COMMAND ----------
# MAGIC %md ## Write to UC Volume (one CSV per model)

# COMMAND ----------
for model_name in full_df["bus_model"].unique():
    mdf  = full_df[full_df["bus_model"]==model_name]
    path = f"{VOLUME_PATH}/bus_telemetry_{model_name}.csv"
    mdf.to_csv(path, index=False)
    print(f"  Written: {path} ({len(mdf):,} rows)")

print("\nData generation complete — files ready in UC Volume.")
