"""
Synthetic telemetry data generator for Volvo Bus Predictive Maintenance POC.

Generates 100,000 records per bus model (10 models = 1,000,000 total rows).
Failure label is derived from a logistic function of risk factors — not random —
ensuring real predictive signal for the ML model.
"""

import math
import os
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
rng = np.random.default_rng(42)

ROWS_PER_MODEL = 100_000
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

DEPOT_REGIONS = ["Gothenburg", "Stockholm", "Malmö", "Oslo", "Copenhagen",
                 "Helsinki", "Tampere", "Bergen", "Aarhus", "Turku"]

ROUTE_PREFIXES = ["GT", "ST", "ML", "OS", "CP", "HE", "TP", "BG", "AR", "TK"]


@dataclass
class BusModelSpec:
    """Per-model distribution parameters reflecting real engineering specs."""
    name: str
    engine_type: str           # diesel | hybrid | electric
    odometer_median_km: float  # typical fleet odometer
    fuel_rate_offset: float    # multiplier vs baseline diesel (1.0)
    vibration_offset: float    # additive offset to log-normal vibration
    max_speed_kph: float       # operational speed cap
    battery_nominal_v: float   # 24V for diesel/hybrid, 600V+ for BEV (normalised)
    dpf_fitted: bool           # DPF absent on pure electric
    num_buses: int             # unique bus IDs for this model


BUS_MODELS: list[BusModelSpec] = [
    BusModelSpec("Volvo_7900_Electric", "electric",  80_000, 0.00, -0.3, 80,  24.5, False, 40),
    BusModelSpec("Volvo_7700",          "diesel",   150_000, 1.00,  0.0, 90,  24.5, True,  40),
    BusModelSpec("Volvo_8900",          "diesel",   200_000, 1.05,  0.1, 100, 24.5, True,  40),
    BusModelSpec("Volvo_9700",          "diesel",   250_000, 1.08,  0.2, 120, 24.5, True,  40),
    BusModelSpec("Volvo_9900",          "diesel",   300_000, 1.12,  0.15,120, 24.5, True,  40),
    BusModelSpec("Volvo_B5LH",          "hybrid",   180_000, 0.70,  0.0, 90,  24.5, True,  40),
    BusModelSpec("Volvo_B7R",           "diesel",   220_000, 1.03,  0.1, 110, 24.5, True,  40),
    BusModelSpec("Volvo_B8R",           "diesel",   240_000, 1.06,  0.15,110, 24.5, True,  40),
    BusModelSpec("Volvo_B11R",          "diesel",   400_000, 1.15,  0.25,120, 24.5, True,  40),
    BusModelSpec("Volvo_B12B",          "diesel",   500_000, 1.20,  0.3, 120, 24.5, True,  40),
]


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_failure_probability(row: dict) -> float:
    """
    Logistic combination of risk factors.
    Calibrated so healthy buses yield ~5% base failure rate;
    worn/stressed buses can reach 80-90%.
    """
    score = 0.0
    score += 2.5  * (row["brake_wear_pct"] / 100.0)
    score += 1.8  * min(row["error_code_count"] / 5.0, 1.0)
    score += 1.2  * min(row["dpf_pressure_kpa"] / 12.0, 1.0)
    score += 1.0  * min(row["odometer_km"] / 800_000.0, 1.0)
    score += 0.8  * max(0.0, (row["engine_temp_c"] - 100.0) / 20.0)
    score += 0.6  * max(0.0, (row["vibration_ms2"] - 2.0) / 3.0)
    score -= 0.5  * min(row["oil_pressure_bar"] / 7.0, 1.0)
    score -= 0.3  * min(row["battery_voltage_v"] / 27.5, 1.0)
    return _logistic(score - 4.4)   # offset keeps fleet-wide rate ~6-18%


def generate_bus_ids(spec: BusModelSpec) -> list[str]:
    tag = spec.name.replace("Volvo_", "VOL-").replace("_", "")[:10]
    return [f"{tag}-{i:04d}" for i in range(1, spec.num_buses + 1)]


def generate_model_data(spec: BusModelSpec) -> pd.DataFrame:
    n = ROWS_PER_MODEL
    bus_ids = generate_bus_ids(spec)

    # ── timestamps: spread over 2 years ──────────────────────────────────────
    start_ts = datetime(2024, 1, 1)
    seconds_range = int(timedelta(days=730).total_seconds())
    timestamps = [
        start_ts + timedelta(seconds=int(s))
        for s in rng.integers(0, seconds_range, n)
    ]

    # ── odometer ─────────────────────────────────────────────────────────────
    # Log-normal anchored to median; add some per-bus-id variation
    bus_id_col = rng.choice(bus_ids, n)
    bus_offsets = {bid: rng.uniform(0.7, 1.3) for bid in bus_ids}
    base_odo = rng.lognormal(
        mean=math.log(spec.odometer_median_km),
        sigma=0.55,
        size=n
    )
    odometer_km = np.clip(
        base_odo * np.array([bus_offsets[b] for b in bus_id_col]),
        1_000, 900_000
    )

    # ── speed ─────────────────────────────────────────────────────────────────
    speed_kph = np.clip(
        rng.beta(2, 3, n) * spec.max_speed_kph,
        0, spec.max_speed_kph
    )

    # ── oil pressure ─────────────────────────────────────────────────────────
    # Electric buses have no oil pressure (set to 0); diesel range 2-7 bar
    if spec.engine_type == "electric":
        oil_pressure_bar = np.zeros(n)
    else:
        oil_pressure_bar = np.clip(rng.normal(4.2, 0.6, n), 1.5, 8.5)

    # ── battery voltage ──────────────────────────────────────────────────────
    battery_voltage_v = np.clip(
        rng.normal(spec.battery_nominal_v, 0.8, n), 20.0, 29.0
    )

    # ── vibration ─────────────────────────────────────────────────────────────
    vibration_ms2 = np.clip(
        rng.lognormal(mean=0.5 + spec.vibration_offset, sigma=0.4, size=n),
        0.0, 6.0
    )

    # ── ambient temperature (Nordic climate) ─────────────────────────────────
    month = np.array([ts.month for ts in timestamps])
    temp_seasonal_mean = -5 + 20 * np.sin((month - 3) * math.pi / 6)
    ambient_temp_c = np.clip(
        rng.normal(0, 1, n) * 8 + temp_seasonal_mean,
        -28, 42
    )

    # ── fuel rate ────────────────────────────────────────────────────────────
    if spec.engine_type == "electric":
        fuel_rate_lph = np.zeros(n)
    else:
        base_fuel = rng.lognormal(mean=2.8, sigma=0.3, size=n)
        fuel_rate_lph = np.clip(base_fuel * spec.fuel_rate_offset, 5.0, 80.0)

    # ── brake wear ────────────────────────────────────────────────────────────
    # Correlated with odometer: higher mileage → higher wear
    wear_base = rng.beta(2, 5, n) * 100
    odo_factor = np.clip(odometer_km / 500_000, 0, 1)
    brake_wear_pct = np.clip(wear_base + odo_factor * 40, 0, 100)

    # ── DPF pressure ─────────────────────────────────────────────────────────
    if not spec.dpf_fitted:
        dpf_pressure_kpa = np.zeros(n)
    else:
        dpf_pressure_kpa = np.clip(
            rng.lognormal(mean=1.2, sigma=0.5, size=n), 0.1, 18.0
        )

    # ── error codes ──────────────────────────────────────────────────────────
    # Poisson; spikes when brake wear or DPF pressure is high
    base_lambda = 0.8 + (brake_wear_pct / 100) * 1.5 + (dpf_pressure_kpa / 18) * 1.2
    error_code_count = rng.poisson(lam=np.clip(base_lambda, 0.1, 8.0)).astype(int)

    # ── engine temperature ────────────────────────────────────────────────────
    if spec.engine_type == "electric":
        engine_temp_c = np.clip(rng.normal(55, 8, n), 30, 90)
    else:
        # Rises with load (speed + ambient temp)
        base_temp = 87 + (speed_kph / spec.max_speed_kph) * 8 + (ambient_temp_c / 42) * 5
        engine_temp_c = np.clip(rng.normal(0, 6, n) + base_temp, 55, 125)

    # ── driver / route / region ───────────────────────────────────────────────
    region_col = rng.choice(DEPOT_REGIONS, n)
    route_prefix = rng.choice(ROUTE_PREFIXES, n)
    route_ids = [f"{pfx}-{rng.integers(1, 200):03d}" for pfx in route_prefix]
    driver_ids = [f"DRV-{fake.bothify('??###??').upper()}" for _ in range(n)]
    record_ids = [str(uuid.uuid4()) for _ in range(n)]

    # ── build row dict for failure label ─────────────────────────────────────
    rows = []
    for i in range(n):
        row = {
            "odometer_km":      odometer_km[i],
            "brake_wear_pct":   brake_wear_pct[i],
            "error_code_count": int(error_code_count[i]),
            "dpf_pressure_kpa": dpf_pressure_kpa[i],
            "engine_temp_c":    engine_temp_c[i],
            "vibration_ms2":    vibration_ms2[i],
            "oil_pressure_bar": oil_pressure_bar[i],
            "battery_voltage_v":battery_voltage_v[i],
        }
        rows.append(row)

    failure_labels = [
        int(random.random() < compute_failure_probability(r)) for r in rows
    ]

    df = pd.DataFrame({
        "record_id":          record_ids,
        "bus_id":             bus_id_col,
        "bus_model":          spec.name,
        "engine_type":        spec.engine_type,
        "event_timestamp":    timestamps,
        "odometer_km":        np.round(odometer_km, 1),
        "speed_kph":          np.round(speed_kph, 2),
        "oil_pressure_bar":   np.round(oil_pressure_bar, 3),
        "battery_voltage_v":  np.round(battery_voltage_v, 3),
        "vibration_ms2":      np.round(vibration_ms2, 4),
        "ambient_temp_c":     np.round(ambient_temp_c, 2),
        "fuel_rate_lph":      np.round(fuel_rate_lph, 3),
        "brake_wear_pct":     np.round(brake_wear_pct, 2),
        "dpf_pressure_kpa":   np.round(dpf_pressure_kpa, 4),
        "error_code_count":   error_code_count,
        "engine_temp_c":      np.round(engine_temp_c, 2),
        "driver_id":          driver_ids,
        "route_id":           route_ids,
        "depot_region":       region_col,
        "next_14_days_failure": failure_labels,
    })

    return df


def validate_dataframe(df: pd.DataFrame, model_name: str) -> bool:
    ok = True
    n = len(df)

    if n != ROWS_PER_MODEL:
        print(f"  [FAIL] {model_name}: expected {ROWS_PER_MODEL} rows, got {n}")
        ok = False

    failure_rate = df["next_14_days_failure"].mean()
    if not (0.04 <= failure_rate <= 0.25):
        print(f"  [WARN] {model_name}: failure rate {failure_rate:.2%} outside expected 4-20%")

    if df["record_id"].nunique() != n:
        print(f"  [FAIL] {model_name}: duplicate record_ids detected")
        ok = False

    required_no_nulls = ["record_id", "bus_id", "bus_model", "next_14_days_failure"]
    for col in required_no_nulls:
        if df[col].isnull().any():
            print(f"  [FAIL] {model_name}: nulls in {col}")
            ok = False

    assert (df["brake_wear_pct"].between(0, 100)).all(), f"{model_name}: brake_wear_pct out of range"
    assert (df["speed_kph"] >= 0).all(), f"{model_name}: negative speed"

    return ok


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_start = time.time()
    total_rows = 0

    print(f"\nGenerating synthetic telemetry — {len(BUS_MODELS)} models × {ROWS_PER_MODEL:,} rows")
    print("=" * 65)

    summary = []
    for spec in BUS_MODELS:
        t0 = time.time()
        print(f"\n  {spec.name} ({spec.engine_type})")

        df = generate_model_data(spec)
        valid = validate_dataframe(df, spec.name)

        fname = os.path.join(OUTPUT_DIR, f"bus_telemetry_{spec.name}.csv")
        df.to_csv(fname, index=False)

        elapsed = time.time() - t0
        failure_rate = df["next_14_days_failure"].mean()
        total_rows += len(df)

        status = "OK" if valid else "WARN"
        print(f"  [{status}] {len(df):,} rows | failure rate: {failure_rate:.2%} | {elapsed:.1f}s → {fname}")
        summary.append({
            "model": spec.name,
            "rows": len(df),
            "failure_rate": f"{failure_rate:.2%}",
            "status": status,
        })

    elapsed_total = time.time() - total_start
    print("\n" + "=" * 65)
    print(f"Done: {total_rows:,} total rows in {elapsed_total:.1f}s")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}\n")

    # Summary table
    print(f"{'Model':<30} {'Rows':>8} {'Failure%':>10} {'Status':>6}")
    print("-" * 60)
    for s in summary:
        print(f"{s['model']:<30} {s['rows']:>8,} {s['failure_rate']:>10} {s['status']:>6}")
    print()


if __name__ == "__main__":
    main()
