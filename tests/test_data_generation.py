"""Unit tests for synthetic data generator."""
import math
import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generation.generate_synthetic_data import (
    compute_failure_probability, BUS_MODELS, generate_model_data,
    validate_dataframe, ROWS_PER_MODEL,
)


class TestFailureProbability:
    def test_high_risk_bus_has_high_probability(self, high_risk_row):
        prob = compute_failure_probability(high_risk_row)
        assert prob > 0.5, f"High-risk bus should have prob > 0.5, got {prob:.3f}"

    def test_low_risk_bus_has_low_probability(self, low_risk_row):
        prob = compute_failure_probability(low_risk_row)
        assert prob < 0.3, f"Low-risk bus should have prob < 0.3, got {prob:.3f}"

    def test_probability_bounded_zero_one(self, sample_bus_row, high_risk_row, low_risk_row):
        for row in [sample_bus_row, high_risk_row, low_risk_row]:
            p = compute_failure_probability(row)
            assert 0.0 <= p <= 1.0, f"Probability out of [0,1]: {p}"

    def test_high_brake_wear_increases_risk(self):
        base = {"odometer_km": 100_000, "brake_wear_pct": 10, "error_code_count": 0,
                "dpf_pressure_kpa": 1, "engine_temp_c": 85, "vibration_ms2": 1,
                "oil_pressure_bar": 4, "battery_voltage_v": 24}
        worn = {**base, "brake_wear_pct": 90}
        assert compute_failure_probability(worn) > compute_failure_probability(base)

    def test_high_error_codes_increases_risk(self):
        base = {"odometer_km": 100_000, "brake_wear_pct": 30, "error_code_count": 0,
                "dpf_pressure_kpa": 2, "engine_temp_c": 85, "vibration_ms2": 1,
                "oil_pressure_bar": 4, "battery_voltage_v": 24}
        faulty = {**base, "error_code_count": 7}
        assert compute_failure_probability(faulty) > compute_failure_probability(base)


class TestBusModelSpecs:
    def test_ten_bus_models_defined(self):
        assert len(BUS_MODELS) == 10

    def test_all_models_have_unique_names(self):
        names = [m.name for m in BUS_MODELS]
        assert len(names) == len(set(names))

    def test_electric_bus_has_zero_fuel_offset(self):
        electric = next(m for m in BUS_MODELS if m.engine_type == "electric")
        assert electric.fuel_rate_offset == 0.0

    def test_electric_bus_has_no_dpf(self):
        electric = next(m for m in BUS_MODELS if m.engine_type == "electric")
        assert not electric.dpf_fitted


class TestDataGeneration:
    def test_row_count(self):
        spec = BUS_MODELS[0]
        df = generate_model_data(spec)
        assert len(df) == ROWS_PER_MODEL

    def test_no_nulls_in_key_columns(self):
        spec = BUS_MODELS[1]
        df = generate_model_data(spec)
        for col in ["record_id", "bus_id", "bus_model", "next_14_days_failure"]:
            assert df[col].isnull().sum() == 0, f"Nulls found in {col}"

    def test_unique_record_ids(self):
        spec = BUS_MODELS[2]
        df = generate_model_data(spec)
        assert df["record_id"].nunique() == len(df)

    def test_target_binary(self):
        spec = BUS_MODELS[3]
        df = generate_model_data(spec)
        assert set(df["next_14_days_failure"].unique()).issubset({0, 1})

    def test_feature_ranges(self):
        spec = BUS_MODELS[0]
        df = generate_model_data(spec)
        assert (df["brake_wear_pct"].between(0, 100)).all()
        assert (df["speed_kph"] >= 0).all()
        assert (df["battery_voltage_v"].between(18, 32)).all()

    def test_validation_passes(self):
        spec = BUS_MODELS[4]
        df = generate_model_data(spec)
        assert validate_dataframe(df, spec.name)
