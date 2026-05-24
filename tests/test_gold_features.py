"""Unit tests for Gold layer feature engineering logic."""
import pytest
import pandas as pd
import numpy as np


def compute_brake_wear_index(brake_wear_pct: float) -> float:
    return round(brake_wear_pct / 100.0, 4)


def compute_composite_risk_score(
    brake_wear_pct: float,
    error_code_count: int,
    dpf_pressure_kpa: float,
    odometer_km: float,
    oil_pressure_anomaly: bool,
    engine_temp_anomaly: bool,
) -> float:
    score = (
        (brake_wear_pct / 100.0) * 0.35 +
        min(error_code_count / 5.0, 1.0) * 0.25 +
        min(dpf_pressure_kpa / 15.0, 1.0) * 0.20 +
        min(odometer_km / 800_000.0, 1.0) * 0.10 +
        int(oil_pressure_anomaly) * 0.05 +
        int(engine_temp_anomaly) * 0.05
    )
    return round(min(score, 1.0), 4)


def apply_risk_tier(prob: float) -> str:
    if prob >= 0.65:
        return "HIGH"
    elif prob >= 0.40:
        return "MEDIUM"
    return "LOW"


class TestBrakeWearIndex:
    def test_zero_wear(self):
        assert compute_brake_wear_index(0) == 0.0

    def test_full_wear(self):
        assert compute_brake_wear_index(100) == 1.0

    def test_half_wear(self):
        assert compute_brake_wear_index(50) == 0.5

    def test_index_bounded(self):
        for pct in [0, 25, 50, 75, 100]:
            idx = compute_brake_wear_index(pct)
            assert 0.0 <= idx <= 1.0


class TestCompositeRiskScore:
    def test_perfect_health_near_zero(self):
        score = compute_composite_risk_score(0, 0, 0, 0, False, False)
        assert score == 0.0

    def test_max_risk_capped_at_one(self):
        score = compute_composite_risk_score(100, 99, 99, 999_999, True, True)
        assert score == 1.0

    def test_brake_wear_dominant(self):
        high_brake = compute_composite_risk_score(90, 0, 0, 0, False, False)
        low_brake  = compute_composite_risk_score(10, 0, 0, 0, False, False)
        assert high_brake > low_brake

    def test_score_monotone_with_error_codes(self):
        s1 = compute_composite_risk_score(50, 0, 3, 200_000, False, False)
        s2 = compute_composite_risk_score(50, 5, 3, 200_000, False, False)
        assert s2 > s1

    def test_no_future_leakage_possible(self):
        # next_14_days_failure must NOT be a parameter
        import inspect
        sig = inspect.signature(compute_composite_risk_score)
        assert "next_14_days_failure" not in sig.parameters


class TestRiskTiers:
    def test_high_tier(self):
        assert apply_risk_tier(0.65) == "HIGH"
        assert apply_risk_tier(0.9) == "HIGH"

    def test_medium_tier(self):
        assert apply_risk_tier(0.40) == "MEDIUM"
        assert apply_risk_tier(0.64) == "MEDIUM"

    def test_low_tier(self):
        assert apply_risk_tier(0.0) == "LOW"
        assert apply_risk_tier(0.39) == "LOW"

    def test_all_buses_have_tier(self):
        probs = np.linspace(0, 1, 100)
        tiers = [apply_risk_tier(p) for p in probs]
        assert all(t in ("HIGH", "MEDIUM", "LOW") for t in tiers)
