"""Shared fixtures for all test modules."""
import math
import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(scope="session")
def sample_bus_row():
    return {
        "odometer_km": 250_000,
        "brake_wear_pct": 75.0,
        "error_code_count": 2,
        "dpf_pressure_kpa": 6.0,
        "engine_temp_c": 92.0,
        "vibration_ms2": 1.5,
        "oil_pressure_bar": 4.0,
        "battery_voltage_v": 24.5,
    }


@pytest.fixture(scope="session")
def high_risk_row():
    return {
        "odometer_km": 780_000,
        "brake_wear_pct": 95.0,
        "error_code_count": 6,
        "dpf_pressure_kpa": 14.0,
        "engine_temp_c": 110.0,
        "vibration_ms2": 4.0,
        "oil_pressure_bar": 1.8,
        "battery_voltage_v": 21.5,
    }


@pytest.fixture(scope="session")
def low_risk_row():
    return {
        "odometer_km": 30_000,
        "brake_wear_pct": 10.0,
        "error_code_count": 0,
        "dpf_pressure_kpa": 1.0,
        "engine_temp_c": 82.0,
        "vibration_ms2": 0.5,
        "oil_pressure_bar": 4.5,
        "battery_voltage_v": 24.8,
    }
