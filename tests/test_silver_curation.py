"""Unit tests for Silver layer curation logic."""
import hashlib
import pytest
import pandas as pd


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def apply_silver_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror of the Silver SQL transforms for unit testing."""
    df = df.copy()
    # Outlier capping
    caps = {
        "odometer_km": (1_000, 900_000), "speed_kph": (0, 200),
        "oil_pressure_bar": (0, 10), "battery_voltage_v": (18, 32),
        "brake_wear_pct": (0, 100), "engine_temp_c": (40, 130),
    }
    for col, (lo, hi) in caps.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    # PII: hash driver_id, drop original
    if "driver_id" in df.columns:
        df["driver_id_hashed"] = df["driver_id"].apply(
            lambda x: sha256(str(x)) if pd.notnull(x) else None
        )
        df = df.drop(columns=["driver_id"])

    # Anomaly flags
    df["oil_pressure_anomaly"] = (
        (df["oil_pressure_bar"] < 2.0) | (df["oil_pressure_bar"] > 7.5)
    )
    df["engine_temp_anomaly"] = (
        (df["engine_temp_c"] < 60) | (df["engine_temp_c"] > 105)
    )
    df["battery_anomaly"] = (
        (df["battery_voltage_v"] < 22.0) | (df["battery_voltage_v"] > 27.5)
    )
    df["brake_critical_flag"] = df["brake_wear_pct"] > 80

    # Quality flag
    df["is_valid_record"] = (
        df["bus_id"].notnull() &
        df["speed_kph"].between(0, 200) &
        df["oil_pressure_bar"].between(0, 10) &
        df["battery_voltage_v"].between(18, 32) &
        df["brake_wear_pct"].between(0, 100) &
        df["next_14_days_failure"].isin([0, 1]) &
        df["engine_temp_c"].between(40, 130)
    )
    return df


@pytest.fixture
def clean_df():
    return pd.DataFrame([{
        "bus_id": "VOL-7700-0001", "bus_model": "Volvo_7700",
        "engine_type": "diesel", "depot_region": "Stockholm",
        "driver_id": "DRV-AB123CD",
        "odometer_km": 200_000.0, "speed_kph": 60.0,
        "oil_pressure_bar": 4.0, "battery_voltage_v": 24.5,
        "vibration_ms2": 1.5, "engine_temp_c": 88.0,
        "brake_wear_pct": 45.0, "next_14_days_failure": 0,
    }])


class TestPIIMasking:
    def test_driver_id_removed(self, clean_df):
        silver = apply_silver_transforms(clean_df)
        assert "driver_id" not in silver.columns

    def test_driver_id_hashed_present(self, clean_df):
        silver = apply_silver_transforms(clean_df)
        assert "driver_id_hashed" in silver.columns

    def test_hash_is_deterministic(self, clean_df):
        s1 = apply_silver_transforms(clean_df)["driver_id_hashed"].iloc[0]
        s2 = apply_silver_transforms(clean_df)["driver_id_hashed"].iloc[0]
        assert s1 == s2

    def test_hash_is_sha256_length(self, clean_df):
        silver = apply_silver_transforms(clean_df)
        assert len(silver["driver_id_hashed"].iloc[0]) == 64


class TestOutlierCapping:
    def test_speed_capped_at_200(self):
        df = pd.DataFrame([{"bus_id": "X", "bus_model": "M", "engine_type": "diesel",
                             "depot_region": "A", "driver_id": "D", "odometer_km": 50_000,
                             "speed_kph": 999, "oil_pressure_bar": 4, "battery_voltage_v": 24,
                             "vibration_ms2": 1, "engine_temp_c": 88, "brake_wear_pct": 30,
                             "next_14_days_failure": 0}])
        silver = apply_silver_transforms(df)
        assert silver["speed_kph"].iloc[0] == 200

    def test_brake_wear_capped_at_100(self):
        df = pd.DataFrame([{"bus_id": "X", "bus_model": "M", "engine_type": "diesel",
                             "depot_region": "A", "driver_id": "D", "odometer_km": 50_000,
                             "speed_kph": 60, "oil_pressure_bar": 4, "battery_voltage_v": 24,
                             "vibration_ms2": 1, "engine_temp_c": 88, "brake_wear_pct": 150,
                             "next_14_days_failure": 0}])
        silver = apply_silver_transforms(df)
        assert silver["brake_wear_pct"].iloc[0] == 100


class TestAnomalyFlags:
    def test_low_oil_pressure_flagged(self, clean_df):
        df = clean_df.copy()
        df["oil_pressure_bar"] = 1.5
        silver = apply_silver_transforms(df)
        assert silver["oil_pressure_anomaly"].iloc[0] is True or silver["oil_pressure_anomaly"].iloc[0] == True

    def test_normal_oil_pressure_not_flagged(self, clean_df):
        silver = apply_silver_transforms(clean_df)
        assert not silver["oil_pressure_anomaly"].iloc[0]

    def test_high_brake_wear_flagged(self, clean_df):
        df = clean_df.copy()
        df["brake_wear_pct"] = 85
        silver = apply_silver_transforms(df)
        assert silver["brake_critical_flag"].iloc[0]

    def test_quality_flag_valid_record(self, clean_df):
        silver = apply_silver_transforms(clean_df)
        assert silver["is_valid_record"].iloc[0]
