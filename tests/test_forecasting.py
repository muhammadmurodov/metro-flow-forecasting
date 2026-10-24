import numpy as np
import pandas as pd
import pytest

from src.features import make_features, features_for_future
from src.forecast import forecast_steps, interval_for_prediction, demand_label


def sample():
    out = []
    for station, start in ((0, 10), (1, 100)):
        for i in range(10):
            out.append({"stationID": station,
                        "interval": pd.Timestamp("2019-01-01 08:00") + pd.Timedelta(minutes=15*i),
                        "inflow": start+i, "outflow": start+2*i, "split": "train"})
    return pd.DataFrame(out)


def test_lags_are_station_isolated():
    features = make_features(sample())
    a = features[features.stationID == 1].iloc[0]
    assert a.inflow == 104
    assert [a[f"inflow_lag_{i}"] for i in range(1,5)] == [103,102,101,100]
    assert a.inflow_roll_mean_4 == 101.5


def test_gap_is_not_mistaken_for_15_minutes():
    raw = sample()
    raw = raw[~((raw.stationID == 0) & (raw.interval == pd.Timestamp("2019-01-01 08:30")))]
    features = make_features(raw)
    assert pd.Timestamp("2019-01-01 08:45") not in set(features[features.stationID == 0].interval)


def test_future_features_have_correct_target_time():
    row = features_for_future(7, pd.Timestamp("2019-01-01 09:15"), [10,11,12,13], "inflow")
    assert row.loc[0, "minute"] == 15
    assert row.loc[0, "inflow_lag_1"] == 13
    assert row.loc[0, "inflow_roll_mean_4"] == 11.5


class LagOneModel:
    def predict(self, frame):
        return frame["inflow_lag_1"].to_numpy() + 1


def test_recursive_forecast_uses_latest_observation_then_prediction():
    artifact = {"target": "inflow", "model": LagOneModel(),
                "features": ["inflow_lag_1"]}
    values = forecast_steps(artifact, 0, pd.Timestamp("2019-01-01 09:00"), [2,3,4,5], 4)
    assert values == [6,7,8,9]


def test_no_negative_prediction_intervals():
    assert interval_for_prediction(5, 20)[0] == 0
    assert demand_label(25, 10, 20) == "high"
    with pytest.raises(ValueError):
        forecast_steps({}, 0, pd.Timestamp("2019-01-01"), [1,2,3,4], 5)
