"""Leakage-safe station-level features for 15-minute target intervals."""

import numpy as np
import pandas as pd

INTERVAL = pd.Timedelta(minutes=15)


def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["interval"] = pd.to_datetime(frame["interval"])
    frame = frame.sort_values(["stationID", "interval"]).reset_index(drop=True)
    if frame.duplicated(["stationID", "interval"]).any():
        raise ValueError("Duplicate station/timestamp pairs")
    frame["hour"] = frame["interval"].dt.hour
    frame["minute"] = frame["interval"].dt.minute
    frame["dayofweek"] = frame["interval"].dt.dayofweek
    frame["is_weekend"] = (frame["dayofweek"] >= 5).astype(int)

    for target in ("inflow", "outflow"):
        grouped = frame.groupby("stationID", sort=False)[target]
        for lag in range(1, 5):
            frame[f"{target}_lag_{lag}"] = grouped.shift(lag)
        frame[f"{target}_roll_mean_4"] = frame[
            [f"{target}_lag_{lag}" for lag in range(1, 5)]
        ].mean(axis=1)

    # Do not treat an earlier row as the preceding 15 minutes across gaps.
    grouped_time = frame.groupby("stationID", sort=False)["interval"]
    contiguous = pd.Series(True, index=frame.index)
    for lag in range(1, 5):
        contiguous &= frame["interval"].sub(grouped_time.shift(lag)).eq(lag * INTERVAL)
    frame = frame.loc[contiguous].copy()
    return frame.reset_index(drop=True)


def feature_columns(target: str) -> list[str]:
    if target not in ("inflow", "outflow"):
        raise ValueError("Unknown target")
    return ["stationID", "hour", "minute", "dayofweek", "is_weekend",
            *[f"{target}_lag_{lag}" for lag in range(1, 5)],
            f"{target}_roll_mean_4"]


def features_for_future(station: int, target_time: pd.Timestamp,
                        recent: list[float], target: str) -> pd.DataFrame:
    """recent is chronological, ending with the last observed/predicted interval."""
    if len(recent) < 4:
        raise ValueError("Four consecutive previous intervals are required")
    ts = pd.Timestamp(target_time)
    values = list(reversed(recent[-4:]))
    row = {"stationID": station, "hour": ts.hour, "minute": ts.minute,
           "dayofweek": ts.dayofweek, "is_weekend": int(ts.dayofweek >= 5)}
    row.update({f"{target}_lag_{i}": float(v) for i, v in enumerate(values, start=1)})
    row[f"{target}_roll_mean_4"] = float(np.mean(values))
    return pd.DataFrame([row], columns=feature_columns(target))
