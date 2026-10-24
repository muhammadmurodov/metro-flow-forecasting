"""Recursive multi-step forecasts and defensible demand labels."""

import numpy as np
import pandas as pd
from .features import INTERVAL, features_for_future


def forecast_steps(artifact: dict, station: int, as_of: pd.Timestamp,
                   observed: list[float], steps: int = 4) -> list[float]:
    """Forecast future 15-minute counts, using own predictions after step one."""
    if not 1 <= steps <= 4:
        raise ValueError("Supported horizons are 15 to 60 minutes")
    history = [float(x) for x in observed]
    if len(history) < 4 or not np.isfinite(history).all():
        raise ValueError("Four valid observations are required")
    target = artifact["target"]
    predictions = []
    for step in range(1, steps + 1):
        row = features_for_future(station, pd.Timestamp(as_of) + step * INTERVAL,
                                  history, target)
        value = max(0., float(artifact["model"].predict(row[artifact["features"]])[0]))
        history.append(value)
        predictions.append(value)
    return predictions


def interval_for_prediction(prediction: float, calibration_error: float,
                            step: int = 1) -> tuple[float, float]:
    """Heuristic widening for recursive forecasts, NOT multi-step calibrated coverage."""
    spread = float(calibration_error) * np.sqrt(step)
    return max(0., prediction - spread), prediction + spread


def demand_label(value: float, low: float, high: float) -> str:
    """Relative demand, not platform occupancy or safety capacity."""
    return "high" if value >= high else "medium" if value >= low else "low"
