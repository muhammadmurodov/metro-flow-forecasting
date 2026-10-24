"""Reproduce corrected features, models, calibration, and evaluation.

Run from repository root: python -m src.train
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from .features import make_features, feature_columns

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/processed/flow_15min_split.csv"
FEATURES = ROOT / "data/processed/flow_features.csv"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"


def metrics(y, pred):
    return {"mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred)))}


def main():
    raw = pd.read_csv(RAW, parse_dates=["interval"])
    frame = make_features(raw)
    frame.to_csv(FEATURES, index=False)
    ARTIFACTS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    report = {"data": "HZMetro Hangzhou, Jan 2019; not Tashkent",
              "splits": frame.groupby("split").size().to_dict(), "targets": {}}
    train, val, test = (frame[frame.split == split].copy()
                        for split in ("train", "val", "test"))
    for target in ("inflow", "outflow"):
        cols = feature_columns(target)
        model = XGBRegressor(n_estimators=280, max_depth=6, learning_rate=0.065,
                             subsample=0.9, colsample_bytree=0.9,
                             objective="reg:squarederror", n_jobs=4, random_state=42)
        model.fit(train[cols], train[target])
        val_pred = np.maximum(0, model.predict(val[cols]))
        test_pred = np.maximum(0, model.predict(test[cols]))
        val_error = np.abs(val[target].to_numpy() - val_pred)
        # Validation-residual quantile is suitable for a first-step, marginal
        # interval heuristic. Coverage on unseen test data is also reported.
        q90 = float(np.quantile(val_error, 0.90))
        coverage = float(np.mean(np.abs(test[target].to_numpy() - test_pred) <= q90))
        baseline_table = train.groupby(["stationID", "hour", "is_weekend"])[target].mean()
        fallback = float(train[target].mean())
        index = pd.MultiIndex.from_frame(test[["stationID", "hour", "is_weekend"]])
        baseline = baseline_table.reindex(index).fillna(fallback).to_numpy()
        # Station-specific percentiles are relative demand indicators only.
        thresholds = train.groupby("stationID")[target].quantile([0.50, 0.90]).unstack()
        thresholds.columns = ["medium", "high"]
        threshold_dict = {str(int(sid)): {"medium": float(r.medium),
                                          "high": float(r.high)}
                          for sid, r in thresholds.iterrows()}
        artifact = {"model": model, "features": cols, "target": target,
                    "calibration_error_p90": q90,
                    "station_thresholds": threshold_dict,
                    "source": "Hangzhou HZMetro, January 2019"}
        filename = "metro_xgb_model.joblib" if target == "inflow" else "metro_xgb_outflow_model.joblib"
        joblib.dump(artifact, ARTIFACTS / filename)
        report["targets"][target] = {
            "validation": metrics(val[target], val_pred),
            "test": metrics(test[target], test_pred),
            "test_baseline": metrics(test[target], baseline),
            "validation_abs_error_q90": q90,
            "test_interval_coverage_at_q90": coverage,
            "test_wape": float(np.abs(test[target] - test_pred).sum() / max(test[target].sum(), 1)),
        }
        tmp = test[["hour", target]].copy()
        tmp["error"] = np.abs(test[target].to_numpy() - test_pred)
        tmp["period"] = np.select([tmp.hour.between(7, 9), tmp.hour.between(17, 19)],
                                   ["morning", "evening"], default="offpeak")
        report["targets"][target]["peak_test_mae"] = {
            key: float(group.error.mean()) for key, group in tmp.groupby("period")}
        print(target, report["targets"][target])
    (REPORTS / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved corrected features, artifacts and evaluation.json")


if __name__ == "__main__":
    main()
