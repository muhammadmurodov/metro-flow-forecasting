# Model Gate Evidence — Metro Flow Forecasting

> **Archived capstone report (pre-correction).** Numbers below are historical
> and no longer reflect the retrained artifacts. See `evaluation.json` for
> corrected, reproducible results from `python -m src.train`.

## 1. Baseline
Historical Average (mean inflow per station × hour × weekend-flag).
Validation MAE: 29.34 | Test MAE: 37.55

## 2. Controlled runs (≥2)
Three models compared under identical split and features:
- Historical Average — val MAE 29.34
- Ridge regression — val MAE 29.29
- XGBoost — val MAE 22.42
All logged to MLflow (experiment: metro-flow-forecasting, 3 runs).

## 3. Split & metric protocol
Chronological split, no shuffling:
- Train: Jan 1–18 | Validation: Jan 19–20 | Test: Jan 21–25
Metrics: MAE (primary), RMSE (secondary). Scalers fit on train only.

## 4. Candidate selection
Selected on VALIDATION MAE (not test). XGBoost chosen (22.42, lowest).

## 5. Protected test status
Test set touched once, after selection. Test MAE: 25.75.

## 6. Baseline comparison
XGBoost beats Historical Average by ~24% on validation MAE (22.42 vs 29.34)
and on test MAE (25.75 vs 37.55).

## 6a. Extension: Outflow model
A symmetric XGBoost model was trained to predict outflow (using outflow lags):
val MAE 21.23, test MAE 23.79. Both models are served in the deployed Streamlit
app (https://metro-flow-forecasting.streamlit.app) with an inflow/outflow/both
selector. Outflow is slightly more predictable than inflow, consistent with
exits being driven by scheduled train arrivals.

## 7. Error analysis
Error broken down on the test set:
- By time: morning peak hardest (MAE ~46), evening peak ~36, off-peak ~19.
  Normalized by volume the error rate is fairly consistent (~11–16%).
- By station: busy stations higher absolute MAE (~34) but quiet stations
  have the highest *relative* error (~16%), which is why MAPE was reported
  with caution for low-count intervals.
See notebook 04_error_analysis.ipynb.
## 8. Inference artifact
artifacts/metro_xgb_model.joblib — bundles model + feature list.

## 9. Reload proof
Model reloaded from disk reproduces identical predictions without refitting
(np.allclose = True). See notebook 03.

## 10. Known limitations
- 25 days: no seasonal coverage.
- Hangzhou model does not transfer directly to Tashkent.
