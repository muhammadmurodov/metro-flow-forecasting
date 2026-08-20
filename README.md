# Metro Passenger Flow Forecasting

**🔗 Live demo:** https://metro-flow-forecasting.streamlit.app
**▶️ Run the demo notebook:** [`notebooks/05_demo.ipynb`](notebooks/05_demo.ipynb) — [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/muhammadmurodov/metro-flow-forecasting/blob/main/notebooks/05_demo.ipynb) (station + time → forecast, runs top-to-bottom from a clean runtime)

Short-term (15-minute) metro passenger inflow forecasting for station-level
crowd management. AI/ML Fundamentals Capstone.

Given a station and its recent flow history, the model predicts how many
passengers will enter that station in the next 15-minute interval — giving
operators lead time to act before congestion forms.

## Results
| Model | Target | Validation MAE | Test MAE |
|---|---|---|---|
| Historical Average (baseline) | inflow | 29.34 | 37.55 |
| Ridge regression | inflow | 29.29 | 39.29 |
| **XGBoost (selected)** | inflow | **22.42** | **25.75** |
| XGBoost | outflow | 21.23 | 23.79 |

XGBoost beats the baseline by ~24% on validation MAE. A symmetric outflow model
(same pipeline, outflow lags) was also trained; outflow predicts slightly better,
likely because exits are driven by scheduled train arrivals.

## Data
HZMetro (Hangzhou Metro), Zenodo DOI 10.5281/zenodo.3145404, CC-BY 4.0.
Raw AFC tap records, 80 stations, 1–25 January 2019.

**Raw data is not included in this repo** (1.3 GB). To reproduce from scratch:
1. Download from https://doi.org/10.5281/zenodo.3145404
2. Place the `record_2019-01-*.csv` files in `data/raw/`
3. Run `notebooks/01_aggregate.ipynb`

The aggregated/processed files needed for modeling are included in
`data/processed/`, so you can skip straight to modeling if preferred.

## How to run
```bash
python3 -m venv metro
source metro/bin/activate
pip install -r requirements.txt
```
Then run the notebooks in order:
1. `01_aggregate.ipynb` — raw AFC → 15-min inflow/outflow per station
2. `02_eda.ipynb` — EDA, leakage-safe split, feature engineering
3. `03_models.ipynb` — baseline, Ridge, XGBoost, MLflow tracking, saved artifact
4. `04_error_analysis.ipynb` — error by peak/off-peak and station busyness
5. `05_demo.ipynb` — clean input→output demo: station + timestamp → forecast + crowding band

## Method
- **Task:** supervised time-series regression (next-interval inflow)
- **Split:** chronological — train Jan 1–18, val Jan 19–20, test Jan 21–25 (no shuffling)
- **Features:** calendar (hour, weekday, weekend) + lags (1–4 intervals) + rolling mean
- **Leakage control:** scalers fit on train only; lags computed within station using only past intervals; test touched once
- **Tracking:** MLflow (3 logged runs)

## Repository structure
data/processed/ aggregated 15-min flow tables
notebooks/ pipeline notebooks (01–05, incl. demo)
artifacts/ saved model + loading docs
reports/ Model Gate evidence + figures
docs/ responsible-AI write-up
presentation/ defense slides

## Responsible AI & Limitations
Full write-up: [`docs/RESPONSIBLE_AI_AND_LIMITATIONS.md`](docs/RESPONSIBLE_AI_AND_LIMITATIONS.md).

- **Advisory only:** supports human dispatch decisions; does not control trains or gates.
- **Privacy:** trained on aggregate 15-min counts only — no passenger identities, trips, or
  biometric/personal data.
- **Fairness:** single city (Hangzhou); accuracy varies by station and time of day (quiet
  stations have the highest *relative* error), which matters for how the tool is used.
- **Limitations:** 25 days → no seasonality; does **not** transfer directly to Tashkent
  (different topology, ridership culture, calendar) — Tashkent is motivation, not a
  deployment claim; point forecasts only (no uncertainty surfaced).

## License
Code: for educational use. Data: CC-BY 4.0 (Hangzhou Metro / Tianchi).
