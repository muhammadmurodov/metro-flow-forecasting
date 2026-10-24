# MetroFlow Intelligence 🚇

An **English / O‘zbekcha / Русский** interactive research dashboard for 15-minute passenger entry/exit forecasts on the **Hangzhou (HZMetro)** network. Built as an AI/ML capstone and extended into an operator-oriented **historical replay demo**.

> The current model is fitted on Hangzhou records dated 1–25 January 2019. The app uses historical sample data.

## Features

- 15-, 30-, and 60-minute forecasts for station inflow and outflow.
- Correct, station-isolated lag features and recursive multi-step inference.
- Interactive Plotly network: click a station to focus it, see its name highlighted and explore with zoom or pan. The station table is selectable and synchronized with the map.
- Demand-based marker sizes and collision-aware placement at the default map view. This is a *topological* diagram, not a geographic map.
- Three prediction modes: inflow, outflow, or both together.
- Relative demand categories from station-specific training-data percentiles, not invented capacity limits.
- Approximate forecast bands from the 90th-percentile absolute validation residual. Multi-step intervals widen heuristically, without a coverage guarantee.
- Test-set model performance vs a historical-average baseline, CSV export, privacy notes, and multilingual UI.
- Automated regression tests for lag station boundaries, timestamps, history gaps and recursion.

### Using the interactive network

1. Choose **Entries**, **Exits**, or **Entries and exits** in the sidebar.
2. Choose a historical date, last observed interval, and forecast horizon.
3. Open **Interactive network**. Marker size reflects predicted passenger count.
   Marker color reflects demand relative to that station's typical history.
4. Click a station circle or select its row in the table. The same station is
   highlighted on both views and in the station selector. Zoom and pan using
   Plotly's toolbar.

The map's collision spacing is designed for the default desktop view. When
zoomed far out or squeezed into a narrow display, small visual overlaps can
occur. Zoom in to inspect tightly grouped parts of the network.

## Quick start

Use Python 3.12. From the project directory:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Packaged models and processed counts make the demo run without downloading the 1.3 GB raw AFC records. Dates in the UI are **historical replay** dates.

## Corrected results (Hangzhou, 15-minute one-step forecasts)

| Target | Validation MAE | Test MAE | Test baseline MAE | Test WAPE |
|---|---:|---:|---:|---:|
| Inflow | 24.31 | 28.06 | 40.37 | 12.70% |
| Outflow | 22.69 | 25.74 | 41.63 | 11.74% |

The corrected inflow test MAE is about **30.5% below** the historical-average baseline. The nominal validation-residual 90th-percentile error band covered only about **86.3%** of held-out inflow test cases; do not call it a calibrated 90% prediction interval. Exact metrics: `reports/evaluation.json`.

## Data and reproducibility

HZMetro original dataset: [Zenodo DOI 10.5281/zenodo.3145404](https://doi.org/10.5281/zenodo.3145404), CC BY 4.0. Raw AFC records are not bundled; the canonical bundled source is `data/processed/flow_15min_split.csv`.

Chronological split: train Jan 1–18, validation Jan 19–20, test Jan 21–25. The feature builder requires uninterrupted 15-minute history. Lags and rolling averages never borrow values from another station.

```bash
python -m src.train       # rebuild features, models, and evaluation JSON
pip install -r requirements-dev.txt
python -m pytest -q       # regression tests
```

### Key files

- `app.py`: Streamlit interface and interactive map.
- `src/features.py`: leakage-safe station features and future-input construction.
- `src/forecast.py`: recursive forecasts and relative demand categories.
- `src/train.py`: deterministic training and reporting.
- `src/i18n.py`: English, Uzbek and Russian interface copy.
- `tests/`: regression tests.
- `reports/evaluation.json`: authoritative corrected results.
- `docs/`: architecture, pilot concept and responsible-AI limitations.
- `notebooks/`: **legacy capstone notebooks** retained for teaching context. Their saved metrics/outputs are outdated. Use `python -m src.train` and `reports/evaluation.json` as the authoritative pipeline.

## Limits

15-minute entries/exits are **not simultaneous platform occupancy**. Green/amber/red categories do not represent actual engineering capacity or crowd safety thresholds. No live ingestion or train/gate control is implemented. See [Responsible AI](docs/RESPONSIBLE_AI_AND_LIMITATIONS.md).

## Licensing

HZMetro data is attributed to the original release under CC BY 4.0. No explicit software license has been granted by this repository.
