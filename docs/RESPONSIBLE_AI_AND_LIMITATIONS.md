# Responsible AI and limitations

1. **Source** — Hangzhou HZMetro counts from 1–25 January 2019.
2. **Privacy** — bundled data holds aggregated 15-minute station counts, not identity, card ID, biometric data or individual trips. Future raw imports require appropriate controls and retention policies.
3. **Forecast semantics** — first horizon uses four consecutive observed intervals, 30–60 minutes recursively use model predictions. Longer-horizon uncertainty widening is a heuristic and has not been calibrated.
4. **Intervals** — calibration is the validation absolute-error 90th percentile, which produced only ~86.3% one-step test coverage on inflow. Do not assert a guaranteed 90% prediction interval.
5. **Demand versus safety** — relative demand labels are derived from historical station count distributions, not platform occupancy, capacity, passenger density or safety thresholds.
6. **Fairness and reliability** — forecast error differs by station and time; morning peaks are harder. Only 25 days are included, without yearly seasonality, most holidays or disruption coverage.
7. **Human responsibility** — advisory-only signals. Trains, gates, staffing and safety decisions require authorized operators and independent live confirmation. No automatic control.
8. **Transfer limitations** — new networks need local authorized data, retraining, chronological testing, calibration and shadow-mode evaluation. Never present these Hangzhou predictions as local forecasts.
