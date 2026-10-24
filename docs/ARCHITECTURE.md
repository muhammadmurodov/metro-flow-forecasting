# Architecture and runtime semantics

`flow_15min_split.csv` → `src.features.make_features` → `src.train` → two `joblib` artifacts + `reports/evaluation.json` → `app.py`.

For a selected last observed interval **t**, the first forecast targets **t+15 minutes**, with features calculated from known values at **t, t−15, t−30, t−45** and calendar fields of **t+15**. Longer horizons insert previously forecast counts as new lag values. The app refuses to predict if observations are not consecutive.

Models are retrained using only the training split. Validation residuals determine approximate one-step uncertainty bands. Test data are reserved for reporting. Nonnegative clipping keeps displayed passenger counts meaningful.

The network is a deterministic NetworkX spring layout rendered as interactive Plotly traces. It is **topological, not geographic**, and does not estimate occupancy or platform capacity.

Deployment: use pinned requirements and treat joblib files as trusted artifacts (pickle-style loading is unsafe for untrusted files). An operator-facing deployment would additionally require authentication, access controls, freshness monitoring and agreed data governance; none of those integrations is bundled.

Potential future work: authenticated real-time feed, station map coordinates, role-based access, holiday/event features, monitoring for distribution shift, and verified station-capacity models.
