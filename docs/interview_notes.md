# Explaining the project in an interview

## Short explanation

I built a Python backtest for Great Britain's half-hourly electricity demand using official NESO data. I checked data availability and daylight-saving timestamps before modelling. A ridge model using calendar features and delayed demand lags reduced 2025 MAE by 24.46% compared with a 28-day seasonal baseline. The uncertainty band undercovered the test year, which I documented rather than claiming it was reliable at its nominal level.

Use that explanation only after running and understanding the code yourself. The repository was developed with AI assistance; all reported figures are from execution on the downloaded data, not invented examples.

## Questions to practise

- **Why not use yesterday's demand?** The selected public historic feed is normally published 21 days late. Yesterday's final demand is unavailable from this source at forecast time. I used a 28-day minimum lag and documented the remaining revision risk.
- **Why ridge?** It is fast, inspectable and a useful first improvement over seasonal rules. Its L2 penalty stabilises correlated lag/calendar features. It is not necessarily the best possible model.
- **What does 24.46% mean?** `(2806.784 - 2120.191) / 2806.784 * 100`, the relative reduction in half-hourly MAE against the stronger of two baselines on 2025.
- **What does MAE tell you?** Average absolute interval error in MW. It does not reveal the worst peak misses, so I also retained predictions and monthly errors and reported RMSE.
- **What went wrong?** Nominal 90% bands achieved 84.83% coverage on the test year; several peaks were underpredicted. Weather, holidays and temporal drift warrant further investigation.
- **What are the time splits?** Train: 2022–2023 after lag warm-up; validation: 2024; test: 2025. Scaling is fitted only on training data. The test model is not refitted using validation or test targets.
- **What do you mean by production-ready?** I do not claim that. This is a reproducible historical evaluation and local dashboard. A service would need publication-vintage data, inference, scheduling and monitoring.

## Suggested CV entry

**GB Electricity Demand Forecasting | Python, pandas, NumPy, time-series validation**

- Built a reproducible forecasting pipeline using 70,128 half-hourly NESO observations, with data-quality checks and daylight-saving-aware timestamp processing.
- Reduced held-out 2025 MAE by 24.5% against a seasonal baseline using regularised regression; evaluated 17,520 intervals and built an interactive results dashboard.

Add the actual published repository URL when available. Keep the CV to two bullets and be ready to explain the publication-delay assumption.
