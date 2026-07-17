# ServerGuard AI — Predictive Server Failure Detection

An end-to-end machine learning system that predicts server hardware failures
before they happen, using infrastructure telemetry (CPU, memory, disk,
network, temperature, power, and error signals). Built to reflect the kind
of predictive-maintenance problem enterprise infrastructure companies
(e.g. HPE, with products like HPE InfoSight) solve in production.

## Problem Statement

Modern data centers run thousands of servers. Unplanned hardware failures
cause downtime, data loss, and financial impact. This project predicts
**failure probability per server** from live telemetry, so ops teams can
act proactively (replace a disk, improve cooling, rebalance load) instead
of reactively.

## Architecture

```
Telemetry Data → Feature Engineering → Model Training (LogReg / RF / XGBoost)
     → Best Model Selected → Flask Dashboard → Real-Time Prediction
```

## Dataset

Real hardware-failure telemetry with the full feature set this project
needs (CPU, memory, network, temperature, power, errors) is not publicly
available in one place — public datasets like Backblaze's drive stats
cover disk SMART attributes only. To build a project that reflects the
complete monitoring picture, this project uses a **synthetic dataset
generated with deliberately engineered, documented correlations**
(`src/generate_data.py`): failure probability is a function of
temperature, error count, and resource saturation (CPU/memory), mirroring
the strongest real-world failure indicators. The generator produces
300 simulated servers × 40 time snapshots each, with a realistic ~10%
failure rate (imbalanced, as real failures are rare).

| Feature | Description |
|---|---|
| cpu_usage | % |
| memory_usage | % |
| disk_usage | % |
| disk_read_mbps / disk_write_mbps | MB/s |
| network_in_mbps / network_out_mbps | MB/s |
| temperature_c | °C |
| fan_speed_rpm | RPM |
| power_watts | Watts |
| error_count | count |
| uptime_hours | hours |
| failure | target (0/1) |

## Feature Engineering

- `disk_io_total`, `network_io_total` — combined activity signals
- `cpu_mem_pressure` — interaction term (CPU+memory saturating together is a stronger signal than either alone)
- `temp_error_interaction` — thermal stress + faults co-occurring
- `uptime_bucket` (one-hot) — non-linear effect of server age

**Time-series degradation features** (`src/trend_features.py`), added on top of
the single-snapshot features above, using each server's history of past
snapshots (never future ones, to avoid leakage):
- `temp_trend_slope`, `cpu_trend_slope` — short-window linear trend (is this server's temperature/CPU rising over time, not just high right now)
- `error_count_rolling_avg` — rolling average of recent errors
- `temp_delta`, `error_delta` — change since the previous snapshot

This was a direct response to the original single-snapshot model's biggest
gap: it had no notion of whether a server was *stable* or *degrading*.
Rolling/trend features were chosen over an LSTM deliberately — with only
40 snapshots × 300 servers, a sequence model would be overparameterized
and hard to validate honestly; trend features are the standard, more
defensible technique for this data volume in real predictive-maintenance
systems.

## Models & Results

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (class_weight=balanced) | 0.225 | 0.714 | 0.342 | 0.777 |
| Random Forest (tuned) | 0.255 | 0.625 | 0.362 | 0.773 |
| XGBoost (tuned) | 0.234 | 0.698 | 0.350 | **0.773** |

*(Numbers above include the time-series trend features. Without them,
ROC-AUC was ~0.774 for Random Forest — trend features did not materially
move aggregate ROC-AUC, discussed below.)*

**Random Forest** is the final model used in the dashboard, though all
three land within half a point of each other — this indicates the
feature set, not the algorithm choice, is the current performance
ceiling.

**Why recall matters more than precision here:** missing a real failure
(false negative) risks downtime; a false alarm just costs an unnecessary
check. The models are tuned to favor catching failures over minimizing
false alarms, via `class_weight="balanced"` / `scale_pos_weight`.

Class imbalance (~10% failure rate) is handled via class weighting rather
than SMOTE, to avoid overfitting to synthetic minority-class points.

**Honest note on the trend features:** `error_count_rolling_avg` ranks
3rd in feature importance (see below) — the model clearly uses it — but
it didn't meaningfully change aggregate ROC-AUC. This is because the
synthetic data generator conditions failure probability on a server's
*current* wear state, not explicitly on the trend leading up to it, so
much of the trend signal is redundant with features the model already
had. This is a real limitation of validating trend-based features on
synthetic data, and is exactly the kind of thing that would need
re-validating against real fleet telemetry before trusting it in
production.

## Explainability

Feature importance and SHAP values confirm `temp_error_interaction` and
`temperature_c` are the strongest predictors, with the new
`error_count_rolling_avg` trend feature ranking 3rd — consistent with
real-world hardware failure patterns (thermal stress and a worsening
error trend are leading indicators of hardware degradation).

See `reports/feature_importance.png` and `reports/shap_summary.png`.

## Project Structure

```
ServerFailurePrediction/
├── data/
│   ├── raw/                  # generated telemetry CSV
│   └── processed/            # train/test splits, trend-augmented data
├── src/
│   ├── generate_data.py      # synthetic data generator
│   ├── trend_features.py     # time-series degradation features (NEW)
│   ├── preprocess.py         # feature engineering + split
│   ├── train_baseline.py     # Logistic Regression baseline
│   ├── train_models.py       # Random Forest + XGBoost, tuned
│   ├── explain.py            # feature importance + SHAP
│   ├── evaluate_final.py     # ROC curve + confusion matrix
│   ├── prepare_dashboard_data.py  # fleet snapshot for the dashboard
│   └── predict.py            # CLI prediction helper
├── tests/
│   └── test_pipeline.py      # pytest suite (NEW): data, features, model sanity checks
├── models/                   # saved .pkl models
├── app/
│   ├── app.py                # Flask dashboard
│   ├── templates/            # HTML pages
│   └── static/               # CSS + result plots
├── reports/                  # generated plots + comparison table
├── Dockerfile                # containerized deployment (NEW)
├── .dockerignore
├── requirements.txt
└── README.md
```

## Running the Project

```bash
pip install -r requirements.txt

# 1. Generate data
python src/generate_data.py

# 2. Feature engineering (includes trend features) + split
python src/preprocess.py

# 3. Train baseline
python src/train_baseline.py

# 4. Train + tune Random Forest and XGBoost
python src/train_models.py

# 5. Explainability (feature importance + SHAP)
python src/explain.py

# 6. Final evaluation plots
python src/evaluate_final.py

# 7. Build the fleet snapshot for the dashboard
python src/prepare_dashboard_data.py

# 8. Run the dashboard
cd app && python app.py
# then open http://127.0.0.1:5000
```

### Running tests

```bash
pip install pytest  # already in requirements.txt
pytest tests/ -v
```

Covers: data generation shape/failure-rate sanity, trend-feature
correctness (no NaNs, no row-count drift), feature engineering not
leaking NaNs, and the trained model producing valid probability outputs
with the expected feature set.

### Running with Docker

```bash
# Run the full pipeline locally first (steps 1-7 above) so models/ and
# data/processed/ are populated, then:
docker build -t serverguard-ai .
docker run -p 5000:5000 serverguard-ai
```
Uses `gunicorn` instead of Flask's dev server inside the container.

## Dashboard Pages

- **Dashboard (`/`)** — fleet overview: total/healthy/critical servers, average health score, sortable server list by risk
- **Server detail (`/server/<id>`)** — live-style metrics, health score, recommended action
- **Predict (`/predict`)** — manual telemetry input → failure probability + risk level
- **Analytics (`/analytics`)** — ROC curve, confusion matrix, feature importance, correlation plots

## Future Improvements

- ~~Time-series modeling to capture degradation trends~~ — **partially
  addressed**: added rolling/trend features (`src/trend_features.py`);
  a full LSTM sequence model would need more snapshots per server than
  this synthetic dataset provides to validate honestly
- Real-time streaming ingestion (Kafka) instead of batch CSV
- Integrate real hardware telemetry (e.g. SMART attributes, IPMI sensor data) alongside the synthetic feature set — the honest next step to validate whether the trend features actually help on real degradation patterns
- Alerting integration (email/Slack) when a server crosses into HIGH risk
- CI pipeline (GitHub Actions) running `pytest` on every push

## Resume Bullet Points

- Developed an end-to-end machine learning application in Python to predict server failures using infrastructure telemetry such as CPU, memory, disk I/O, network traffic, temperature, and power metrics.
- Compared multiple classification models (Logistic Regression, Random Forest, XGBoost) and selected the best model using precision, recall, F1-score, and ROC-AUC, with explicit handling of class imbalance via class weighting.
- Built a Flask-based dashboard for real-time failure prediction, fleet-wide server health monitoring, and visualization of feature importance and model performance metrics.
