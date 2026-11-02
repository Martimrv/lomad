# LOMAD: Local Anomaly Detection in Maritime Trajectories Using LSTM Prediction and Visual Analytics

Test

Maritime surveillance operators face an ever-growing volume of Automatic Identification System (AIS) data and must decide, in near real time, which vessel behaviours warrant investigation. Fully automated detectors are rarely deployed without human oversight given incomplete data, missing ground-truth labels, and the evolving nature of anomalous patterns. We present LOMAD, an interactive visual-analytics tool that combines per-vessel LSTM trajectory prediction with multi-layered anomaly scoring---trajectory-based, prediction-based, and contextual---to detect \textit{local anomalies}: deviations confined to a portion of a vessel's track. Scores are surfaced through coordinated geographic, temporal, and explanatory views, allowing operators to rank vessels by severity, inspect supporting evidence, and decide whether flagged behaviour is genuinely suspicious or a data artefact. We demonstrate LOMAD on real AIS traffic and show how the integration of three complementary detectors with template-based explanations facilitates intuitive and interpretable exploration of vessel behaviour.

## Prerequisites

- Python 3.12+
- Node.js 20.15.1+ (for the React dashboard)
- npm 10.7.0+

## Installation

### Quick start (Windows)

A virtual environment named `venv` must already exist and be activated by the script. Run one of:

```bat
:: cmd
setup.bat
```

```powershell
# PowerShell
.\setup.ps1
```

### Manual setup

1. **Create + activate a virtual environment:**
   - macOS / Linux:
     ```bash
     python3 -m venv venv && source venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     python -m venv venv; .\venv\Scripts\Activate.ps1
     ```
   - Windows (cmd):
     ```bat
     python -m venv venv && venv\Scripts\activate.bat
     ```

2. **Install Python dependencies:**
   - macOS:
     ```bash
     pip install -r requirements_mac.txt
     ```
   - Windows / Linux:
     ```bash
     pip install -r requirements.txt
     ```

3. **Install Node.js dependencies:**
   ```bash
   cd dashboard-app
   npm install
   cd ..
   ```

## Running the application

The pipeline runs in five steps. The processed CSVs in `backend/data/` are committed so you can skip steps 1–3 and go straight to launching the API + dashboard if you only want to explore.

### 1. Preprocess raw AIS data

The preprocessor expects a raw AIS CSV at `backend/data/AIS_2022_03_31.csv`.

```bash
python -m backend.scripts.load_data
```

Output: `backend/data/cleaned_data.csv`.

### 2. Train the LSTM trajectory predictor

```bash
python backend/scripts/lstm_model.py
```

Outputs:
- `backend/data/vessel_position_predictions.csv` — per-row true vs predicted positions
- `backend/data/vessel_position_metrics.csv` — per-vessel error metrics

### 3. Generate anomaly results

Prediction-error anomalies (used by `/anomalies/queue` and `/anomalies/predictions`):

```bash
python backend/scripts/lstm_anomaly.py
```

Output: `backend/data/ais_anomaly_results.csv`.

Optional richer anomaly layers — run individually or via the runner script:

```bash
python run_detector.py
# or
python backend/scripts/trajectory_anomaly_detector.py
python backend/scripts/advanced_anomaly_detector.py
```

Outputs:
- `backend/data/trajectory_anomalies.csv`
- `backend/data/trajectory_anomaly_metrics.csv`
- `backend/data/advanced_anomalies.csv`
- `backend/data/anomaly_events.csv`

### 4. Start the FastAPI backend

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

### 5. Start the React dashboard (new terminal)

```bash
cd dashboard-app
npm start
```

### URLs

- API: http://127.0.0.1:8000/
- Interactive API docs: http://127.0.0.1:8000/docs
- React dashboard: http://localhost:3000/

## Configuration

Two paths can be overridden via environment variables:

| Variable                  | Default                                         |
| ------------------------- | ----------------------------------------------- |
| `PREPROCESSED_DATA_PATH`  | `backend/data/cleaned_data.csv`                 |
| `ANOMALY_MODEL_PATH`      | `backend/model/anomaly_detector.joblib`         |

## Data processing pipeline (`backend/scripts/load_data.py`)

- Renames raw AIS columns (`MMSI`, `BaseDateTime`, `LAT`, `LON`, `SOG`, `COG`, `Heading`) to lowercase.
- Validates timestamps and keeps only numeric MMSI values.
- Clips outliers in `sog`, `cog`, `heading`, `latitude`, `longitude` via quantile bounds (avoids excessive row loss).
- Sorts per vessel chronologically.
- Writes the cleaned dataset to `backend/data/cleaned_data.csv` (unscaled — normalization happens inside the model script).

## Trajectory prediction model (`backend/scripts/lstm_model.py`)

LSTM with **delta prediction** (predicting movement deltas rather than absolute positions, which improved position accuracy by ~10×).

- Input features: `sog`, `cog`, `heading`, `latitude`, `longitude` (sequence of 15 timesteps)
- Stack: `LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(32) + BatchNorm → Dense(16) → [Dense(4) deltas, Dense(1, σ) anomaly score]`
- One model per vessel (skips vessels with <100 samples)
- 80/20 train/test split with EarlyStopping + ReduceLROnPlateau
- Anomaly flag at the 95th percentile of combined position+motion error

See [backend/scripts/README.md](backend/scripts/README.md) for full architecture, metrics, and extension notes.

## Anomaly detection layers

| Layer | Script | Output | Approach |
| --- | --- | --- | --- |
| Prediction error | `lstm_model.py`, `lstm_anomaly.py` | `ais_anomaly_results.csv` | Residuals between LSTM-predicted and observed positions; threshold at mean + 3σ (or 95th percentile in `lstm_model.py`). |
| Trajectory rules | `trajectory_anomaly_detector.py` | `trajectory_anomalies.csv`, `trajectory_anomaly_metrics.csv` | Direct rule-based detection: speed/course step changes, loitering, drift, zigzag, traffic deviation. |
| Advanced contextual | `advanced_anomaly_detector.py` | `advanced_anomalies.csv`, `anomaly_events.csv` | Combines all signals with temporal/contextual features and event clustering. |

## API endpoints

### Health & status

- `GET /` — welcome message and endpoint index
- `GET /health` — service health and dataset availability
- `GET /models/status` — generated dataset and detector artifact availability
- `GET /anomalies/status` — anomaly model availability and latest training metrics

### Data

- `GET /data/preprocessed?limit&offset` — paginated rows from `cleaned_data.csv` (max `limit=1000`)
- `GET /data/sample?n` — first `n` rows (max 100)
- `GET /data/stats` — record counts and per-column numeric stats

### Trajectories

- `GET /trajectories?ship_type&segment_id&mmsi&limit&include_coordinates` — trajectories grouped by `segment_id`
- `GET /trajectories/mmsi?limit` — list of MMSIs with sample counts

### Predictions

- `GET /predictions/vessels?limit_per_vessel` — true vs predicted coordinates per vessel
- `GET /predictions/metrics?top_n` — per-vessel MSE and anomaly counts

### Anomalies

- `GET /anomalies/queue?limit` — recent alert summaries derived from `ais_anomaly_results.csv`
- `GET /anomalies/predictions?limit` — per-row anomaly prediction points
- `GET /trajectory-anomalies/vessels?limit_per_vessel`
- `GET /trajectory-anomalies/metrics`
- `GET /trajectory-anomalies/types` — catalog of trajectory anomaly types
- `GET /advanced-anomalies/vessels?limit_per_vessel`
- `GET /advanced-anomalies/events?limit`
- `GET /advanced-anomalies/summary`

### Explanations

- `GET /vessel/{mmsi}/explanation` — natural-language explanation of prediction-error anomalies
- `GET /vessel/{mmsi}/trajectory-explanation` — natural-language explanation of trajectory anomalies

## Tech stack

### Backend
- **FastAPI** — web framework
- **Uvicorn** — ASGI server
- **Pandas / NumPy** — data manipulation
- **scikit-learn** — preprocessing
- **TensorFlow / Keras** — LSTM model

### Frontend
- **React 19** + **Create React App**
- **react-leaflet** + **Leaflet** — interactive map
- **Recharts** — charts
- **axios** — HTTP client

## Frontend dashboard

- **Map view**: vessel trajectories on OpenStreetMap, color-coded by ship type, with start/end markers when a trajectory is selected.
- **Charts**: SOG / COG / heading timelines, combined normalized features view.
- **Filters**: ship type, MMSI, segment.
- **Anomaly views**: trajectory anomalies, advanced events, and natural-language explanations per vessel.
- Backend polling for live data; click a trajectory to drill in.

Components: `App.js`, `components/TrajectoryViewer.js`, `components/TrajectoryMap.js`, `components/AnomalyDashboard.js`.
