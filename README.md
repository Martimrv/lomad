# AIS LLM Boat Trajectory Prediction

A research project for analyzing and predicting vessel trajectories using AIS (Automatic Identification System) data and Large Language Models.

### Prerequisites

- Python 3.12+ (Apple Silicon / macOS)
- Node.js 20.15.1+ (for React dashboard)
- npm 10.7.0+ (for package management)
- Virtual environment activated

## Installation

### Manual Setup
1. **Navigate to the project directory:**
   ```bash
   cd "/Users/amsoaa/Documents/Linnaeus University/Research projects/AISLLM"
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements_mac.txt
   ```

4. **Install Node.js dependencies:**
   ```bash
   cd dashboard-app
   npm install
   cd ..
   ```

5. **Update requirements (after installing new packages):**
   ```bash
   pip freeze > requirements_mac.txt
   ```

### Running the Application

1. **Preprocess the AIS data:**
   ```bash
   python backend/scripts/preprocess_ais.py
   ```

2. **Train the trajectory prediction model (optional but recommended):**
   ```bash
   python backend/scripts/trajectory_predictor.py
   ```

3. **Start the FastAPI backend:**
   ```bash
   python -m uvicorn backend.main:app --reload --port 8000
   ```

4. **Start the React dashboard (in a new terminal):**
   ```bash
   cd dashboard-app
   npm start
   ```

5. **Access the Application:**
   - Backend API: http://127.0.0.1:8000/
   - API documentation: http://127.0.0.1:8000/docs
   - React Dashboard: http://localhost:3000/
   - Sample data: http://127.0.0.1:8000/data/sample

## Data Processing Pipeline

### AIS Data Preprocessing

The preprocessing pipeline (`backend/scripts/preprocess_ais.py`) performs comprehensive data preparation for AIS trajectory analysis:

#### 1. **Filtering**
   - **MMSI Validation:** Removes rows with invalid or missing vessel identifiers
   - **Speed Filtering:** Removes negative speeds and extreme outliers (>100 knots)
   - **Course Filtering:** Validates COG (Course Over Ground) values (0-360 degrees)
   - **Heading Validation:** Ensures heading values are within valid range (0-360 degrees)

#### 2. **Cleaning**
   - **Outlier Removal:** Uses IQR (Interquartile Range) method to detect and remove speed outliers
   - **Missing Value Imputation:** Fills missing numeric values using median (per-vessel when possible, otherwise global median)
   - **Status Standardization:** Standardizes navigational status values

#### 3. **Normalization**
   - **Speed Normalization:** Normalizes SOG (Speed Over Ground) to [0, 1] range using min-max scaling
   - **Course Normalization:** Normalizes COG to [0, 1] range (0-360° → 0-1)
   - **Heading Normalization:** Normalizes heading to [0, 1] range (0-360° → 0-1)

#### 4. **Resampling**
   - **Row-based Resampling:** Since timestamp data may not be available, uses row-based resampling
   - Configurable interval (default: 1 = no resampling)
   - Maintains chronological order per vessel

#### 5. **Trajectory Segmentation**
   - **Vessel-based Splitting:** Groups data by vessel ID (MMSI)
   - **Time Gap Detection:** Splits trajectories when row index gap exceeds threshold (default: 100 rows)
   - **Segment ID Assignment:** Creates unique segment identifiers for each trajectory segment

#### Usage
```bash
# Run preprocessing with default parameters
python backend/scripts/preprocess_ais.py

# Or customize parameters in the script:
# - resample_interval: Resampling frequency (default: 1)
# - time_gap_threshold: Row gap threshold for trajectory splitting (default: 100)
```

#### Output
- **File:** `backend/data/ais_data_preprocessed.csv`

## Anomaly Detection Pipeline

The anomaly module augments trajectory predictions with live alerts when vessels deviate from normal behaviour.

### Training

1. Ensure `backend/data/ais_data_preprocessed.csv` exists (run the preprocessing script first).
2. Train the anomaly detector:
   ```bash
   python backend/scripts/train_anomaly_detector.py --input backend/data/ais_data_preprocessed.csv
   ```
   - Computes feature-wise normalization statistics.
   - Uses the specified quantile (default `0.95`) of point-wise MSE as the alert threshold.
   - Saves the bundle to `backend/model/anomaly_detector.joblib` plus metrics to `backend/model/anomaly_metrics.json`.
   - Optional flags:
     - `--max-rows 5000` to cap rows loaded for faster experiments.
     - `--quantile 0.98` to tighten/loosen the alert sensitivity.

### Backend Integration

- `GET /trajectories` now enriches each segment with anomaly scores when a model is available.
- Detected anomalies are pushed to an in-memory queue exposed via:
  - `GET /anomalies/queue` – latest alerts for map/dashboard consumption.
  - `GET /anomalies/status` – model availability + most recent evaluation metrics.

### Frontend Behaviour

- The dashboard polls `/anomalies/queue` every 10 seconds.
- Anomalous trajectories are highlighted on the map and in the sidebar list.
- Selecting an alert focuses the corresponding vessel and displays detailed anomaly metadata.
- **Added Columns:**
  - `timestamp`: Real timestamps (when available)
  - `latitude`, `longitude`: Real geographic coordinates (degrees)
  - `row_index`: Row index for time proxy
  - `sog_normalized`: Normalized speed (0-1)
  - `cog_normalized`: Normalized course (0-1)
  - `heading_normalized`: Normalized heading (0-1)
  - `segment_id`: Unique trajectory segment identifier
  - `shiptype`: Vessel type name (e.g., "Passenger", "Cargo", "Tanker")

### Trajectory Prediction Model

The project now uses a single lightweight transformer that predicts the next AIS point from recent motion samples.

#### Model Overview
- **Input features (3D):** `sog`, `cog`, `heading`
- **Sequence length:** Last 16 timesteps (`SEQ_LEN`)
- **Prediction horizon:** Next 1 timestep (`PRED_LEN`)
- **Loss / metric:** Mean Squared Error (MSE)
- **Artifacts:** `transformer_trajectory.pth`, `input_norm.npz`, `latent_features.npy`, `metrics_report.txt`

#### Training
```bash
python backend/scripts/trajectory_predictor.py
```

#### Evaluation
```bash
python backend/scripts/evaluate_trajectory.py
```

See `backend/scripts/README_TRAJECTORY_MODELS.md` for more details.

### Legacy Processing 
1. **Data Cleaning:**
   - Convert timestamps to UTC datetime
   - Validate GPS coordinates (-90≤lat≤90, -180≤lon≤180)
   - Convert numeric fields and remove invalid entries
   - Fill missing destinations with "Unknown"

2. **Vessel Grouping:**
   - Create VESSEL_ID based on destination
   - Sort by timestamp for chronological order
   - Preserve all original data points

3. **Output:**
   - Clean CSV file with 8 columns
   - All 10,000 records preserved
   - Real coordinates maintained

## Tech Stack

### Backend
- **FastAPI** - Modern web framework for building APIs
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Uvicorn** - ASGI server for FastAPI

### Data Science
- **Scikit-learn** - Machine learning library
- **Python-dateutil** - Date/time utilities
- **Pytz** - Timezone handling

### Frontend
- **React** - Modern JavaScript library for building user interfaces
- **Create React App** - Development environment and build tooling
- **Recharts** - Charting library for trajectory visualization
- **React-Leaflet** - React wrapper for Leaflet maps
- **Leaflet** - Interactive map library (similar to MovingPandas folium/hvplot)
- **Axios** - HTTP client for API communication

### Development
- **Virtual Environment** - Isolated Python environment
- **Requirements.txt** - Python dependency management
- **package.json** - Node.js dependency management

## API Endpoints

### GET /
- **Description:** Welcome message with available endpoints
- **Response:** JSON with welcome message and endpoint list

### GET /health
- **Description:** Health check endpoint
- **Response:** JSON with system status and model availability

### GET /data/preprocessed
- **Description:** Get preprocessed AIS data with pagination
- **Parameters:**
  - `limit` (int, default: 100, max: 1000): Number of records to return
  - `offset` (int, default: 0): Number of records to skip
- **Response:** JSON with data array and metadata

### GET /data/sample
- **Description:** Get sample preprocessed data
- **Parameters:**
  - `n` (int, default: 10, max: 100): Number of sample records
- **Response:** JSON array with sample vessel data

### GET /data/stats
- **Description:** Get statistics about the preprocessed dataset
- **Response:** JSON with dataset statistics (total records, unique vessels, numeric stats)

### GET /models/status
- **Description:** Check availability of the trajectory model and cached artifacts
- **Response:** JSON with transformer availability and latent feature status

### POST /predict/trajectory
- **Description:** Predict the next AIS point from the latest observations
- **Request Body:**
  ```json
  {
    "sequence": [[sog, cog, heading], ...],  // Last 16 timesteps
    "model_type": "transformer"
  }
  ```
  - **Note:** Input features should be normalized with the same statistics used for training (see `input_norm.npz`)
- **Response:**
  ```json
  {
    "predictions": [[sog, cog, heading]],
    "model_type": "transformer",
    "input_length": 16,
    "prediction_length": 1
  }
  ```
- **Note:** Requires the transformer weights (`transformer_trajectory.pth`). Train using `python backend/scripts/trajectory_predictor.py`

### GET /trajectories
- **Description:** Get trajectory data for visualization
- **Parameters:**
  - `ship_type` (string, optional): Filter by ship type (e.g., "Passenger", "Cargo")
  - `segment_id` (int, optional): Filter by specific segment ID
  - `limit` (int, default: 100): Maximum number of segments to return
- **Response:** JSON with trajectory data grouped by segment

### GET /trajectories/ship-types
- **Description:** Get list of available ship types with counts
- **Response:** JSON array with ship type statistics

## Frontend Dashboard

The React dashboard provides interactive visualization of AIS trajectory data, inspired by [MovingPandas ship data analysis](https://movingpandas.github.io/movingpandas-website/2-analysis-examples/ship-data.html).

### Features
- **Map Visualization:** Interactive map showing vessel trajectories on real-world maps (similar to [MovingPandas example](https://movingpandas.github.io/movingpandas-website/2-analysis-examples/ship-data.html))
  - Trajectories displayed as colored lines on OpenStreetMap
  - Color-coded by ship type (Passenger=blue, Tanker=red, Cargo=orange, etc.)
  - Click trajectories on map to view details
  - Start/end markers for selected trajectories
- **Trajectory Visualization:** Interactive charts showing speed, course, and heading over time
- **Ship Type Filtering:** Filter trajectories by vessel type (Passenger, Cargo, etc.)
- **Multi-Chart View:** 
  - Speed Over Ground (SOG) timeline
  - Course Over Ground (COG) timeline
  - Heading timeline
  - Combined normalized features view
- **Trajectory Selection:** Click on any trajectory to view detailed charts and map
- **Real-time Data:** Fetches data from the FastAPI backend

### Components
- `TrajectoryViewer` - Main component for trajectory visualization
- `TrajectoryMap` - Interactive map component using react-leaflet
- Uses Recharts for interactive charts
- Uses Leaflet/OpenStreetMap for geographic visualization
- Responsive design for desktop and mobile

### Usage
1. Start the backend: `uvicorn backend.main:app --reload`
2. Start the frontend: `cd dashboard-app && npm start`
3. Open http://localhost:3000 in your browser
4. Select a ship type filter or browse all trajectories
5. Click on a trajectory to view detailed charts

### Research Goals
- [x] Develop transformer-based trajectory prediction model
- [x] Create interactive dashboard for vessel tracking
- [ ] Implement real-time AIS data ingestion
- [x] Add machine learning pipeline for pattern recognition

---
