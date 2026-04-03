# AIS-LLM Model Documentation

## Overview

This directory contains the machine learning models for AIS vessel trajectory prediction and anomaly detection. The primary model is a **dual-output LSTM** that predicts vessel movement and flags anomalies.

## Architecture

### Delta-Based LSTM Model

The model uses a **delta prediction approach** rather than absolute position prediction. This significantly improves trajectory accuracy.

```
Input Sequence (15 timesteps)     →    LSTM Stack    →    Dual Outputs
[SOG, COG, Heading, Lat, Lon]          ↓                   ├── Predicted Deltas (Δsog, Δcog, Δlat, Δlon)
                                   64 → 32 units          └── Anomaly Score (0-1)
```

#### Why Delta Prediction?

| Approach | Problem | Result |
|----------|---------|--------|
| **Absolute** | Model predicts mean position | Trajectories 20-30 km off |
| **Delta** | Model predicts movement changes | Trajectories ~2 km accuracy |

### Model Components

```
Input Layer (sequence_length, 5 features)
    ↓
LSTM(64, return_sequences=True)
    ↓
Dropout(0.2)
    ↓
LSTM(32)
    ↓
Dropout(0.2)
    ↓
Dense(32, relu) + BatchNorm
    ↓
Dense(16, relu)
    ↓
┌─────────────────┬─────────────────┐
│   Predictions   │  Anomaly Score  │
│   Dense(4)      │  Dense(1,σ)     │
└─────────────────┴─────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `lstm_model.py` | Main training script for trajectory prediction |
| `load_data.py` | Data loading and preprocessing |

## Usage

### Training the Model

```bash
cd backend/scripts
python lstm_model.py
```

This will:
1. Load cleaned AIS data from `backend/data/cleaned_data.csv`
2. Train a separate model for each vessel (personalized prediction)
3. Generate predictions and anomaly flags
4. Save results to:
   - `backend/data/vessel_position_predictions.csv`
   - `backend/data/vessel_position_metrics.csv`

### Data Preparation

```bash
python load_data.py
```

This preprocesses raw AIS data and saves to `backend/data/cleaned_data.csv`.

## Input Features

| Feature | Description | Unit |
|---------|-------------|------|
| `sog` | Speed Over Ground | knots |
| `cog` | Course Over Ground | degrees (0-360) |
| `heading` | Vessel heading | degrees (0-360) |
| `latitude` | Position latitude | degrees |
| `longitude` | Position longitude | degrees |

## Output Predictions

### Position Predictions
- `pred_sog`, `pred_cog` - Predicted speed and course
- `pred_latitude`, `pred_longitude` - Predicted position
- Computed as: `predicted = previous + predicted_delta`

### Anomaly Detection

The model uses **prediction error-based anomaly detection**:

```python
# Calculate errors
error_sog = |true_sog - pred_sog|
error_cog = |true_cog - pred_cog|  # with wrap-around handling
error_lat = |true_lat - pred_lat|
error_lon = |true_lon - pred_lon|

# Normalize and combine (position weighted higher)
combined_error = (0.15 * error_sog_norm + 
                  0.15 * error_cog_norm + 
                  0.35 * error_lat_norm + 
                  0.35 * error_lon_norm)

# Flag anomalies at 95th percentile
is_anomaly = combined_error > percentile(combined_error, 95)
```

## Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sequence Length | 15 (adaptive) | Input timesteps |
| Epochs | 50 | With early stopping |
| Batch Size | 32 | Training batch size |
| Train/Test Split | 80/20 | Per vessel |
| Min Data Points | 100 | Skip vessels with less |

### Callbacks

- **EarlyStopping**: Patience=10, monitors validation loss
- **ReduceLROnPlateau**: Reduces learning rate when stuck

## Metrics

Per-vessel metrics saved to `vessel_position_metrics.csv`:

| Metric | Description |
|--------|-------------|
| `mse_sog` | Mean Squared Error for speed |
| `mse_cog` | Mean Squared Error for course |
| `mse_latitude` | Mean Squared Error for latitude |
| `mse_longitude` | Mean Squared Error for longitude |
| `mse_delta_lat` | MSE for latitude delta prediction |
| `mse_delta_lon` | MSE for longitude delta prediction |
| `anomaly_count` | Number of anomalies detected |
| `anomaly_rate` | Percentage of anomalous points |
| `anomaly_threshold` | 95th percentile threshold used |

## Performance

After delta-based training:
- **Mean Latitude Error**: ~2.3 km (down from 26 km)
- **Mean Longitude Error**: ~11.7 km (down from 35 km)
- **Anomaly Detection Rate**: ~5.5%

## Extending the Model

### Adding New Features

1. Update `features` list in `lstm_model.py`
2. Update `delta_cols` if the feature should be delta-predicted
3. Adjust scaler indices accordingly

### Custom Anomaly Thresholds

Modify the percentile threshold:
```python
threshold = np.percentile(combined_error, 95)  # Change 95 to desired value
```

### Multi-step Prediction

The current model predicts one step ahead. For multi-step:
1. Use the predicted output as input for next prediction
2. Implement recursive prediction loop
3. Be aware of error accumulation

## Dependencies

```
tensorflow>=2.10
numpy
pandas
scikit-learn
```

