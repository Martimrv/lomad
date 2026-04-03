import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Input,
    Bidirectional, LayerNormalization, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from load_data import load_and_clean_data

# Get the script's directory to build absolute paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

CLEANED_DATA_PATH = DATA_DIR / "cleaned_data.csv"
RAW_DATA_PATH = DATA_DIR / "AIS_2022_03_31.csv"

# Improved hyperparameters
SEQ_LEN = 20         # Longer context window for better temporal patterns
LSTM_UNITS_1 = 128   # Wider first LSTM layer  
LSTM_UNITS_2 = 64    # Wider second LSTM layer
DENSE_UNITS = 64
DROPOUT_RATE = 0.25
LEARNING_RATE = 0.0005
EPOCHS = 80
BATCH_SIZE = 32
PATIENCE = 15
MIN_VESSEL_POINTS = 80


def add_temporal_features(vessel_df):
    """Add time-based features to improve prediction accuracy."""
    df = vessel_df.copy()
    
    # Time since last observation (in seconds)
    df["time_diff"] = df["timestamp"].diff().dt.total_seconds().fillna(0)
    # Clamp extreme gaps
    df["time_diff"] = df["time_diff"].clip(0, 3600)
    
    # Cyclical encoding of hour/day to capture patterns
    if "timestamp" in df.columns:
        hours = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
        df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    else:
        df["hour_sin"] = 0.0
        df["hour_cos"] = 0.0
    
    return df


def create_sequences_with_deltas(features_scaled, deltas_scaled, seq=SEQ_LEN):
    """
    Create sequences where:
    - X = sequence of scaled features (including current position)
    - y = delta (change) in [sog, cog, lat, lon] for the NEXT timestep
    """
    X, y = [], []
    for i in range(len(features_scaled) - seq - 1):
        X.append(features_scaled[i:i+seq])
        y.append(deltas_scaled[i+seq])  # Delta at next position
    return np.array(X), np.array(y)


# Load data
if CLEANED_DATA_PATH.exists():
    print(f"Loading cleaned dataset from {CLEANED_DATA_PATH} ...")
    df = pd.read_csv(CLEANED_DATA_PATH)
else:
    print("Cleaned dataset not found. Building it via load_data.py ...")
    df = load_and_clean_data(raw_path=RAW_DATA_PATH)
    CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"Saved cleaned dataset to {CLEANED_DATA_PATH}")

# Sort by timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["mmsi", "timestamp"])

all_metrics = []
all_predictions = []

unique_vessels = df["mmsi"].unique()
print(f"Training models for {len(unique_vessels)} vessels...\n")

for mmsi in unique_vessels:
    print(f"Training vessel: {mmsi}")

    vessel = df[df["mmsi"] == mmsi].copy()
    vessel = vessel.sort_values("timestamp").reset_index(drop=True)

    # Skip vessels with insufficient data
    if len(vessel) < MIN_VESSEL_POINTS:
        print(f"Skipping {mmsi} (not enough data: {len(vessel)} points)")
        continue

    # Add temporal features
    vessel = add_temporal_features(vessel)

    # === DELTA-BASED APPROACH ===
    # Calculate deltas (changes) for position and motion
    vessel["delta_sog"] = vessel["sog"].diff()
    vessel["delta_cog"] = vessel["cog"].diff()
    vessel["delta_lat"] = vessel["latitude"].diff()
    vessel["delta_lon"] = vessel["longitude"].diff()
    
    # Handle COG wrap-around (e.g., 359 -> 1 should be +2, not -358)
    vessel["delta_cog"] = vessel["delta_cog"].apply(
        lambda x: x if pd.isna(x) else (x + 180) % 360 - 180
    )
    
    # Drop the first row (NaN from diff)
    vessel = vessel.dropna().reset_index(drop=True)
    
    if len(vessel) < MIN_VESSEL_POINTS:
        print(f"Skipping {mmsi} (not enough data after diff: {len(vessel)} points)")
        continue

    # Enhanced features: current state + motion + temporal
    feature_cols = [
        "sog", "cog", "heading", "latitude", "longitude",
        "time_diff", "hour_sin", "hour_cos"
    ]
    delta_cols = ["delta_sog", "delta_cog", "delta_lat", "delta_lon"]
    
    # Use RobustScaler for better outlier handling
    feature_scaler = RobustScaler()
    delta_scaler = RobustScaler()
    
    features_scaled = feature_scaler.fit_transform(vessel[feature_cols])
    deltas_scaled = delta_scaler.fit_transform(vessel[delta_cols])

    # Train/test split (80/20)
    split = int(len(features_scaled) * 0.8)
    train_features = features_scaled[:split]
    test_features = features_scaled[split:]
    train_deltas = deltas_scaled[:split]
    test_deltas = deltas_scaled[split:]

    # Adaptive sequence length
    seq = min(SEQ_LEN, len(train_features) // 4)
    if seq < 5:
        print(f"Skipping {mmsi} (sequence too short)")
        continue

    X_train, y_train = create_sequences_with_deltas(train_features, train_deltas, seq)
    X_test, y_test = create_sequences_with_deltas(test_features, test_deltas, seq)

    if len(X_test) < 5 or len(X_train) < 20:
        print(f"Skipping {mmsi} (not enough sequences)")
        continue

    # === IMPROVED MODEL ARCHITECTURE ===
    inputs = Input(shape=(seq, X_train.shape[2]))
    
    # Bidirectional LSTM for better temporal feature extraction
    x = Bidirectional(LSTM(LSTM_UNITS_1, return_sequences=True, kernel_regularizer=l2(1e-5)))(inputs)
    x = LayerNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    x = LSTM(LSTM_UNITS_2, kernel_regularizer=l2(1e-5))(x)
    x = LayerNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    # Dense layers with residual-like structure
    x = Dense(DENSE_UNITS, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)
    x = Dense(32, activation="relu")(x)

    # Prediction head: predict DELTAS (changes) for [sog, cog, lat, lon]
    pred_out = Dense(4, name="predictions")(x)

    model = Model(inputs, pred_out)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="huber",  # Huber loss is more robust to outliers than MSE
        metrics=["mae"]
    )

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=0
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=0
    )

    # Train
    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    # === PREDICTION ===
    preds_scaled = model.predict(X_test, verbose=0)

    # Inverse transform predicted deltas
    pred_deltas = delta_scaler.inverse_transform(preds_scaled)
    pred_delta_sog = pred_deltas[:, 0]
    pred_delta_cog = pred_deltas[:, 1]
    pred_delta_lat = pred_deltas[:, 2]
    pred_delta_lon = pred_deltas[:, 3]

    # True deltas
    true_deltas = delta_scaler.inverse_transform(y_test)
    true_delta_sog = true_deltas[:, 0]
    true_delta_cog = true_deltas[:, 1]
    true_delta_lat = true_deltas[:, 2]
    true_delta_lon = true_deltas[:, 3]

    # Get the actual positions from the test set for reconstruction
    test_vessel = vessel.iloc[split:].reset_index(drop=True)
    
    # Indices in test_vessel that correspond to our predictions
    n_preds = len(pred_deltas)
    max_idx = len(test_vessel) - 1
    pred_indices = list(range(seq, min(seq + n_preds, max_idx + 1)))
    valid_len = len(pred_indices)
    
    if valid_len == 0:
        print(f"Skipping {mmsi} (no valid prediction indices)")
        continue
    
    pred_delta_sog = pred_delta_sog[:valid_len]
    pred_delta_cog = pred_delta_cog[:valid_len]
    pred_delta_lat = pred_delta_lat[:valid_len]
    pred_delta_lon = pred_delta_lon[:valid_len]
    true_delta_sog = true_delta_sog[:valid_len]
    true_delta_cog = true_delta_cog[:valid_len]
    true_delta_lat = true_delta_lat[:valid_len]
    true_delta_lon = true_delta_lon[:valid_len]

    # Get previous positions (position BEFORE the delta was applied)
    prev_indices = [i - 1 for i in pred_indices]
    prev_sog = test_vessel["sog"].iloc[prev_indices].values
    prev_cog = test_vessel["cog"].iloc[prev_indices].values
    prev_lat = test_vessel["latitude"].iloc[prev_indices].values
    prev_lon = test_vessel["longitude"].iloc[prev_indices].values

    # Reconstruct predicted positions: prev_position + predicted_delta
    pred_sog = prev_sog + pred_delta_sog
    pred_cog = (prev_cog + pred_delta_cog) % 360  # Keep COG in [0, 360)
    pred_lat = prev_lat + pred_delta_lat
    pred_lon = prev_lon + pred_delta_lon

    # True positions
    true_sog = test_vessel["sog"].iloc[pred_indices].values
    true_cog = test_vessel["cog"].iloc[pred_indices].values
    true_lat = test_vessel["latitude"].iloc[pred_indices].values
    true_lon = test_vessel["longitude"].iloc[pred_indices].values
    timestamps = test_vessel["timestamp"].iloc[pred_indices].values

    # === ANOMALY DETECTION (based on prediction errors) ===
    error_sog = np.abs(true_sog - pred_sog)
    error_cog = np.abs(true_cog - pred_cog)
    # Handle COG wrap-around for error
    error_cog = np.minimum(error_cog, 360 - error_cog)
    error_lat = np.abs(true_lat - pred_lat)
    error_lon = np.abs(true_lon - pred_lon)

    # Convert lat/lon errors to approximate meters for consistent weighting
    lat_error_m = error_lat * 111_320  # ~111.32 km per degree latitude
    lon_error_m = error_lon * 111_320 * np.cos(np.radians(np.mean(true_lat)))
    position_error_m = np.sqrt(lat_error_m**2 + lon_error_m**2)

    def robust_normalize(arr):
        """Normalize using median/IQR for robustness to outliers."""
        q25, q75 = np.percentile(arr, [25, 75])
        iqr = q75 - q25
        if iqr == 0:
            return np.zeros_like(arr)
        return np.clip((arr - q25) / (iqr * 1.5), 0, 1)

    error_sog_norm = robust_normalize(error_sog)
    error_cog_norm = robust_normalize(error_cog)
    error_pos_norm = robust_normalize(position_error_m)

    # Combined error: position dominates, but SOG/COG also contribute
    combined_error = (
        0.15 * error_sog_norm +
        0.15 * error_cog_norm +
        0.70 * error_pos_norm
    )

    # Adaptive threshold: use IQR-based approach instead of fixed percentile
    q75 = np.percentile(combined_error, 75)
    iqr = np.percentile(combined_error, 75) - np.percentile(combined_error, 25)
    threshold = q75 + 1.5 * iqr  # Standard outlier detection threshold
    threshold = max(threshold, np.percentile(combined_error, 93))  # Minimum 93rd percentile
    is_anomaly = (combined_error > threshold).astype(int)

    # === METRICS ===
    mse_sog = mean_squared_error(true_sog, pred_sog)
    mse_cog = mean_squared_error(true_cog, pred_cog)
    mse_lat = mean_squared_error(true_lat, pred_lat)
    mse_lon = mean_squared_error(true_lon, pred_lon)
    mse_mean = (mse_sog + mse_cog + mse_lat + mse_lon) / 4
    
    mae_lat = mean_absolute_error(true_lat, pred_lat)
    mae_lon = mean_absolute_error(true_lon, pred_lon)

    # Also compute delta MSE (how well we predict the changes)
    mse_delta_lat = mean_squared_error(true_delta_lat, pred_delta_lat)
    mse_delta_lon = mean_squared_error(true_delta_lon, pred_delta_lon)
    
    # Position error in meters (more interpretable)
    mean_pos_error_m = float(np.mean(position_error_m))
    median_pos_error_m = float(np.median(position_error_m))

    all_metrics.append({
        "mmsi": mmsi,
        "mse_sog": mse_sog,
        "mse_cog": mse_cog,
        "mse_latitude": mse_lat,
        "mse_longitude": mse_lon,
        "mse_delta_lat": mse_delta_lat,
        "mse_delta_lon": mse_delta_lon,
        "mse_mean": mse_mean,
        "mae_latitude": mae_lat,
        "mae_longitude": mae_lon,
        "mean_position_error_m": mean_pos_error_m,
        "median_position_error_m": median_pos_error_m,
        "anomaly_count": int(is_anomaly.sum()),
        "anomaly_rate": float(is_anomaly.mean()),
        "anomaly_threshold": float(threshold)
    })

    # Save predictions
    vessel_pred_df = pd.DataFrame({
        "mmsi": mmsi,
        "timestamp": timestamps,
        "pred_sog": pred_sog,
        "pred_cog": pred_cog,
        "pred_latitude": pred_lat,
        "pred_longitude": pred_lon,
        "true_sog": true_sog,
        "true_cog": true_cog,
        "true_latitude": true_lat,
        "true_longitude": true_lon,
        "error_sog": error_sog,
        "error_cog": error_cog,
        "error_lat": error_lat,
        "error_lon": error_lon,
        "combined_error": combined_error,
        "is_anomaly": is_anomaly,
    })

    all_predictions.append(vessel_pred_df)

# Save results
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(DATA_DIR / "vessel_position_metrics.csv", index=False)

full_pred_df = pd.concat(all_predictions)
full_pred_df.to_csv(DATA_DIR / "vessel_position_predictions.csv", index=False)

print("\n" + "="*50)
print("Training finished!")
print("="*50)

# Summary statistics
total_predictions = len(full_pred_df)
total_anomalies = full_pred_df["is_anomaly"].sum()
anomaly_rate = (total_anomalies / total_predictions) * 100

print(f"\n=== Anomaly Detection Summary ===")
print(f"Total predictions: {total_predictions}")
print(f"Total anomalies flagged: {total_anomalies}")
print(f"Overall anomaly rate: {anomaly_rate:.2f}%")

# Position accuracy summary
mean_lat_error = full_pred_df["error_lat"].mean()
mean_lon_error = full_pred_df["error_lon"].mean()

# Detailed position accuracy in meters
avg_lat_m = mean_lat_error * 111_320
avg_lon_m = mean_lon_error * 111_320 * np.cos(np.radians(full_pred_df["true_latitude"].mean()))
avg_pos_m = np.sqrt(avg_lat_m**2 + avg_lon_m**2)

print(f"\n=== Position Accuracy ===")
print(f"Mean Latitude Error: {mean_lat_error:.6f}° ({mean_lat_error * 111:.2f} km)")
print(f"Mean Longitude Error: {mean_lon_error:.6f}° ({mean_lon_error * 85:.2f} km)")
print(f"Mean Position Error: {avg_pos_m:.1f} meters")
print(f"\n=== Per-vessel Metrics Summary ===")
print(f"Vessels trained: {len(metrics_df)}")
if len(metrics_df) > 0 and 'mean_position_error_m' in metrics_df.columns:
    print(f"Median vessel position error: {metrics_df['mean_position_error_m'].median():.1f} m")
    print(f"Best vessel error: {metrics_df['mean_position_error_m'].min():.1f} m")
    print(f"Worst vessel error: {metrics_df['mean_position_error_m'].max():.1f} m")

print("\nSaved:")
print(f" - {DATA_DIR / 'vessel_position_metrics.csv'}")
print(f" - {DATA_DIR / 'vessel_position_predictions.csv'}")
