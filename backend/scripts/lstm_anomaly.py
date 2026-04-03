import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Paths
RAW_DATA_PATH = Path("backend/data/cleaned_data.csv")
OUTPUT_PATH = Path("backend/data/ais_anomaly_results.csv")

SEQ_LEN = 10  # increase time steps? 
TEST_SIZE = 0.2
NUM_FEATURES = 5  # sog, cog, heading, latitude, longitude

NUMERIC_COLS = ["sog", "cog", "heading", "latitude", "longitude"]

df = pd.read_csv(RAW_DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["mmsi"] = df["mmsi"].astype(str).str.strip()

# Normalize numeric features
scaler = StandardScaler()
df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])

# Prepare sequences per vessel
def create_sequences(data, seq_len=SEQ_LEN):
    sequences = []
    targets = []
    for i in range(len(data) - seq_len):
        seq = data.iloc[i:i+seq_len][NUMERIC_COLS].values.astype(np.float32)
        target = data.iloc[i+seq_len][["latitude", "longitude"]].values.astype(np.float32)
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

X_all, y_all = [], []
mmsi_all = []

for mmsi, df_vessel in df.groupby("mmsi"):
    df_vessel = df_vessel.sort_values("timestamp")
    if len(df_vessel) < SEQ_LEN + 1:
        continue
    X_seq, y_seq = create_sequences(df_vessel, SEQ_LEN)
    X_all.append(X_seq)
    y_all.append(y_seq)
    mmsi_all.extend([mmsi]*len(X_seq))

X_all = np.vstack(X_all)
y_all = np.vstack(y_all)
mmsi_all = np.array(mmsi_all)

# Train/test split
X_train, X_test, y_train, y_test, mmsi_train, mmsi_test = train_test_split(
    X_all, y_all, mmsi_all, test_size=TEST_SIZE, random_state=42
)

# Build LSTM
model = Sequential([
    LSTM(64, input_shape=(SEQ_LEN, NUM_FEATURES)),
    Dense(32, activation="relu"),
    Dense(2)  # predict latitude and longitude
])

model.compile(optimizer="adam", loss="mse")
model.summary()

# Train
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Predict
y_pred = model.predict(X_test)

# Inverse scale predictions
scaled_cols = NUMERIC_COLS.copy()
lat_idx = scaled_cols.index("latitude")
lon_idx = scaled_cols.index("longitude")

# Recreate full scaled array for inverse transform
y_pred_full = np.zeros((len(y_pred), len(NUMERIC_COLS)))
y_pred_full[:, lat_idx] = y_pred[:, 0]
y_pred_full[:, lon_idx] = y_pred[:, 1]
y_pred_inv = scaler.inverse_transform(y_pred_full)
y_pred_lat = y_pred_inv[:, lat_idx]
y_pred_lon = y_pred_inv[:, lon_idx]

y_test_full = np.zeros((len(y_test), len(NUMERIC_COLS)))
y_test_full[:, lat_idx] = y_test[:, 0]
y_test_full[:, lon_idx] = y_test[:, 1]
y_test_inv = scaler.inverse_transform(y_test_full)
y_test_lat = y_test_inv[:, lat_idx]
y_test_lon = y_test_inv[:, lon_idx]

# Compute errors
errors = np.sqrt((y_pred_lat - y_test_lat)**2 + (y_pred_lon - y_test_lon)**2)

# Compute MSE per vessel
results_df = pd.DataFrame({
    "mmsi": mmsi_test,
    "lat_true": y_test_lat,
    "lon_true": y_test_lon,
    "lat_pred": y_pred_lat,
    "lon_pred": y_pred_lon,
    "error": errors
})

# Flag anomalies
threshold = results_df["error"].mean() + 3*results_df["error"].std()
results_df["anomaly"] = results_df["error"] > threshold

# Save results
results_df.to_csv(OUTPUT_PATH, index=False)
print(f"Anomaly detection results saved to {OUTPUT_PATH}")
