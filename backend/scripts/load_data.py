import pandas as pd
from pathlib import Path
from typing import Iterable, Tuple
from sklearn.preprocessing import StandardScaler

RAW_DATA_PATH = Path("backend/data/AIS_2022_03_31.csv")
CLEANED_DATA_PATH = Path("backend/data/cleaned_data.csv")

RENAME_MAP = {
    "MMSI": "mmsi",
    "BaseDateTime": "timestamp",
    "LAT": "latitude",
    "LON": "longitude",
    "SOG": "sog",
    "COG": "cog",
    "Heading": "heading",
}

NUMERIC_COLS = ["sog", "cog", "heading", "latitude", "longitude"]
OUTPUT_COLUMNS = ["mmsi", "timestamp", "sog", "cog", "heading", "latitude", "longitude"]

# AIS vessel type code for Cargo vessels
CARGO_VESSEL_TYPE = 70


def _clip_outliers(df: pd.DataFrame, columns: Iterable[str], quantiles: Tuple[float, float]) -> pd.DataFrame:
    """Clip all numerical outliers in one pass (prevents excessive row loss)."""
    low, high = quantiles
    cleaned_df = df.copy()

    bounds = {}
    for col in columns:
        q_low, q_high = cleaned_df[col].quantile([low, high])
        bounds[col] = (q_low, q_high)

    mask = pd.Series(True, index=cleaned_df.index)
    for col, (q_low, q_high) in bounds.items():
        mask &= (cleaned_df[col] >= q_low) & (cleaned_df[col] <= q_high)

    return cleaned_df[mask]


def load_and_clean_data(
    raw_path: Path = RAW_DATA_PATH,
    limit: int | None = 20000,
    quantile_bounds: Tuple[float, float] = (0.01, 0.99),
    vessel_type: int | None = CARGO_VESSEL_TYPE,
) -> pd.DataFrame:

    df = pd.read_csv(raw_path)

    # Filter by vessel type before renaming/dropping columns
    if vessel_type is not None and "VesselType" in df.columns:
        df = df[df["VesselType"] == vessel_type]

    # Rename
    df = df.rename(columns=RENAME_MAP)
    df = df[OUTPUT_COLUMNS]

    # Timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # MMSI as string, remove invalids
    df["mmsi"] = df["mmsi"].astype(str).str.strip()
    df = df[df["mmsi"].str.isnumeric()]  # keep only real MMSIs

    # Sort
    df = df.sort_values(["mmsi", "timestamp"])

    # Remove extreme outliers
    df = _clip_outliers(df, NUMERIC_COLS, quantile_bounds)

    # Optional limit
    if limit is not None:
        df = df.head(limit)

    return df


def normalize_data(df: pd.DataFrame):
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[NUMERIC_COLS] = scaler.fit_transform(df_scaled[NUMERIC_COLS])
    return df_scaled, scaler


if __name__ == "__main__":
    print("Loading and cleaning data...")
    df = load_and_clean_data()

    # Save RAW (unscaled) data - normalization will be done in lstm_model.py
    print("Saving cleaned data (unscaled)...")
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"Saved cleaned dataset: {CLEANED_DATA_PATH}")
    print(f"Total records: {len(df)}")
