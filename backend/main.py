import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .data_sources import (
    CachedFrame,
    get_advanced_anomaly_tracks,
    get_anomaly_alert_fallback,
    get_anomaly_events,
    get_anomaly_prediction_points,
    get_prediction_metrics,
    get_prediction_tracks,
    get_trajectory_anomaly_metrics,
    get_trajectory_anomaly_tracks,
)

"""
Main backend application for AIS data processing and analysis.
Provides API endpoints for data access, trajectory predictions, and
multi-layer anomaly detection.

Run with: uvicorn backend.main:app --reload
"""

app = FastAPI(
    title="AIS LLM Boat Trajectory Prediction",
    version="1.0",
    description="Backend API for AIS trajectory analytics powered by LSTM-based pipelines",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"


def _path_from_env(env_var: str, default_path: Path) -> str:
    override = os.getenv(env_var)
    if override:
        return str(Path(override).expanduser())
    return str(default_path)


PREPROCESSED_DATA_PATH = _path_from_env(
    "PREPROCESSED_DATA_PATH",
    DATA_DIR / "cleaned_data.csv",
)
ANOMALY_MODEL_PATH = _path_from_env(
    "ANOMALY_MODEL_PATH",
    MODEL_DIR / "anomaly_detector.joblib",
)
ANOMALY_METRICS_PATH = MODEL_DIR / "anomaly_metrics.json"
VSL_PREDICTIONS_PATH = DATA_DIR / "vessel_position_predictions.csv"
VSL_METRICS_PATH = DATA_DIR / "vessel_position_metrics.csv"
ANOMALY_RESULTS_PATH = DATA_DIR / "ais_anomaly_results.csv"

# Mtime-cached loader for the cleaned input dataset. Each .get_frame() call
# returns a fresh copy, so endpoints are free to filter/mutate it.
_cleaned_data_cache = CachedFrame(PREPROCESSED_DATA_PATH)


def _load_cleaned_data() -> pd.DataFrame:
    """Return the cleaned AIS dataframe, raising 404 if it has not been generated yet."""
    if not os.path.exists(PREPROCESSED_DATA_PATH):
        raise HTTPException(
            status_code=404,
            detail=(
                "Preprocessed data file not found. Run preprocessing first: "
                "python -m backend.scripts.load_data"
            ),
        )
    return _cleaned_data_cache.get_frame()


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the AIS LLM Boat Trajectory Prediction Backend!",
        "version": "1.0",
        "endpoints": {
            "data": "/data/preprocessed",
            "sample": "/data/sample",
            "trajectories": "/trajectories",
            "models": "/models/status",
            "docs": "/docs",
        },
    }


@app.get("/data/preprocessed")
def get_preprocessed_data(limit: int = 100, offset: int = 0):
    """Get preprocessed AIS data with pagination."""
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    limit = max(0, min(limit, 1000))

    df = _load_cleaned_data()
    page = df.iloc[offset : offset + limit]
    return {
        "data": page.to_dict(orient="records"),
        "total_returned": len(page),
        "offset": offset,
        "limit": limit,
    }


@app.get("/data/sample")
def get_sample_data(n: int = 10):
    """Get sample preprocessed data."""
    n = max(0, min(n, 100))
    df = _load_cleaned_data()
    return df.head(n).to_dict(orient="records")


@app.get("/data/stats")
def get_data_stats():
    """Get statistics about the preprocessed dataset."""
    df = _load_cleaned_data()

    stats = {
        "total_records": len(df),
        "unique_vessels": int(df["mmsi"].nunique()) if "mmsi" in df.columns else 0,
        "unique_segments": int(df["segment_id"].nunique()) if "segment_id" in df.columns else 0,
        "columns": df.columns.tolist(),
        "numeric_stats": {},
    }
    for col in df.select_dtypes(include=[np.number]).columns:
        stats["numeric_stats"][col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        }
    return stats


@app.get("/predictions/vessels")
def get_prediction_vessels(limit_per_vessel: int = Query(150, ge=10, le=500)):
    """Expose predicted vs true vessel positions for map overlays."""
    tracks, updated_at = get_prediction_tracks(limit_per_vessel=limit_per_vessel)
    if not tracks:
        raise HTTPException(
            status_code=404,
            detail="Prediction file not found or empty. Generate vessel_position_predictions.csv first.",
        )
    return {
        "tracks": tracks,
        "limit_per_vessel": limit_per_vessel,
        "last_updated": updated_at,
    }


@app.get("/predictions/metrics")
def get_prediction_metrics_view(top_n: int = Query(50, ge=5, le=200)):
    """Expose per-vessel prediction error metrics."""
    metrics, updated_at = get_prediction_metrics(top_n=top_n)
    if not metrics:
        raise HTTPException(
            status_code=404,
            detail="Prediction metrics file not found or empty. Run evaluation to refresh vessel_position_metrics.csv.",
        )
    return {
        "metrics": metrics,
        "top_n": top_n,
        "last_updated": updated_at,
    }


@app.get("/anomalies/predictions")
def get_anomaly_prediction_view(limit: int = Query(250, ge=10, le=1000)):
    """Expose anomaly prediction rows for visualization."""
    points, updated_at = get_anomaly_prediction_points(limit=limit)
    if not points:
        raise HTTPException(
            status_code=404,
            detail="AIS anomaly results file not found or empty. Generate ais_anomaly_results.csv first.",
        )
    return {
        "predictions": points,
        "limit": limit,
        "last_updated": updated_at,
    }


@app.get("/vessel/{mmsi}/explanation")
def get_vessel_explanation(mmsi: int):
    """Generate natural language explanation for prediction-based vessel anomalies."""
    from .explanation import TemplateGenerator

    tracks, _ = get_prediction_tracks(limit_per_vessel=500)

    vessel_track = next((t for t in tracks if t.get("mmsi") == mmsi), None)
    if not vessel_track:
        raise HTTPException(
            status_code=404,
            detail=f"No prediction data found for vessel {mmsi}",
        )

    points = vessel_track.get("points", [])
    if not points:
        raise HTTPException(
            status_code=404,
            detail=f"No trajectory points available for vessel {mmsi}",
        )

    metrics_list, _ = get_prediction_metrics(top_n=200)
    vessel_metrics = next((m for m in metrics_list if m.get("mmsi") == mmsi), None)

    generator = TemplateGenerator()
    explanation = generator.generate(mmsi, points, vessel_metrics)
    return explanation.to_dict()


@app.get("/vessel/{mmsi}/trajectory-explanation")
def get_vessel_trajectory_explanation(mmsi: int):
    """Generate natural language explanation for trajectory-based anomalies."""
    from .explanation.generator import TrajectoryAnomalyGenerator

    tracks, _ = get_trajectory_anomaly_tracks(limit_per_vessel=500)

    vessel_track = next((t for t in tracks if t.get("mmsi") == mmsi), None)
    if not vessel_track:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No trajectory anomaly data found for vessel {mmsi}. "
                "Run trajectory_anomaly_detector.py first."
            ),
        )

    points = vessel_track.get("points", [])
    if not points:
        raise HTTPException(
            status_code=404,
            detail=f"No trajectory points available for vessel {mmsi}",
        )

    generator = TrajectoryAnomalyGenerator()
    explanation = generator.generate(mmsi, points, None)
    return explanation.to_dict()


@app.get("/models/status")
def get_models_status():
    """Report availability of generated datasets and anomaly detector artifacts."""
    return {
        "datasets": {
            "cleaned_data": os.path.exists(PREPROCESSED_DATA_PATH),
            "vessel_predictions": VSL_PREDICTIONS_PATH.exists(),
            "prediction_metrics": VSL_METRICS_PATH.exists(),
            "anomaly_results": ANOMALY_RESULTS_PATH.exists(),
        },
        "anomaly_detector_artifacts": {
            "joblib_model": os.path.exists(ANOMALY_MODEL_PATH),
            "metrics": ANOMALY_METRICS_PATH.exists(),
        },
        "notes": "Run backend/scripts/load_data.py, lstm_model.py, and lstm_anomaly.py to refresh datasets.",
    }


@app.get("/trajectories")
def get_trajectories(
    ship_type: Optional[str] = None,
    segment_id: Optional[int] = None,
    mmsi: Optional[int] = None,
    limit: int = 100,
    include_coordinates: bool = True,
):
    """Get trajectory data for visualization, grouped by segment_id."""
    df = _load_cleaned_data()

    if "segment_id" not in df.columns:
        df["segment_id"] = df["mmsi"] if "mmsi" in df.columns else 0
    if "row_index" not in df.columns:
        df["row_index"] = range(len(df))

    if ship_type:
        df = df[df["shiptype"] == ship_type]
    if segment_id is not None:
        df = df[df["segment_id"] == segment_id]
    if mmsi is not None:
        df = df[df["mmsi"] == mmsi]

    trajectories = []
    for seg_id, group in df.groupby("segment_id"):
        if len(trajectories) >= limit:
            break

        sort_col = "timestamp" if "timestamp" in group.columns else "row_index"
        group = group.sort_values(sort_col).reset_index(drop=True)

        has_coords = "latitude" in group.columns and "longitude" in group.columns
        points = []
        for _, row in group.iterrows():
            point = {
                "row_index": int(row["row_index"]) if "row_index" in row else 0,
                "sog": float(row["sog"]) if "sog" in row else 0.0,
                "cog": float(row["cog"]) if "cog" in row else 0.0,
                "heading": float(row["heading"]) if "heading" in row else 0.0,
                "sog_normalized": float(row["sog_normalized"]) if "sog_normalized" in row else 0.0,
                "cog_normalized": float(row["cog_normalized"]) if "cog_normalized" in row else 0.0,
                "heading_normalized": float(row["heading_normalized"]) if "heading_normalized" in row else 0.0,
            }
            if has_coords:
                lat_val = row.get("latitude")
                lon_val = row.get("longitude")
                if pd.notna(lat_val) and pd.notna(lon_val):
                    point["lat"] = float(lat_val)
                    point["lon"] = float(lon_val)
            points.append(point)

        trajectories.append({
            "segment_id": int(seg_id),
            "mmsi": int(group["mmsi"].iloc[0]) if "mmsi" in group.columns else None,
            "ship_type": group["shiptype"].iloc[0] if "shiptype" in group.columns else None,
            "points": points,
        })

    if not include_coordinates:
        for trajectory in trajectories:
            for point in trajectory["points"]:
                point.pop("lat", None)
                point.pop("lon", None)

    return {
        "trajectories": trajectories,
        "total": len(trajectories),
        "filters": {
            "ship_type": ship_type,
            "segment_id": segment_id,
            "mmsi": mmsi,
        },
    }


@app.get("/trajectories/mmsi")
def get_available_mmsi(limit: int = Query(500, ge=10, le=5000)):
    """Get list of available MMSI identifiers and sample counts."""
    df = _load_cleaned_data()
    if "mmsi" not in df.columns:
        return {"mmsi": []}

    counts = (
        df["mmsi"]
        .dropna()
        .astype(int)
        .value_counts()
        .head(limit)
        .to_dict()
    )
    return {"mmsi": [{"id": int(k), "count": int(v)} for k, v in counts.items()]}


@app.get("/anomalies/queue")
def get_anomaly_queue(limit: int = Query(50, ge=1, le=500)):
    """Return recent anomaly alerts derived from the offline detection results CSV."""
    alerts, last_updated = get_anomaly_alert_fallback(limit=limit)
    if not alerts:
        raise HTTPException(
            status_code=404,
            detail=(
                "ais_anomaly_results.csv is missing or empty. "
                "Run python backend/scripts/lstm_anomaly.py to generate it."
            ),
        )

    capped = min(limit, len(alerts))
    return {
        "alerts": alerts[:capped],
        "total": len(alerts),
        "limit": capped,
        "source": "offline_results",
        "last_updated": last_updated,
    }


@app.get("/anomalies/status")
def get_anomaly_status():
    """Report anomaly artifact availability and the latest training metrics."""
    metrics: Optional[dict] = None
    if ANOMALY_METRICS_PATH.exists():
        try:
            with open(ANOMALY_METRICS_PATH) as fh:
                metrics = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            metrics = {"error": str(exc)}

    return {
        "anomaly_model_available": os.path.exists(ANOMALY_MODEL_PATH),
        "metrics": metrics,
        "results_csv_available": ANOMALY_RESULTS_PATH.exists(),
    }


# =============================================================================
# TRAJECTORY-BASED ANOMALY DETECTION ENDPOINTS
# Direct trajectory analysis (speed/course changes, loitering, drift, zigzag).
# =============================================================================

@app.get("/trajectory-anomalies/vessels")
def get_trajectory_anomaly_vessels(
    limit_per_vessel: int = Query(500, ge=10, le=2000)
):
    """Get trajectory-based anomaly data for all vessels."""
    tracks, updated_at = get_trajectory_anomaly_tracks(limit_per_vessel=limit_per_vessel)

    if not tracks:
        raise HTTPException(
            status_code=404,
            detail="Trajectory anomaly data not found. Run backend/scripts/trajectory_anomaly_detector.py first.",
        )

    return {
        "tracks": tracks,
        "vessel_count": len(tracks),
        "last_updated": updated_at,
    }


@app.get("/trajectory-anomalies/metrics")
def get_trajectory_anomaly_metrics_endpoint():
    """Get per-vessel trajectory anomaly metrics and breakdown."""
    metrics, updated_at = get_trajectory_anomaly_metrics()

    if not metrics:
        raise HTTPException(
            status_code=404,
            detail="Trajectory anomaly metrics not found. Run backend/scripts/trajectory_anomaly_detector.py first.",
        )

    return {
        "metrics": metrics,
        "vessel_count": len(metrics),
        "last_updated": updated_at,
    }


@app.get("/trajectory-anomalies/types")
def get_trajectory_anomaly_types():
    """Return descriptions of all anomaly types detected by trajectory analysis."""
    return {
        "anomaly_types": [
            {"type": "speed_change_critical", "severity": 5, "label": "Critical Speed Change",
             "description": "Sudden severe change in vessel speed (>6 knots change)", "color": "#e74c3c"},
            {"type": "course_change_critical", "severity": 5, "label": "Critical Course Change",
             "description": "Sharp sudden turn (>60° course change)", "color": "#c0392b"},
            {"type": "speed_change_warning", "severity": 3, "label": "Speed Change Warning",
             "description": "Moderate sudden change in vessel speed (3-6 knots)", "color": "#e67e22"},
            {"type": "course_change_warning", "severity": 3, "label": "Course Change Warning",
             "description": "Moderate sudden turn (30-60° course change)", "color": "#f39c12"},
            {"type": "speed_too_high", "severity": 4, "label": "Excessive Speed",
             "description": "Vessel traveling unusually fast (>25 knots)", "color": "#9b59b6"},
            {"type": "speed_too_low", "severity": 1, "label": "Very Low Speed",
             "description": "Vessel nearly stationary (<0.5 knots)", "color": "#3498db"},
            {"type": "loitering", "severity": 2, "label": "Loitering",
             "description": "Vessel staying in small area at low speed", "color": "#1abc9c"},
            {"type": "drift_detected", "severity": 2, "label": "Drift Detected",
             "description": "Heading differs significantly from course (wind/current drift)", "color": "#2ecc71"},
            {"type": "zigzag_pattern", "severity": 3, "label": "Zigzag Pattern",
             "description": "Frequent direction changes detected", "color": "#f1c40f"},
            {"type": "speed_deviation_from_traffic", "severity": 4, "label": "Speed Deviation from Traffic",
             "description": "Vessel speed differs significantly from nearby vessels", "color": "#8e44ad"},
            {"type": "course_deviation_from_traffic", "severity": 4, "label": "Course Deviation from Traffic",
             "description": "Vessel course differs significantly from nearby vessels", "color": "#9b59b6"},
            {"type": "combined_traffic_deviation", "severity": 5, "label": "Combined Traffic Deviation",
             "description": "Both speed and course differ from nearby vessels", "color": "#c0392b"},
        ]
    }


# =============================================================================
# ADVANCED ANOMALY DETECTION ENDPOINTS
# Combined scoring, contextual anomalies, temporal clustering.
# =============================================================================

@app.get("/advanced-anomalies/vessels")
def get_advanced_anomaly_vessels(
    limit_per_vessel: int = Query(500, ge=10, le=2000)
):
    """Get advanced anomaly data with combined scoring, contextual anomalies, and temporal features."""
    tracks, updated_at = get_advanced_anomaly_tracks(limit_per_vessel=limit_per_vessel)

    if not tracks:
        raise HTTPException(
            status_code=404,
            detail="Advanced anomaly data not found. Run backend/scripts/advanced_anomaly_detector.py first.",
        )

    return {
        "tracks": tracks,
        "vessel_count": len(tracks),
        "last_updated": updated_at,
    }


@app.get("/advanced-anomalies/events")
def get_anomaly_events_endpoint(
    limit: int = Query(100, ge=1, le=500)
):
    """Get anomaly events (grouped consecutive anomalies)."""
    events, updated_at = get_anomaly_events(limit=limit)

    if not events:
        raise HTTPException(
            status_code=404,
            detail="Anomaly events not found. Run backend/scripts/advanced_anomaly_detector.py first.",
        )

    return {
        "events": events,
        "event_count": len(events),
        "last_updated": updated_at,
    }


@app.get("/advanced-anomalies/summary")
def get_advanced_anomaly_summary():
    """Get summary statistics for advanced anomaly detection."""
    tracks, _ = get_advanced_anomaly_tracks(limit_per_vessel=100)
    events, events_updated = get_anomaly_events(limit=500)

    if not tracks:
        raise HTTPException(
            status_code=404,
            detail="Advanced anomaly data not found.",
        )

    total_vessels = len(tracks)
    total_anomalies = sum(t["anomaly_count"] for t in tracks)
    total_contextual = sum(t["contextual_anomaly_count"] for t in tracks)
    total_events = len(events)

    anomaly_type_counts: dict = {}
    for track in tracks:
        for atype, count in track.get("anomaly_types", {}).items():
            anomaly_type_counts[atype] = anomaly_type_counts.get(atype, 0) + count

    category_counts: dict = {}
    for track in tracks:
        cat = track.get("ship_category", "Unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    return {
        "total_vessels": total_vessels,
        "total_anomalies": total_anomalies,
        "total_contextual_anomalies": total_contextual,
        "total_events": total_events,
        "anomaly_type_distribution": anomaly_type_counts,
        "ship_category_distribution": category_counts,
        "last_updated": events_updated,
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "preprocessed_data_exists": os.path.exists(PREPROCESSED_DATA_PATH),
        "datasets_available": {
            "vessel_predictions": VSL_PREDICTIONS_PATH.exists(),
            "prediction_metrics": VSL_METRICS_PATH.exists(),
            "anomaly_results": ANOMALY_RESULTS_PATH.exists(),
            "trajectory_anomalies": (DATA_DIR / "trajectory_anomalies.csv").exists(),
            "advanced_anomalies": (DATA_DIR / "advanced_anomalies.csv").exists(),
            "anomaly_events": (DATA_DIR / "anomaly_events.csv").exists(),
        },
    }
