from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import pandas as pd
import numpy as np
import os
import sys

# Add scripts directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from .data_sources import (
    get_anomaly_alert_fallback,
    get_anomaly_prediction_points,
    get_prediction_metrics,
    get_prediction_tracks,
    get_trajectory_anomaly_tracks,
    get_trajectory_anomaly_metrics,
    get_advanced_anomaly_tracks,
    get_anomaly_events,
)

"""
Main backend application for AIS data processing and analysis.
Provides API endpoints for data access and trajectory prediction.

Run with: uvicorn backend.main:app --reload
"""

app = FastAPI(
    title="AIS LLM Boat Trajectory Prediction",
    version="1.0",
    description="Backend API for AIS trajectory analytics powered by LSTM-based pipelines"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000",
                   "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directories / configurable paths
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
    DATA_DIR / "cleaned_data.csv"
)
ANOMALY_MODEL_PATH = _path_from_env(
    "ANOMALY_MODEL_PATH",
    MODEL_DIR / "anomaly_detector.joblib"
)
VSL_PREDICTIONS_PATH = DATA_DIR / "vessel_position_predictions.csv"
VSL_METRICS_PATH = DATA_DIR / "vessel_position_metrics.csv"

# Global state
_anomaly_detector = None
_anomaly_alert_queue = deque(maxlen=200)


def get_anomaly_detector():
    """Load anomaly detection model if available."""
    global _anomaly_detector
    if _anomaly_detector is not None:
        return _anomaly_detector

    if not os.path.exists(ANOMALY_MODEL_PATH):
        return None

    try:
        from backend.anomaly_detection import AnomalyDetector

        _anomaly_detector = AnomalyDetector(ANOMALY_MODEL_PATH)
        return _anomaly_detector
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load anomaly detector: {exc}",
        ) from exc


def _record_anomaly_alert(trajectory: dict, anomaly_data: dict):
    """
    Store anomaly alerts in an in-memory queue for frontend consumption.
    Avoid duplicates for the same segment_id unless the score increases.
    """
    alert_id = f"{trajectory.get('segment_id')}:{trajectory.get('mmsi')}"
    score = anomaly_data.get("score", 0.0)

    existing_index = None
    for idx, alert in enumerate(_anomaly_alert_queue):
        if alert["alert_id"] == alert_id:
            existing_index = idx
            break

    alert_payload = {
        "alert_id": alert_id,
        "segment_id": trajectory.get("segment_id"),
        "mmsi": trajectory.get("mmsi"),
        "ship_type": trajectory.get("ship_type"),
        "score": score,
        "threshold": anomaly_data.get("threshold"),
        "details": anomaly_data.get("details", {}),
        "evaluated_points": anomaly_data.get("evaluated_points"),
        "detected_at": anomaly_data.get("generated_at", datetime.now(timezone.utc).isoformat()),
    }

    if existing_index is not None:
        existing_alert = _anomaly_alert_queue[existing_index]
        if score > existing_alert.get("score", 0.0):
            _anomaly_alert_queue.remove(existing_alert)
            _anomaly_alert_queue.appendleft(alert_payload)
        else:
            # Refresh timestamp to indicate recent detection
            existing_alert["detected_at"] = alert_payload["detected_at"]
            existing_alert["score"] = max(existing_alert.get("score", 0.0), score)
    else:
        _anomaly_alert_queue.appendleft(alert_payload)


@app.get("/")
def read_root():
    """Welcome endpoint"""
    return {
        "message": "Welcome to the AIS LLM Boat Trajectory Prediction Backend!",
        "version": "1.0",
        "endpoints": {
            "data": "/data/preprocessed",
            "sample": "/data/sample",
            "predict": "/predict/trajectory",
            "models": "/models/status",
            "docs": "/docs"
        }
    }


@app.get("/data/preprocessed")
def get_preprocessed_data(limit: int = 100, offset: int = 0):
    """
    Get preprocessed AIS data
    
    Args:
        limit: Number of records to return (default: 100, max: 1000)
        offset: Number of records to skip (default: 0)
    
    Returns:
        JSON array with preprocessed AIS data
    """
    try:
        if not os.path.exists(PREPROCESSED_DATA_PATH):
            raise HTTPException(
                status_code=404,
                detail=f"Preprocessed data file not found. Run preprocessing first: python backend/scripts/preprocess_ais.py"
            )
        
        limit = min(limit, 1000) 
        df = pd.read_csv(PREPROCESSED_DATA_PATH, skiprows=range(1, offset + 1), nrows=limit)
        return {
            "data": df.to_dict(orient="records"),
            "total_returned": len(df),
            "offset": offset,
            "limit": limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading data: {str(e)}")


@app.get("/data/sample")
def get_sample_data(n: int = 10):
    """
    Get sample preprocessed data
    
    Args:
        n: Number of sample records (default: 10, max: 100)
    
    Returns:
        JSON array with sample vessel data
    """
    try:
        if not os.path.exists(PREPROCESSED_DATA_PATH):
            # Fallback to old file if exists
            fallback_file = 'backend/data/ais_data_processed_simple.csv'
            if os.path.exists(fallback_file):
                df = pd.read_csv(fallback_file)
                return df.head(min(n, 100)).to_dict(orient="records")
            raise HTTPException(
                status_code=404,
                detail="No data file found. Run preprocessing first."
            )
        
        n = min(n, 100)  # Cap at 100
        df = pd.read_csv(PREPROCESSED_DATA_PATH, nrows=n)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading sample data: {str(e)}")


@app.get("/data/stats")
def get_data_stats():
    """
    Get statistics about the preprocessed dataset
    
    Returns:
        JSON with dataset statistics
    """
    try:
        if not os.path.exists(PREPROCESSED_DATA_PATH):
            raise HTTPException(
                status_code=404,
                detail="Preprocessed data file not found. Run preprocessing first."
            )
        
        df = pd.read_csv(PREPROCESSED_DATA_PATH)
        
        stats = {
            "total_records": len(df),
            "unique_vessels": int(df['mmsi'].nunique()) if 'mmsi' in df.columns else 0,
            "unique_segments": int(df['segment_id'].nunique()) if 'segment_id' in df.columns else 0,
            "columns": df.columns.tolist(),
            "numeric_stats": {}
        }
        
        # Add statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            stats["numeric_stats"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max())
            }
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing statistics: {str(e)}")


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
    """
    Generate natural language explanation for vessel anomalies.
    
    Uses the prompt encoder to convert numerical predictions into
    structured tokens, then generates human-readable explanations.
    
    Args:
        mmsi: Vessel MMSI identifier
    
    Returns:
        JSON with explanation, risk level, and anomaly breakdown
    """
    from .explanation import TemplateGenerator
    
    # Get vessel prediction data
    tracks, _ = get_prediction_tracks(limit_per_vessel=500)
    
    # Find the vessel
    vessel_track = None
    for track in tracks:
        if track.get("mmsi") == mmsi:
            vessel_track = track
            break
    
    if not vessel_track:
        raise HTTPException(
            status_code=404,
            detail=f"No prediction data found for vessel {mmsi}"
        )
    
    points = vessel_track.get("points", [])
    if not points:
        raise HTTPException(
            status_code=404,
            detail=f"No trajectory points available for vessel {mmsi}"
        )
    
    # Get metrics if available
    metrics_list, _ = get_prediction_metrics(top_n=200)
    vessel_metrics = None
    for m in metrics_list:
        if m.get("mmsi") == mmsi:
            vessel_metrics = m
            break
    
    # Generate explanation
    generator = TemplateGenerator()
    explanation = generator.generate(mmsi, points, vessel_metrics)
    
    return explanation.to_dict()


@app.get("/vessel/{mmsi}/trajectory-explanation")
def get_vessel_trajectory_explanation(mmsi: int):
    """
    Generate natural language explanation for trajectory-based anomalies.
    
    Uses direct trajectory analysis (speed/course changes) instead of
    prediction-based anomaly detection.
    
    Args:
        mmsi: Vessel MMSI identifier
    
    Returns:
        JSON with explanation, risk level, and anomaly breakdown
    """
    from .explanation.generator import TrajectoryAnomalyGenerator
    
    # Get vessel trajectory anomaly data
    tracks, _ = get_trajectory_anomaly_tracks(limit_per_vessel=500)
    
    # Find the vessel
    vessel_track = None
    for track in tracks:
        if track.get("mmsi") == mmsi:
            vessel_track = track
            break
    
    if not vessel_track:
        raise HTTPException(
            status_code=404,
            detail=f"No trajectory anomaly data found for vessel {mmsi}. Run trajectory_anomaly_detector.py first."
        )
    
    points = vessel_track.get("points", [])
    if not points:
        raise HTTPException(
            status_code=404,
            detail=f"No trajectory points available for vessel {mmsi}"
        )
    
    # Generate explanation using trajectory anomaly generator
    generator = TrajectoryAnomalyGenerator()
    explanation = generator.generate(mmsi, points, None)
    
    return explanation.to_dict()


@app.get("/models/status")
def get_models_status():
    """
    Report availability of generated datasets and anomaly detector artifacts.
    """
    return {
        "datasets": {
            "cleaned_data": os.path.exists(PREPROCESSED_DATA_PATH),
            "vessel_predictions": VSL_PREDICTIONS_PATH.exists(),
            "prediction_metrics": VSL_METRICS_PATH.exists(),
            "anomaly_results": (DATA_DIR / "ais_anomaly_results.csv").exists(),
        },
        "anomaly_detector_artifacts": {
            "joblib_model": os.path.exists(ANOMALY_MODEL_PATH),
            "metrics": (MODEL_DIR / "anomaly_metrics.json").exists(),
        },
        "notes": "Run backend/scripts/load_data.py, lstm_model.py, and lstm_anomaly.py to refresh datasets.",
    }


@app.get("/trajectories")
def get_trajectories(
    ship_type: Optional[str] = None,
    segment_id: Optional[int] = None,
    mmsi: Optional[int] = None,
    limit: int = 100,
    include_coordinates: bool = True
):
    """
    Get trajectory data for visualization
    
    Args:
        ship_type: Filter by ship type (e.g., "Passenger", "Cargo")
        segment_id: Filter by specific segment ID
        limit: Maximum number of segments to return
        include_coordinates: Include latitude/longitude values when present in the dataset
    
    Returns:
        JSON with trajectory data grouped by segment
    """
    try:
        if not os.path.exists(PREPROCESSED_DATA_PATH):
            raise HTTPException(
                status_code=404,
                detail="Preprocessed data file not found. Run preprocessing first."
            )
        
        df = pd.read_csv(PREPROCESSED_DATA_PATH)

        # Ensure essential structural columns exist
        if 'segment_id' not in df.columns:
            df = df.copy()
            if 'mmsi' in df.columns:
                df['segment_id'] = df['mmsi']
            else:
                df['segment_id'] = 0
        if 'row_index' not in df.columns:
            df = df.copy()
            df['row_index'] = range(len(df))
        
        # Apply filters
        if ship_type:
            df = df[df['shiptype'] == ship_type]
        if segment_id is not None:
            df = df[df['segment_id'] == segment_id]
        if mmsi is not None:
            df = df[df['mmsi'] == mmsi]
        
        # Group by segment_id to create trajectories
        trajectories = []
        segments = df.groupby('segment_id')
        
        detector = None
        try:
            detector = get_anomaly_detector()
        except HTTPException:
            detector = None

        for seg_id, group in segments:
            if len(trajectories) >= limit:
                break
            
            # Sort by timestamp if available, else row_index
            if 'timestamp' in group.columns:
                group = group.sort_values('timestamp').reset_index(drop=True)
            else:
                group = group.sort_values('row_index').reset_index(drop=True)
            
            # Extract trajectory points
            points = []
            for idx, row in group.iterrows():
                point = {
                    "row_index": int(row['row_index']) if 'row_index' in row else 0,
                    "sog": float(row['sog']) if 'sog' in row else 0.0,
                    "cog": float(row['cog']) if 'cog' in row else 0.0,
                    "heading": float(row['heading']) if 'heading' in row else 0.0,
                    "sog_normalized": float(row['sog_normalized']) if 'sog_normalized' in row else 0.0,
                    "cog_normalized": float(row['cog_normalized']) if 'cog_normalized' in row else 0.0,
                    "heading_normalized": float(row['heading_normalized']) if 'heading_normalized' in row else 0.0,
                }
                # Add real coordinates if available
                if 'latitude' in group.columns and 'longitude' in group.columns:
                    lat_val = row.get('latitude')
                    lon_val = row.get('longitude')
                    if pd.notna(lat_val) and pd.notna(lon_val):
                        point['lat'] = float(lat_val)
                        point['lon'] = float(lon_val)
                points.append(point)
            
            trajectory = {
                "segment_id": int(seg_id),
                "mmsi": int(group['mmsi'].iloc[0]) if 'mmsi' in group.columns else None,
                "ship_type": group['shiptype'].iloc[0] if 'shiptype' in group.columns else None,
                "points": points
            }

            if detector:
                try:
                    anomaly_score = detector.score_points(points)
                    if anomaly_score:
                        anomaly_dict = anomaly_score.__dict__
                        trajectory["anomaly"] = anomaly_dict
                        if anomaly_score.is_anomalous:
                            _record_anomaly_alert(trajectory, anomaly_dict)
                except Exception as anomaly_exc:
                    trajectory["anomaly_error"] = str(anomaly_exc)

            trajectories.append(trajectory)
        
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
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading trajectories: {str(e)}")


@app.get("/trajectories/mmsi")
def get_available_mmsi(limit: int = Query(500, ge=10, le=5000)):
    """Get list of available MMSI identifiers and sample counts."""
    try:
        if not os.path.exists(PREPROCESSED_DATA_PATH):
            raise HTTPException(
                status_code=404,
                detail="Preprocessed data file not found."
            )

        df = pd.read_csv(PREPROCESSED_DATA_PATH, usecols=lambda c: c in {"mmsi"})

        if 'mmsi' not in df.columns:
            return {"mmsi": []}

        counts = (
            df['mmsi']
            .dropna()
            .astype(int)
            .value_counts()
            .head(limit)
            .to_dict()
        )

        return {
            "mmsi": [{"id": int(k), "count": int(v)} for k, v in counts.items()]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading MMSI list: {str(e)}")


@app.get("/anomalies/queue")
def get_anomaly_queue(limit: int = 50):
    """Return the most recent vessel anomaly alerts."""
    detector_available = os.path.exists(ANOMALY_MODEL_PATH)

    if detector_available:
        try:
            get_anomaly_detector()
        except HTTPException:
            detector_available = False

    if detector_available:
        limit = max(1, min(limit, len(_anomaly_alert_queue) or 1))
        alerts = list(_anomaly_alert_queue)[:limit]
        return {
            "alerts": alerts,
            "total": len(_anomaly_alert_queue),
            "limit": limit,
            "source": "realtime_detector",
        }

    # Fallback to offline anomaly results CSV
    fallback_alerts, last_updated = get_anomaly_alert_fallback(limit=limit)
    if fallback_alerts:
        capped_limit = min(limit, len(fallback_alerts))
        return {
            "alerts": fallback_alerts[:capped_limit],
            "total": len(fallback_alerts),
            "limit": capped_limit,
            "source": "offline_results",
            "last_updated": last_updated,
        }

    raise HTTPException(
        status_code=404,
        detail="No anomaly detector available and ais_anomaly_results.csv is missing. Run backend/scripts/lstm_anomaly.py to generate fallback data.",
    )


@app.get("/anomalies/status")
def get_anomaly_status():
    """Report anomaly detector availability and latest evaluation metrics."""
    available = os.path.exists(ANOMALY_MODEL_PATH)
    metrics = None

    if available:
        try:
            from backend.anomaly_detection import load_latest_metrics

            metrics = load_latest_metrics()
        except Exception as exc:
            metrics = {"error": str(exc)}

    return {
        "anomaly_model_available": available,
        "metrics": metrics,
        "alert_queue_size": len(_anomaly_alert_queue),
    }

# TRAJECTORY-BASED ANOMALY DETECTION ENDPOINTS
# These use direct trajectory analysis (speed/course changes) instead of 
# prediction-based anomaly detection.

@app.get("/trajectory-anomalies/vessels")
def get_trajectory_anomaly_vessels(
    limit_per_vessel: int = Query(500, ge=10, le=2000)
):
    """
    Get trajectory-based anomaly data for all vessels.
    
    This returns anomalies detected by analyzing the trajectory directly:
    - Sudden speed changes (acceleration/deceleration)
    - Sudden course changes (sharp turns)
    - Loitering behavior
    - Drift detection
    - Zigzag patterns
    """
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
    """
    Get per-vessel trajectory anomaly metrics and breakdown.
    """
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
    """
    Return descriptions of all anomaly types detected by trajectory analysis.
    """
    return {
        "anomaly_types": [
            {
                "type": "speed_change_critical",
                "severity": 5,
                "label": "Critical Speed Change",
                "description": "Sudden severe change in vessel speed (>6 knots change)",
                "color": "#e74c3c"
            },
            {
                "type": "course_change_critical",
                "severity": 5,
                "label": "Critical Course Change",
                "description": "Sharp sudden turn (>60° course change)",
                "color": "#c0392b"
            },
            {
                "type": "speed_change_warning",
                "severity": 3,
                "label": "Speed Change Warning",
                "description": "Moderate sudden change in vessel speed (3-6 knots)",
                "color": "#e67e22"
            },
            {
                "type": "course_change_warning",
                "severity": 3,
                "label": "Course Change Warning",
                "description": "Moderate sudden turn (30-60° course change)",
                "color": "#f39c12"
            },
            {
                "type": "speed_too_high",
                "severity": 4,
                "label": "Excessive Speed",
                "description": "Vessel traveling unusually fast (>25 knots)",
                "color": "#9b59b6"
            },
            {
                "type": "speed_too_low",
                "severity": 1,
                "label": "Very Low Speed",
                "description": "Vessel nearly stationary (<0.5 knots)",
                "color": "#3498db"
            },
            {
                "type": "loitering",
                "severity": 2,
                "label": "Loitering",
                "description": "Vessel staying in small area at low speed",
                "color": "#1abc9c"
            },
            {
                "type": "drift_detected",
                "severity": 2,
                "label": "Drift Detected",
                "description": "Heading differs significantly from course (wind/current drift)",
                "color": "#2ecc71"
            },
            {
                "type": "zigzag_pattern",
                "severity": 3,
                "label": "Zigzag Pattern",
                "description": "Frequent direction changes detected",
                "color": "#f1c40f"
            },
            {
                "type": "speed_deviation_from_traffic",
                "severity": 4,
                "label": "Speed Deviation from Traffic",
                "description": "Vessel speed differs significantly from nearby vessels",
                "color": "#8e44ad"
            },
            {
                "type": "course_deviation_from_traffic",
                "severity": 4,
                "label": "Course Deviation from Traffic",
                "description": "Vessel course differs significantly from nearby vessels",
                "color": "#9b59b6"
            },
            {
                "type": "combined_traffic_deviation",
                "severity": 5,
                "label": "Combined Traffic Deviation",
                "description": "Both speed and course differ from nearby vessels",
                "color": "#c0392b"
            }
        ]
    }


# =============================================================================
# ADVANCED ANOMALY DETECTION ENDPOINTS
# Features: Combined scoring, contextual anomalies, temporal clustering
# =============================================================================

@app.get("/advanced-anomalies/vessels")
def get_advanced_anomaly_vessels(
    limit_per_vessel: int = Query(500, ge=10, le=2000)
):
    """
    Get advanced anomaly data with combined scoring, contextual anomalies,
    and temporal features.
    
    Includes:
    - Temporal features (hour, day_of_week, time_period)
    - Contextual anomalies (relative to nearby vessels)
    - Event clustering (grouped consecutive anomalies)
    - Combined anomaly scores from multiple detection methods
    """
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
    """
    Get anomaly events (grouped consecutive anomalies).
    
    Events are clusters of anomalies that occur close together in time,
    representing a single behavioral incident rather than isolated points.
    """
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
    """
    Get summary statistics for advanced anomaly detection.
    """
    tracks, _ = get_advanced_anomaly_tracks(limit_per_vessel=100)
    events, events_updated = get_anomaly_events(limit=500)
    
    if not tracks:
        raise HTTPException(
            status_code=404,
            detail="Advanced anomaly data not found.",
        )
    
    # Calculate statistics
    total_vessels = len(tracks)
    total_anomalies = sum(t["anomaly_count"] for t in tracks)
    total_contextual = sum(t["contextual_anomaly_count"] for t in tracks)
    total_events = len(events)
    
    # Risk distribution
    risk_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    anomaly_type_counts = {}
    
    for track in tracks:
        for atype, count in track.get("anomaly_types", {}).items():
            anomaly_type_counts[atype] = anomaly_type_counts.get(atype, 0) + count
    
    # Ship category distribution
    category_counts = {}
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
    """Health check endpoint"""
    return {
        "status": "healthy",
        "preprocessed_data_exists": os.path.exists(PREPROCESSED_DATA_PATH),
        "datasets_available": {
            "vessel_predictions": VSL_PREDICTIONS_PATH.exists(),
            "prediction_metrics": VSL_METRICS_PATH.exists(),
            "anomaly_results": (DATA_DIR / "ais_anomaly_results.csv").exists(),
            "trajectory_anomalies": (DATA_DIR / "trajectory_anomalies.csv").exists(),
            "advanced_anomalies": (DATA_DIR / "advanced_anomalies.csv").exists(),
            "anomaly_events": (DATA_DIR / "anomaly_events.csv").exists(),
        }
    }


