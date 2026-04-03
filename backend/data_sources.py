import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PREDICTION_FILE = os.path.join(DATA_DIR, "vessel_position_predictions.csv")
PREDICTION_METRICS_FILE = os.path.join(DATA_DIR, "vessel_position_metrics.csv")
ANOMALY_RESULTS_FILE = os.path.join(DATA_DIR, "ais_anomaly_results.csv")
TRAJECTORY_ANOMALY_FILE = os.path.join(DATA_DIR, "trajectory_anomalies.csv")
TRAJECTORY_ANOMALY_METRICS_FILE = os.path.join(DATA_DIR, "trajectory_anomaly_metrics.csv")
ADVANCED_ANOMALY_FILE = os.path.join(DATA_DIR, "advanced_anomalies.csv")
ANOMALY_EVENTS_FILE = os.path.join(DATA_DIR, "anomaly_events.csv")


class _CachedFrame:
    """Lightweight cache to avoid re-reading CSVs on every request."""

    def __init__(self, path: str):
        self.path = path
        self._mtime: Optional[float] = None
        self._frame: Optional[pd.DataFrame] = None

    def get_frame(self) -> pd.DataFrame:
        if not os.path.exists(self.path):
            return pd.DataFrame()

        mtime = os.path.getmtime(self.path)
        if self._frame is None or self._mtime != mtime:
            self._frame = pd.read_csv(self.path)
            self._mtime = mtime
        return self._frame.copy()

    def last_updated(self) -> Optional[str]:
        if self._mtime is None and os.path.exists(self.path):
            self._mtime = os.path.getmtime(self.path)
        if self._mtime is None:
            return None
        return datetime.fromtimestamp(self._mtime).isoformat()


_prediction_cache = _CachedFrame(PREDICTION_FILE)
_prediction_metrics_cache = _CachedFrame(PREDICTION_METRICS_FILE)
_anomaly_results_cache = _CachedFrame(ANOMALY_RESULTS_FILE)
_trajectory_anomaly_cache = _CachedFrame(TRAJECTORY_ANOMALY_FILE)
_trajectory_anomaly_metrics_cache = _CachedFrame(TRAJECTORY_ANOMALY_METRICS_FILE)
_advanced_anomaly_cache = _CachedFrame(ADVANCED_ANOMALY_FILE)
_anomaly_events_cache = _CachedFrame(ANOMALY_EVENTS_FILE)


def _round(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    try:
        float_val = float(value)
        # Check for NaN and Inf which are not JSON compliant
        if pd.isna(float_val) or float_val != float_val or abs(float_val) == float('inf'):
            return None
        return round(float_val, digits)
    except (ValueError, TypeError):
        return None


def get_prediction_tracks(limit_per_vessel: int = 150) -> Tuple[List[Dict], Optional[str]]:
    """
    Return true vs predicted coordinates grouped per vessel for map display.

    Args:
        limit_per_vessel: maximum number of chronological samples per vessel.
    """
    df = _prediction_cache.get_frame()
    if df.empty:
        return [], None

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["mmsi", "timestamp"])

    tracks: List[Dict] = []
    for mmsi, group in df.groupby("mmsi"):
        trimmed = group.head(limit_per_vessel)
        points = []
        for _, row in trimmed.iterrows():
            # Convert timestamp to string for JSON serialization
            ts = row.get("timestamp")
            ts_str = str(ts) if pd.notna(ts) else None
            
            points.append(
                {
                    "timestamp": ts_str,
                    # Position predictions
                    "pred_latitude": _round(row.get("pred_latitude")),
                    "pred_longitude": _round(row.get("pred_longitude")),
                    "true_latitude": _round(row.get("true_latitude")),
                    "true_longitude": _round(row.get("true_longitude")),
                    # SOG/COG predictions
                    "pred_sog": _round(row.get("pred_sog"), digits=4),
                    "pred_cog": _round(row.get("pred_cog"), digits=4),
                    "true_sog": _round(row.get("true_sog"), digits=4),
                    "true_cog": _round(row.get("true_cog"), digits=4),
                    # Error metrics
                    "error_sog": _round(row.get("error_sog"), digits=6),
                    "error_cog": _round(row.get("error_cog"), digits=6),
                    "error_lat": _round(row.get("error_lat"), digits=8),
                    "error_lon": _round(row.get("error_lon"), digits=8),
                    "combined_error": _round(row.get("combined_error"), digits=6),
                    # Anomaly flag
                    "is_anomaly": int(row.get("is_anomaly", 0)) if pd.notna(row.get("is_anomaly")) else 0,
                }
            )

        tracks.append(
            {
                "mmsi": int(mmsi) if pd.notna(mmsi) else None,
                "points": points,
                "sample_count": len(points),
            }
        )

    return tracks, _prediction_cache.last_updated()


def get_prediction_metrics(top_n: int = 50) -> Tuple[List[Dict], Optional[str]]:
    """
    Return per-vessel error metrics sorted by mean MSE.
    """
    df = _prediction_metrics_cache.get_frame()
    if df.empty:
        return [], None

    df = df.sort_values("mse_mean")
    metrics = []
    for _, row in df.head(top_n).iterrows():
        metrics.append(
            {
                "mmsi": int(row["mmsi"]) if pd.notna(row["mmsi"]) else None,
                "mse_sog": _round(row.get("mse_sog"), digits=10),
                "mse_cog": _round(row.get("mse_cog"), digits=10),
                "mse_latitude": _round(row.get("mse_latitude"), digits=10),
                "mse_longitude": _round(row.get("mse_longitude"), digits=10),
                "mse_mean": _round(row.get("mse_mean"), digits=10),
                "anomaly_count": int(row.get("anomaly_count", 0)) if pd.notna(row.get("anomaly_count")) else 0,
                "anomaly_rate": _round(row.get("anomaly_rate"), digits=4),
                "anomaly_threshold": _round(row.get("anomaly_threshold"), digits=6),
            }
        )

    return metrics, _prediction_metrics_cache.last_updated()


def get_anomaly_prediction_points(limit: int = 250) -> Tuple[List[Dict], Optional[str]]:
    """
    Return anomaly prediction rows with errors for visual overlays.
    """
    df = _anomaly_results_cache.get_frame()
    if df.empty:
        return [], None

    df = df.head(limit)
    points = []
    for _, row in df.iterrows():
        points.append(
            {
                "mmsi": int(row["mmsi"]) if pd.notna(row["mmsi"]) else None,
                "lat_true": _round(row.get("lat_true")),
                "lon_true": _round(row.get("lon_true")),
                "lat_pred": _round(row.get("lat_pred")),
                "lon_pred": _round(row.get("lon_pred")),
                "error": _round(row.get("error"), digits=6),
                "anomaly": bool(row.get("anomaly")),
            }
        )

    return points, _anomaly_results_cache.last_updated()


def get_anomaly_alert_fallback(limit: int = 50) -> Tuple[List[Dict], Optional[str]]:
    """
    Build synthetic anomaly alerts from the offline results CSV.
    """
    points, updated_at = get_anomaly_prediction_points(limit=limit)
    if not points:
        return [], updated_at

    alerts: List[Dict] = []
    for idx, point in enumerate(points):
        alert_id = f"offline-{point.get('mmsi') or 'unknown'}-{idx}"
        alerts.append(
            {
                "alert_id": alert_id,
                "segment_id": None,
                "mmsi": point.get("mmsi"),
                "ship_type": None,
                "score": point.get("error"),
                "threshold": None,
                "details": {
                    "lat_true": point.get("lat_true"),
                    "lon_true": point.get("lon_true"),
                    "lat_pred": point.get("lat_pred"),
                    "lon_pred": point.get("lon_pred"),
                    "anomaly": point.get("anomaly"),
                },
                "evaluated_points": 1,
                "detected_at": updated_at,
                "source": "ais_anomaly_results.csv",
                "is_anomalous": point.get("anomaly", False),
            }
        )

    return alerts, updated_at


def get_trajectory_anomaly_tracks(limit_per_vessel: int = 500) -> Tuple[List[Dict], Optional[str]]:
    """
    Return trajectory-based anomaly data grouped per vessel.
    This uses direct trajectory analysis (speed changes, course changes, etc.)
    instead of prediction-based anomaly detection.
    """
    df = _trajectory_anomaly_cache.get_frame()
    if df.empty:
        return [], None

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["mmsi", "timestamp"])

    tracks: List[Dict] = []
    for mmsi, group in df.groupby("mmsi"):
        trimmed = group.head(limit_per_vessel)
        points = []
        for _, row in trimmed.iterrows():
            ts = row.get("timestamp")
            ts_str = str(ts) if pd.notna(ts) else None
            
            points.append(
                {
                    "timestamp": ts_str,
                    "latitude": _round(row.get("latitude")),
                    "longitude": _round(row.get("longitude")),
                    "sog": _round(row.get("sog"), digits=2),
                    "cog": _round(row.get("cog"), digits=2),
                    "heading": _round(row.get("heading"), digits=2),
                    "speed_change": _round(row.get("speed_change"), digits=3),
                    "course_change": _round(row.get("course_change"), digits=2),
                    "anomaly_type": row.get("anomaly_type", "none"),
                    "is_anomaly": int(row.get("is_anomaly", 0)) if pd.notna(row.get("is_anomaly")) else 0,
                    "severity": int(row.get("severity", 0)) if pd.notna(row.get("severity")) else 0,
                }
            )

        # Calculate vessel-level stats
        anomaly_points = [p for p in points if p["is_anomaly"] == 1]
        anomaly_types = {}
        for p in anomaly_points:
            at = p["anomaly_type"]
            anomaly_types[at] = anomaly_types.get(at, 0) + 1

        tracks.append(
            {
                "mmsi": int(mmsi) if pd.notna(mmsi) else None,
                "points": points,
                "sample_count": len(points),
                "anomaly_count": len(anomaly_points),
                "anomaly_rate": len(anomaly_points) / len(points) if points else 0,
                "anomaly_types": anomaly_types,
            }
        )

    # Sort by anomaly count descending
    tracks.sort(key=lambda x: x["anomaly_count"], reverse=True)

    return tracks, _trajectory_anomaly_cache.last_updated()


def get_trajectory_anomaly_metrics() -> Tuple[List[Dict], Optional[str]]:
    """
    Return per-vessel trajectory anomaly metrics.
    """
    df = _trajectory_anomaly_metrics_cache.get_frame()
    if df.empty:
        return [], None

    metrics = []
    for _, row in df.iterrows():
        # Parse anomaly breakdown if it's a string
        breakdown = row.get("anomaly_breakdown", {})
        if isinstance(breakdown, str):
            try:
                import ast
                breakdown = ast.literal_eval(breakdown)
            except:
                breakdown = {}

        metrics.append(
            {
                "mmsi": int(row["mmsi"]) if pd.notna(row["mmsi"]) else None,
                "total_points": int(row.get("total_points", 0)),
                "anomaly_points": int(row.get("anomaly_points", 0)),
                "anomaly_rate": _round(row.get("anomaly_rate"), digits=4),
                "anomaly_breakdown": breakdown,
                "max_speed_change": _round(row.get("max_speed_change"), digits=3),
                "max_course_change": _round(row.get("max_course_change"), digits=2),
                "avg_speed": _round(row.get("avg_speed"), digits=2),
                "speed_std": _round(row.get("speed_std"), digits=2),
            }
        )

    return metrics, _trajectory_anomaly_metrics_cache.last_updated()


def get_advanced_anomaly_tracks(limit_per_vessel: int = 500) -> Tuple[List[Dict], Optional[str]]:
    """
    Return advanced anomaly data with combined scoring, contextual anomalies,
    and temporal features.
    """
    df = _advanced_anomaly_cache.get_frame()
    if df.empty:
        return [], None

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["mmsi", "timestamp"])

    tracks: List[Dict] = []
    for mmsi, group in df.groupby("mmsi"):
        trimmed = group.head(limit_per_vessel)
        points = []
        for _, row in trimmed.iterrows():
            ts = row.get("timestamp")
            ts_str = str(ts) if pd.notna(ts) else None
            
            # Safely get values with fallbacks
            time_period_val = row.get("time_period")
            time_period = str(time_period_val) if pd.notna(time_period_val) else "unknown"
            
            ship_cat_val = row.get("ship_category")
            ship_category = str(ship_cat_val) if pd.notna(ship_cat_val) else "Unknown"
            
            anomaly_type_val = row.get("anomaly_type")
            anomaly_type = str(anomaly_type_val) if pd.notna(anomaly_type_val) else "none"
            
            ctx_type_val = row.get("contextual_anomaly_type")
            ctx_anomaly_type = str(ctx_type_val) if pd.notna(ctx_type_val) else "none"
            
            event_id_val = row.get("event_id")
            event_id = str(event_id_val) if pd.notna(event_id_val) else None
            
            risk_val = row.get("risk_level")
            risk_level = str(risk_val) if pd.notna(risk_val) else "low"
            
            # Handle contextual_anomaly which could be various types
            ctx_anomaly_val = row.get("contextual_anomaly", False)
            contextual_anomaly = bool(ctx_anomaly_val) if pd.notna(ctx_anomaly_val) else False
            
            points.append(
                {
                    "timestamp": ts_str,
                    "latitude": _round(row.get("latitude")),
                    "longitude": _round(row.get("longitude")),
                    "sog": _round(row.get("sog"), digits=2),
                    "cog": _round(row.get("cog"), digits=2),
                    "heading": _round(row.get("heading"), digits=2),
                    # Temporal features
                    "hour": int(row.get("hour", 0)) if pd.notna(row.get("hour")) else None,
                    "day_of_week": int(row.get("day_of_week", 0)) if pd.notna(row.get("day_of_week")) else None,
                    "time_period": time_period,
                    "is_weekend": int(row.get("is_weekend", 0)) if pd.notna(row.get("is_weekend")) else 0,
                    "ship_category": ship_category,
                    # Motion features
                    "speed_change": _round(row.get("speed_change"), digits=3),
                    "course_change": _round(row.get("course_change"), digits=2),
                    "acceleration": _round(row.get("acceleration"), digits=3),
                    "rate_of_turn": _round(row.get("rate_of_turn"), digits=2),
                    # Anomaly info
                    "anomaly_type": anomaly_type,
                    "is_anomaly": int(row.get("is_anomaly", 0)) if pd.notna(row.get("is_anomaly")) else 0,
                    "severity": int(row.get("severity", 0)) if pd.notna(row.get("severity")) else 0,
                    # Contextual anomaly
                    "contextual_anomaly": contextual_anomaly,
                    "contextual_anomaly_type": ctx_anomaly_type,
                    "neighbor_count": int(row.get("neighbor_count", 0)) if pd.notna(row.get("neighbor_count")) else 0,
                    # Event clustering
                    "event_id": event_id,
                    # Combined scoring
                    "trajectory_score": _round(row.get("trajectory_score"), digits=3),
                    "prediction_score": _round(row.get("prediction_score"), digits=3),
                    "contextual_score": _round(row.get("contextual_score"), digits=3),
                    "combined_anomaly_score": _round(row.get("combined_anomaly_score"), digits=3),
                    "risk_level": risk_level,
                }
            )

        # Calculate vessel-level stats
        anomaly_points = [p for p in points if p["is_anomaly"] == 1]
        contextual_anomalies = [p for p in points if p["contextual_anomaly"]]
        
        # Count anomaly types
        anomaly_types = {}
        for p in anomaly_points:
            at = p["anomaly_type"]
            anomaly_types[at] = anomaly_types.get(at, 0) + 1
        
        # Get unique events
        events = set(p["event_id"] for p in points if p["event_id"])

        # Get ship category safely
        vessel_ship_category = "Unknown"
        if points:
            vessel_ship_category = points[0].get("ship_category", "Unknown")
        
        tracks.append(
            {
                "mmsi": int(mmsi) if pd.notna(mmsi) else None,
                "points": points,
                "sample_count": len(points),
                "anomaly_count": len(anomaly_points),
                "contextual_anomaly_count": len(contextual_anomalies),
                "event_count": len(events),
                "anomaly_rate": len(anomaly_points) / len(points) if points else 0,
                "anomaly_types": anomaly_types,
                "ship_category": vessel_ship_category,
            }
        )

    # Sort by anomaly count descending
    tracks.sort(key=lambda x: x["anomaly_count"], reverse=True)

    return tracks, _advanced_anomaly_cache.last_updated()


def get_anomaly_events(limit: int = 100) -> Tuple[List[Dict], Optional[str]]:
    """
    Return anomaly events (grouped consecutive anomalies).
    """
    df = _anomaly_events_cache.get_frame()
    if df.empty:
        return [], None

    df = df.sort_values("duration_seconds", ascending=False).head(limit)
    
    events = []
    for _, row in df.iterrows():
        # Parse anomaly types if string
        anomaly_types = row.get("anomaly_types", {})
        if isinstance(anomaly_types, str):
            try:
                import ast
                anomaly_types = ast.literal_eval(anomaly_types)
            except:
                anomaly_types = {}

        events.append(
            {
                "event_id": row.get("event_id"),
                "mmsi": int(row["mmsi"]) if pd.notna(row["mmsi"]) else None,
                "start_time": str(row.get("start_time")) if pd.notna(row.get("start_time")) else None,
                "end_time": str(row.get("end_time")) if pd.notna(row.get("end_time")) else None,
                "duration_seconds": _round(row.get("duration_seconds"), digits=0),
                "point_count": int(row.get("point_count", 0)),
                "dominant_anomaly_type": row.get("dominant_anomaly_type", "unknown"),
                "anomaly_types": anomaly_types,
                "center_latitude": _round(row.get("center_latitude")),
                "center_longitude": _round(row.get("center_longitude")),
                "max_severity": int(row.get("max_severity", 0)) if pd.notna(row.get("max_severity")) else 0,
                "max_speed_change": _round(row.get("max_speed_change"), digits=2),
                "max_course_change": _round(row.get("max_course_change"), digits=1),
            }
        )

    return events, _anomaly_events_cache.last_updated()


