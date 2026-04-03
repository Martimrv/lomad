"""
Prompt Encoder Module

Converts numerical AIS predictions and features into structured text tokens
that can be processed by template-based generators or LLMs.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class VesselState:
    """Encoded vessel state with structured tokens"""
    mmsi: int
    speed_token: str
    course_token: str
    position_token: str
    anomaly_score_token: str
    risk_level: str
    risk_token: str
    
    # Raw values for template filling
    raw_values: Dict[str, Any]


@dataclass  
class AnomalyContext:
    """Context about detected anomalies"""
    has_anomaly: bool
    anomaly_type: str  # "position", "speed", "course", "combined"
    severity: str  # "low", "medium", "high", "critical"
    
    # Deviation details
    position_error_km: float
    speed_error_knots: float
    course_error_deg: float
    combined_error: float
    
    # Tokens
    deviation_tokens: Dict[str, str]


class PromptEncoder:
    """
    Encodes numerical AIS data into structured text tokens.
    
    These tokens can be:
    1. Used directly by template-based generators
    2. Fed to an LLM as a structured prompt
    3. Converted to embeddings for cross-modal alignment
    """
    
    # Risk level thresholds based on combined error
    RISK_THRESHOLDS = {
        "low": 0.2,
        "medium": 0.4,
        "high": 0.6,
        "critical": 0.8
    }
    
    # Anomaly type thresholds (normalized errors)
    ANOMALY_TYPE_THRESHOLD = 0.3
    
    def __init__(self):
        pass
    
    def encode_vessel_state(
        self,
        mmsi: int,
        sog: float,
        cog: float,
        latitude: float,
        longitude: float,
        anomaly_score: float = 0.0,
        combined_error: float = 0.0
    ) -> VesselState:
        """
        Encode current vessel state as structured tokens.
        
        Args:
            mmsi: Vessel identifier
            sog: Speed over ground (knots)
            cog: Course over ground (degrees)
            latitude: Current latitude
            longitude: Current longitude
            anomaly_score: Model's anomaly probability (0-1)
            combined_error: Combined prediction error (0-1)
        
        Returns:
            VesselState with encoded tokens
        """
        # Determine risk level
        risk_level = self._calculate_risk_level(combined_error, anomaly_score)
        
        # Create structured tokens
        speed_token = f"<SPEED>{sog:.1f} kn</SPEED>"
        course_token = f"<COURSE>{cog:.0f}°</COURSE>"
        position_token = f"<POSITION>({latitude:.4f}, {longitude:.4f})</POSITION>"
        anomaly_score_token = f"<ANOMALY_SCORE>{anomaly_score:.2f}</ANOMALY_SCORE>"
        risk_token = f"<RISK>{risk_level.upper()}</RISK>"
        
        return VesselState(
            mmsi=mmsi,
            speed_token=speed_token,
            course_token=course_token,
            position_token=position_token,
            anomaly_score_token=anomaly_score_token,
            risk_level=risk_level,
            risk_token=risk_token,
            raw_values={
                "sog": sog,
                "cog": cog,
                "latitude": latitude,
                "longitude": longitude,
                "anomaly_score": anomaly_score,
                "combined_error": combined_error
            }
        )
    
    def encode_anomaly_context(
        self,
        true_sog: float,
        pred_sog: float,
        true_cog: float,
        pred_cog: float,
        true_lat: float,
        pred_lat: float,
        true_lon: float,
        pred_lon: float,
        combined_error: float,
        is_anomaly: bool
    ) -> AnomalyContext:
        """
        Encode anomaly detection context with deviation details.
        
        Args:
            true_*: Actual values
            pred_*: Predicted values
            combined_error: Combined normalized error (0-1)
            is_anomaly: Whether this point was flagged as anomaly
        
        Returns:
            AnomalyContext with deviation analysis
        """
        # Calculate individual errors
        speed_error = abs(true_sog - pred_sog)
        
        # Handle COG wrap-around (359° to 1° should be 2°, not 358°)
        course_error = abs(true_cog - pred_cog)
        course_error = min(course_error, 360 - course_error)
        
        # Position error in km (approximate)
        lat_error_km = abs(true_lat - pred_lat) * 111  # 1 degree ≈ 111 km
        lon_error_km = abs(true_lon - pred_lon) * 85   # varies by latitude, ~85 km at mid-latitudes
        position_error_km = np.sqrt(lat_error_km**2 + lon_error_km**2)
        
        # Determine primary anomaly type
        anomaly_type = self._determine_anomaly_type(
            speed_error, course_error, position_error_km, combined_error
        )
        
        # Determine severity
        severity = self._determine_severity(combined_error)
        
        # Create deviation tokens
        deviation_tokens = {
            "position": f"<POSITION_ERROR>{position_error_km:.2f} km</POSITION_ERROR>",
            "speed": f"<SPEED_ERROR>{speed_error:.1f} kn</SPEED_ERROR>",
            "course": f"<COURSE_ERROR>{course_error:.0f}°</COURSE_ERROR>",
            "combined": f"<COMBINED_ERROR>{combined_error:.3f}</COMBINED_ERROR>"
        }
        
        return AnomalyContext(
            has_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
            severity=severity,
            position_error_km=position_error_km,
            speed_error_knots=speed_error,
            course_error_deg=course_error,
            combined_error=combined_error,
            deviation_tokens=deviation_tokens
        )
    
    def encode_trajectory_summary(
        self,
        points: List[Dict]
    ) -> Dict[str, Any]:
        """
        Encode a summary of the vessel's trajectory behavior.
        
        Args:
            points: List of trajectory points with predictions and errors
        
        Returns:
            Dictionary with trajectory summary tokens
        """
        if not points:
            return {
                "summary_token": "<TRAJECTORY>No data available</TRAJECTORY>",
                "anomaly_count": 0,
                "anomaly_rate": 0.0,
                "avg_position_error_km": 0.0,
                "max_position_error_km": 0.0,
                "trend": "unknown"
            }
        
        # Calculate statistics
        anomaly_count = sum(1 for p in points if p.get("is_anomaly", 0) == 1)
        anomaly_rate = anomaly_count / len(points) if points else 0
        
        # Position errors
        position_errors = []
        for p in points:
            lat_err = abs(p.get("true_latitude", 0) - p.get("pred_latitude", 0)) * 111
            lon_err = abs(p.get("true_longitude", 0) - p.get("pred_longitude", 0)) * 85
            position_errors.append(np.sqrt(lat_err**2 + lon_err**2))
        
        avg_error = np.mean(position_errors) if position_errors else 0
        max_error = np.max(position_errors) if position_errors else 0
        
        # Determine trend (are anomalies increasing, decreasing, or stable?)
        if len(points) >= 10:
            first_half = points[:len(points)//2]
            second_half = points[len(points)//2:]
            first_anomalies = sum(1 for p in first_half if p.get("is_anomaly", 0) == 1)
            second_anomalies = sum(1 for p in second_half if p.get("is_anomaly", 0) == 1)
            
            if second_anomalies > first_anomalies * 1.5:
                trend = "increasing"
            elif first_anomalies > second_anomalies * 1.5:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        # Create summary token
        summary_token = (
            f"<TRAJECTORY>"
            f"Points: {len(points)}, "
            f"Anomalies: {anomaly_count} ({anomaly_rate*100:.1f}%), "
            f"Avg Error: {avg_error:.2f} km, "
            f"Trend: {trend}"
            f"</TRAJECTORY>"
        )
        
        return {
            "summary_token": summary_token,
            "total_points": len(points),
            "anomaly_count": anomaly_count,
            "anomaly_rate": anomaly_rate,
            "avg_position_error_km": avg_error,
            "max_position_error_km": max_error,
            "trend": trend
        }
    
    def _calculate_risk_level(self, combined_error: float, anomaly_score: float) -> str:
        """Calculate risk level based on error and anomaly score"""
        # Weighted combination of both metrics
        risk_metric = 0.7 * combined_error + 0.3 * anomaly_score
        
        if risk_metric >= self.RISK_THRESHOLDS["critical"]:
            return "critical"
        elif risk_metric >= self.RISK_THRESHOLDS["high"]:
            return "high"
        elif risk_metric >= self.RISK_THRESHOLDS["medium"]:
            return "medium"
        else:
            return "low"
    
    def _determine_anomaly_type(
        self,
        speed_error: float,
        course_error: float,
        position_error_km: float,
        combined_error: float
    ) -> str:
        """Determine the primary type of anomaly"""
        if combined_error < 0.1:
            return "none"
        
        # Normalize errors to comparable scales
        # Speed: typical variation ~5 knots
        speed_norm = speed_error / 5.0
        # Course: typical variation ~30 degrees
        course_norm = course_error / 30.0
        # Position: typical variation ~2 km
        position_norm = position_error_km / 2.0
        
        # Find dominant factor
        factors = {
            "speed": speed_norm,
            "course": course_norm,
            "position": position_norm
        }
        
        max_factor = max(factors, key=factors.get)
        max_value = factors[max_factor]
        
        # Check if multiple factors are significant
        significant_count = sum(1 for v in factors.values() if v > 0.3)
        
        if significant_count >= 2:
            return "combined"
        elif max_value > 0.3:
            return max_factor
        else:
            return "minor"
    
    def _determine_severity(self, combined_error: float) -> str:
        """Determine anomaly severity from combined error"""
        if combined_error >= 0.8:
            return "critical"
        elif combined_error >= 0.5:
            return "high"
        elif combined_error >= 0.3:
            return "medium"
        else:
            return "low"

