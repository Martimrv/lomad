"""
Direct Trajectory Anomaly Detection

Detects anomalies by analyzing trajectory patterns directly:
- Sudden speed changes (acceleration/deceleration)
- Sudden course changes (sharp turns)
- Loitering behavior (low speed + position clustering)
- Drift detection (heading vs course mismatch)
- Speed anomalies (unusually high/low speed)

This approach analyzes the trajectory itself rather than comparing
predictions vs actual values.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import timedelta

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
CLEANED_DATA_PATH = DATA_DIR / "cleaned_data.csv"
OUTPUT_PATH = DATA_DIR / "trajectory_anomalies.csv"
METRICS_PATH = DATA_DIR / "trajectory_anomaly_metrics.csv"


@dataclass
class AnomalyThresholds:
    """Configurable thresholds for anomaly detection.
    
    Calibrated against AIS_2022_03_31 dataset statistics:
      - Speed changes: 99.8% of diffs < 3 knots, 99.9% < 5 knots
      - Course changes: 87.9% < 30°, 93.4% < 60°, 95.3% < 90°
      - ~52% of points have SOG <= 0.5 (normal at-anchor behaviour)
      - COG is unreliable at low speeds → drift needs higher speed gate
    Target overall anomaly rate: ~5-10%
    """
    # Speed change thresholds (knots per time step)
    speed_change_warning: float = 5.0      # Top 0.1% of speed diffs
    speed_change_critical: float = 10.0    # Extreme speed change

    # Course change thresholds (degrees)
    course_change_warning: float = 90.0    # Top ~6.6% of course diffs
    course_change_critical: float = 150.0  # Extreme turn

    # Absolute speed thresholds (knots)
    speed_too_low: float = 0.0             # Disabled — at-anchor is normal
    speed_too_high: float = 25.0           # Unusually fast

    # Loitering detection
    loiter_speed_threshold: float = 0.3    # Very low speed
    loiter_distance_threshold: float = 0.2 # km - tight radius
    loiter_time_points: int = 20           # Many consecutive points needed

    # Drift detection (heading vs course mismatch)
    drift_angle_threshold: float = 90.0    # Large heading-COG deviation
    drift_speed_min: float = 5.0           # Only meaningful at cruising speed

    # Zigzag detection
    zigzag_course_changes: int = 5         # More direction changes needed
    zigzag_window: int = 10                # Larger window
    zigzag_min_angle: float = 45.0         # Larger angle to count


class TrajectoryAnomalyDetector:
    """
    Detects anomalies by analyzing vessel trajectory patterns directly.
    """
    
    def __init__(self, thresholds: Optional[AnomalyThresholds] = None):
        self.thresholds = thresholds or AnomalyThresholds()
    
    def detect_speed_changes(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect sudden speed changes (acceleration/deceleration).
        Returns anomaly type for each row.
        """
        speed_diff = df['sog'].diff().abs()
        
        anomaly_type = pd.Series('none', index=df.index)
        anomaly_type[speed_diff >= self.thresholds.speed_change_critical] = 'speed_change_critical'
        anomaly_type[(speed_diff >= self.thresholds.speed_change_warning) & 
                     (speed_diff < self.thresholds.speed_change_critical)] = 'speed_change_warning'
        
        return anomaly_type
    
    def detect_course_changes(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect sudden course changes (sharp turns).
        Handles the 360-degree wrap-around.
        """
        cog_diff = df['cog'].diff()
        
        # Handle wrap-around (e.g., 350 -> 10 should be 20 degrees, not -340)
        cog_diff = cog_diff.apply(
            lambda x: min(abs(x), 360 - abs(x)) if pd.notna(x) else np.nan
        )
        
        anomaly_type = pd.Series('none', index=df.index)
        anomaly_type[cog_diff >= self.thresholds.course_change_critical] = 'course_change_critical'
        anomaly_type[(cog_diff >= self.thresholds.course_change_warning) & 
                     (cog_diff < self.thresholds.course_change_critical)] = 'course_change_warning'
        
        return anomaly_type
    
    def detect_speed_anomalies(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect absolute speed anomalies (too fast or too slow).
        At-rest/anchored vessels (SOG=0) are normal and not flagged.
        """
        anomaly_type = pd.Series('none', index=df.index)
        if self.thresholds.speed_too_low > 0:
            anomaly_type[df['sog'] <= self.thresholds.speed_too_low] = 'speed_too_low'
        anomaly_type[df['sog'] >= self.thresholds.speed_too_high] = 'speed_too_high'
        
        return anomaly_type
    
    def detect_drift(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect drift: when heading differs significantly from course over ground.
        This indicates the vessel is being pushed by currents/wind.
        """
        # Only check when vessel is moving
        moving_mask = df['sog'] >= self.thresholds.drift_speed_min

        # AIS heading 511 means "not available" — exclude from drift detection
        # Amilcar spot!!
        heading_valid_mask = (df['heading'] != 511) & (df['heading'] != 511.0)
        
        # Calculate heading-COG difference
        heading_cog_diff = (df['heading'] - df['cog']).abs()
        heading_cog_diff = heading_cog_diff.apply(
            lambda x: min(x, 360 - x) if pd.notna(x) else np.nan
        )
        
        anomaly_type = pd.Series('none', index=df.index)
        drift_mask = moving_mask & (heading_cog_diff >= self.thresholds.drift_angle_threshold)
        anomaly_type[drift_mask] = 'drift_detected'
        
        return anomaly_type
    
    def detect_loitering(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect loitering: vessel moving slowly in a small area.
        Excludes stationary/anchored vessels (SOG ≈ 0 throughout), since
        those are normal port behaviour, not suspicious loitering.
        """
        anomaly_type = pd.Series('none', index=df.index)
        
        if len(df) < self.thresholds.loiter_time_points:
            return anomaly_type
        
        # Convert lat/lon to approximate km
        lat_km = df['latitude'] * 111  # ~111 km per degree latitude
        lon_km = df['longitude'] * 111 * np.cos(np.radians(df['latitude'].mean()))
        
        window = self.thresholds.loiter_time_points
        
        for i in range(window, len(df)):
            window_df = df.iloc[i-window:i]
            
            mean_speed = window_df['sog'].mean()
            
            # Skip if speed is above loitering threshold (vessel is moving normally)
            if mean_speed > self.thresholds.loiter_speed_threshold:
                continue
            
            # Skip if vessel is fully stationary (anchored/berthed, not loitering)
            # Loitering requires SOME movement — at least 30% of points with SOG > 0
            moving_fraction = (window_df['sog'] > 0.1).mean()
            if moving_fraction < 0.3:
                continue
            
            # Check if position stays within threshold
            lat_range = lat_km.iloc[i-window:i].max() - lat_km.iloc[i-window:i].min()
            lon_range = lon_km.iloc[i-window:i].max() - lon_km.iloc[i-window:i].min()
            
            distance = np.sqrt(lat_range**2 + lon_range**2)
            
            if distance <= self.thresholds.loiter_distance_threshold:
                anomaly_type.iloc[i] = 'loitering'
        
        return anomaly_type
    
    def detect_zigzag(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect zigzag pattern: frequent direction changes.
        """
        anomaly_type = pd.Series('none', index=df.index)
        
        if len(df) < self.thresholds.zigzag_window:
            return anomaly_type
        
        # Calculate course changes
        cog_diff = df['cog'].diff()
        cog_diff = cog_diff.apply(
            lambda x: (x + 180) % 360 - 180 if pd.notna(x) else 0
        )
        
        window = self.thresholds.zigzag_window
        
        for i in range(window, len(df)):
            window_changes = cog_diff.iloc[i-window:i]
            
            # Count significant direction changes (sign changes in course diff)
            significant_changes = window_changes.abs() >= self.thresholds.zigzag_min_angle
            sign_changes = (window_changes[significant_changes].shift() * 
                          window_changes[significant_changes] < 0).sum()
            
            if sign_changes >= self.thresholds.zigzag_course_changes:
                anomaly_type.iloc[i] = 'zigzag_pattern'
        
        return anomaly_type
    
    def analyze_vessel(self, vessel_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all anomaly detections on a single vessel's trajectory.
        """
        df = vessel_df.copy()
        
        # Calculate deltas for reference
        df['speed_change'] = df['sog'].diff().abs()
        df['course_change'] = df['cog'].diff().apply(
            lambda x: min(abs(x), 360 - abs(x)) if pd.notna(x) else np.nan
        )
        
        # Run all detectors
        speed_change_anomaly = self.detect_speed_changes(df)
        course_change_anomaly = self.detect_course_changes(df)
        speed_anomaly = self.detect_speed_anomalies(df)
        drift_anomaly = self.detect_drift(df)
        loiter_anomaly = self.detect_loitering(df)
        zigzag_anomaly = self.detect_zigzag(df)
        
        # Combine anomalies with priority
        # Priority: critical > warning > behavioral patterns > none
        df['anomaly_type'] = 'none'
        
        # Layer anomalies (lower priority first, higher priority overwrites)
        df.loc[loiter_anomaly != 'none', 'anomaly_type'] = loiter_anomaly[loiter_anomaly != 'none']
        df.loc[zigzag_anomaly != 'none', 'anomaly_type'] = zigzag_anomaly[zigzag_anomaly != 'none']
        df.loc[drift_anomaly != 'none', 'anomaly_type'] = drift_anomaly[drift_anomaly != 'none']
        df.loc[speed_anomaly != 'none', 'anomaly_type'] = speed_anomaly[speed_anomaly != 'none']
        df.loc[course_change_anomaly == 'course_change_warning', 'anomaly_type'] = 'course_change_warning'
        df.loc[speed_change_anomaly == 'speed_change_warning', 'anomaly_type'] = 'speed_change_warning'
        df.loc[course_change_anomaly == 'course_change_critical', 'anomaly_type'] = 'course_change_critical'
        df.loc[speed_change_anomaly == 'speed_change_critical', 'anomaly_type'] = 'speed_change_critical'
        
        # Binary anomaly flag
        df['is_anomaly'] = (df['anomaly_type'] != 'none').astype(int)
        
        # Severity scoring
        severity_map = {
            'none': 0,
            'speed_too_low': 1,
            'loitering': 2,
            'drift_detected': 2,
            'zigzag_pattern': 3,
            'speed_change_warning': 3,
            'course_change_warning': 3,
            'speed_too_high': 4,
            'speed_change_critical': 5,
            'course_change_critical': 5
        }
        df['severity'] = df['anomaly_type'].map(severity_map)
        
        return df
    
    def get_anomaly_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for detected anomalies.
        """
        anomaly_counts = df['anomaly_type'].value_counts().to_dict()
        
        total_points = len(df)
        anomaly_points = (df['is_anomaly'] == 1).sum()
        
        return {
            'total_points': total_points,
            'anomaly_points': anomaly_points,
            'anomaly_rate': anomaly_points / total_points if total_points > 0 else 0,
            'anomaly_breakdown': anomaly_counts,
            'max_speed_change': df['speed_change'].max() if 'speed_change' in df else None,
            'max_course_change': df['course_change'].max() if 'course_change' in df else None,
            'avg_speed': df['sog'].mean(),
            'speed_std': df['sog'].std()
        }


def main():
    """Run trajectory anomaly detection on all vessels."""
    
    print("=" * 60)
    print("Trajectory Anomaly Detection")
    print("=" * 60)
    
    # Load data
    if not CLEANED_DATA_PATH.exists():
        print(f"Error: {CLEANED_DATA_PATH} not found.")
        print("Please run load_data.py first.")
        return
    
    print(f"\nLoading data from {CLEANED_DATA_PATH}...")
    df = pd.read_csv(CLEANED_DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['mmsi', 'timestamp'])
    
    print(f"Loaded {len(df)} data points from {df['mmsi'].nunique()} vessels")
    
    # Initialize detector
    detector = TrajectoryAnomalyDetector()
    
    # Process each vessel
    all_results = []
    all_metrics = []
    
    unique_vessels = df['mmsi'].unique()
    print(f"\nAnalyzing {len(unique_vessels)} vessels...")
    
    for i, mmsi in enumerate(unique_vessels):
        vessel_df = df[df['mmsi'] == mmsi].copy()
        vessel_df = vessel_df.sort_values('timestamp').reset_index(drop=True)
        
        if len(vessel_df) < 10:
            continue
        
        # Analyze trajectory
        result_df = detector.analyze_vessel(vessel_df)
        result_df['mmsi'] = mmsi
        
        # Get summary
        summary = detector.get_anomaly_summary(result_df)
        summary['mmsi'] = mmsi
        
        all_results.append(result_df)
        all_metrics.append(summary)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(unique_vessels)} vessels...")
    
    # Combine results
    full_results = pd.concat(all_results, ignore_index=True)
    metrics_df = pd.DataFrame(all_metrics)
    
    # Save results
    output_cols = [
        'mmsi', 'timestamp', 'latitude', 'longitude', 'sog', 'cog', 'heading',
        'speed_change', 'course_change', 'anomaly_type', 'is_anomaly', 'severity'
    ]
    full_results[output_cols].to_csv(OUTPUT_PATH, index=False)
    metrics_df.to_csv(METRICS_PATH, index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DETECTION COMPLETE")
    print("=" * 60)
    
    total_points = len(full_results)
    total_anomalies = full_results['is_anomaly'].sum()
    
    print(f"\nTotal data points analyzed: {total_points}")
    print(f"Total anomalies detected: {total_anomalies}")
    print(f"Overall anomaly rate: {total_anomalies/total_points*100:.2f}%")
    
    print("\n--- Anomaly Breakdown ---")
    anomaly_counts = full_results['anomaly_type'].value_counts()
    for anomaly_type, count in anomaly_counts.items():
        if anomaly_type != 'none':
            pct = count / total_points * 100
            print(f"  {anomaly_type}: {count} ({pct:.2f}%)")
    
    print(f"\nResults saved to:")
    print(f"  - {OUTPUT_PATH}")
    print(f"  - {METRICS_PATH}")


if __name__ == "__main__":
    main()

