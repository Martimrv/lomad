"""
Advanced Anomaly Detection System

This module implements:
1. Feature Engineering - Ship type, time of day, day of week
2. Contextual Anomalies - Detection relative to nearby vessels
3. Temporal Clustering - Group consecutive anomalies into events
4. Combined Scoring - Merge trajectory and prediction-based scores

Author: AIS LLM Research Project
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
CLEANED_DATA_PATH = DATA_DIR / "cleaned_data.csv"
TRAJECTORY_ANOMALY_PATH = DATA_DIR / "trajectory_anomalies.csv"
PREDICTION_PATH = DATA_DIR / "vessel_position_predictions.csv"
OUTPUT_PATH = DATA_DIR / "advanced_anomalies.csv"
EVENTS_PATH = DATA_DIR / "anomaly_events.csv"


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    """
    Adds temporal and categorical features to AIS data.
    """
    
    # Ship type mapping (common AIS ship types)
    SHIP_TYPE_CATEGORIES = {
        'Cargo': [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
        'Tanker': [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
        'Passenger': [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
        'Fishing': [30],
        'Tug': [31, 32, 52],
        'Pleasure': [36, 37],
        'Sailing': [36],
        'Military': [35],
        'Law Enforcement': [55],
        'Other': []  # Default
    }
    
    def __init__(self):
        # Create reverse mapping
        self.type_to_category = {}
        for category, codes in self.SHIP_TYPE_CATEGORIES.items():
            for code in codes:
                self.type_to_category[code] = category
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features with cyclical encoding."""
        df = df.copy()
        
        if 'timestamp' not in df.columns:
            return df
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Cyclical encoding for hour (0-23 -> sin/cos)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Cyclical encoding for day of week (0-6 -> sin/cos)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Cyclical encoding for month (1-12 -> sin/cos)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Time period categories
        df['time_period'] = pd.cut(
            df['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )
        
        # Weekend flag
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def add_ship_type_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ship type category features."""
        df = df.copy()
        
        if 'shiptype' in df.columns:
            # Map numeric ship type to category
            df['ship_category'] = df['shiptype'].apply(
                lambda x: self.type_to_category.get(x, 'Other') if pd.notna(x) else 'Unknown'
            )
        else:
            df['ship_category'] = 'Unknown'
        
        return df
    
    def add_motion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived motion features."""
        df = df.copy()
        
        # Speed acceleration (change in speed)
        if 'sog' in df.columns:
            df['acceleration'] = df.groupby('mmsi')['sog'].diff()
        
        # Rate of turn (change in course)
        if 'cog' in df.columns:
            cog_diff = df.groupby('mmsi')['cog'].diff()
            # Handle wrap-around
            df['rate_of_turn'] = cog_diff.apply(
                lambda x: min(abs(x), 360 - abs(x)) if pd.notna(x) else np.nan
            )
        
        # Heading-course deviation (indicates drift)
        if 'heading' in df.columns and 'cog' in df.columns:
            heading_cog_diff = (df['heading'] - df['cog']).abs()
            df['heading_cog_deviation'] = heading_cog_diff.apply(
                lambda x: min(x, 360 - x) if pd.notna(x) else np.nan
            )
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering."""
        df = self.add_temporal_features(df)
        df = self.add_ship_type_features(df)
        df = self.add_motion_features(df)
        return df


# =============================================================================
# CONTEXTUAL ANOMALY DETECTION
# =============================================================================

class ContextualAnomalyDetector:
    """
    Detects anomalies relative to nearby vessels and traffic patterns.
    """
    
    def __init__(
        self,
        spatial_radius_km: float = 5.0,
        temporal_window_minutes: int = 30,
        min_neighbors: int = 3,
        speed_deviation_threshold: float = 3.0,  # std deviations
        course_deviation_threshold: float = 90.0  # degrees
    ):
        self.spatial_radius_km = spatial_radius_km
        self.temporal_window_minutes = temporal_window_minutes
        self.min_neighbors = min_neighbors
        self.speed_deviation_threshold = speed_deviation_threshold
        self.course_deviation_threshold = course_deviation_threshold
    
    def haversine_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two points in km."""
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def find_neighbors(
        self, 
        df: pd.DataFrame, 
        target_idx: int,
        target_row: pd.Series
    ) -> pd.DataFrame:
        """Find vessels near the target vessel at the same time."""
        target_lat = target_row['latitude']
        target_lon = target_row['longitude']
        target_time = target_row['timestamp']
        target_mmsi = target_row['mmsi']
        
        # Time window
        time_start = target_time - timedelta(minutes=self.temporal_window_minutes)
        time_end = target_time + timedelta(minutes=self.temporal_window_minutes)
        
        # Filter by time and exclude self
        time_mask = (df['timestamp'] >= time_start) & (df['timestamp'] <= time_end)
        other_vessels = df[time_mask & (df['mmsi'] != target_mmsi)].copy()
        
        if other_vessels.empty:
            return pd.DataFrame()
        
        # Calculate distances
        other_vessels['distance_km'] = other_vessels.apply(
            lambda row: self.haversine_distance(
                target_lat, target_lon,
                row['latitude'], row['longitude']
            ),
            axis=1
        )
        
        # Filter by distance
        neighbors = other_vessels[other_vessels['distance_km'] <= self.spatial_radius_km]
        
        return neighbors
    
    def detect_contextual_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies relative to nearby vessels."""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['timestamp', 'mmsi'])
        
        # Initialize columns
        df['contextual_anomaly'] = False
        df['contextual_anomaly_type'] = 'none'
        df['neighbor_count'] = 0
        df['speed_vs_neighbors'] = np.nan
        df['course_vs_neighbors'] = np.nan
        
        # Sample for efficiency (process every Nth point)
        sample_rate = max(1, len(df) // 5000)
        sample_indices = df.index[::sample_rate]
        
        print(f"  Analyzing {len(sample_indices)} sample points for contextual anomalies...")
        
        for i, idx in enumerate(sample_indices):
            if i % 500 == 0:
                print(f"    Processed {i}/{len(sample_indices)} points...")
            
            row = df.loc[idx]
            
            if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                continue
            
            neighbors = self.find_neighbors(df, idx, row)
            
            df.loc[idx, 'neighbor_count'] = len(neighbors)
            
            if len(neighbors) < self.min_neighbors:
                continue
            
            # Compare speed to neighbors
            if 'sog' in neighbors.columns and not neighbors['sog'].isna().all():
                neighbor_speed_mean = neighbors['sog'].mean()
                neighbor_speed_std = neighbors['sog'].std()
                
                if neighbor_speed_std > 0:
                    speed_z = abs(row['sog'] - neighbor_speed_mean) / neighbor_speed_std
                    df.loc[idx, 'speed_vs_neighbors'] = speed_z
                    
                    if speed_z > self.speed_deviation_threshold:
                        df.loc[idx, 'contextual_anomaly'] = True
                        df.loc[idx, 'contextual_anomaly_type'] = 'speed_deviation_from_traffic'
            
            # Compare course to neighbors
            if 'cog' in neighbors.columns and not neighbors['cog'].isna().all():
                neighbor_cog_mean = neighbors['cog'].mean()
                course_diff = abs(row['cog'] - neighbor_cog_mean)
                course_diff = min(course_diff, 360 - course_diff)
                
                df.loc[idx, 'course_vs_neighbors'] = course_diff
                
                if course_diff > self.course_deviation_threshold:
                    df.loc[idx, 'contextual_anomaly'] = True
                    if df.loc[idx, 'contextual_anomaly_type'] == 'speed_deviation_from_traffic':
                        df.loc[idx, 'contextual_anomaly_type'] = 'combined_traffic_deviation'
                    else:
                        df.loc[idx, 'contextual_anomaly_type'] = 'course_deviation_from_traffic'
        
        return df


# =============================================================================
# TEMPORAL CLUSTERING
# =============================================================================
# Re-exported from event_clustering so the trajectory detector and the
# advanced detector share one implementation. Try relative first (for -m
# invocation), fall back to absolute (for direct python script invocation).
try:
    from .event_clustering import TemporalClusterer  # noqa: F401
except ImportError:  # script invocation: scripts/ is on sys.path directly
    from event_clustering import TemporalClusterer  # noqa: F401


# =============================================================================
# COMBINED SCORING
# =============================================================================

class CombinedScorer:
    """
    Merges trajectory-based and prediction-based anomaly scores.
    """
    
    def __init__(
        self,
        trajectory_weight: float = 0.5,
        prediction_weight: float = 0.3,
        contextual_weight: float = 0.2
    ):
        self.trajectory_weight = trajectory_weight
        self.prediction_weight = prediction_weight
        self.contextual_weight = contextual_weight
    
    def normalize_scores(self, scores: pd.Series) -> pd.Series:
        """Normalize scores to 0-1 range."""
        min_val = scores.min()
        max_val = scores.max()
        
        if max_val - min_val == 0:
            return pd.Series(0, index=scores.index)
        
        return (scores - min_val) / (max_val - min_val)
    
    def calculate_combined_score(
        self,
        trajectory_df: pd.DataFrame,
        prediction_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate combined anomaly score from multiple sources.
        """
        df = trajectory_df.copy()
        
        # Trajectory score (based on severity)
        if 'severity' in df.columns:
            df['trajectory_score'] = self.normalize_scores(df['severity'].fillna(0))
        else:
            df['trajectory_score'] = (df['is_anomaly'] == 1).astype(float)
        
        # Prediction error score (if available)
        df['prediction_score'] = 0.0
        if prediction_df is not None and not prediction_df.empty:
            # Merge prediction errors
            pred_cols = ['mmsi', 'timestamp', 'combined_error', 'is_anomaly']
            pred_subset = prediction_df[[c for c in pred_cols if c in prediction_df.columns]].copy()
            
            if 'combined_error' in pred_subset.columns:
                pred_subset['prediction_score'] = self.normalize_scores(
                    pred_subset['combined_error'].fillna(0)
                )
                
                # Merge on mmsi and approximate timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                pred_subset['timestamp'] = pd.to_datetime(pred_subset['timestamp'])
                
                # Simple merge by mmsi (not perfect but reasonable)
                pred_by_mmsi = pred_subset.groupby('mmsi')['prediction_score'].mean()
                df['prediction_score'] = df['mmsi'].map(pred_by_mmsi).fillna(0)
        
        # Contextual score
        if 'contextual_anomaly' in df.columns:
            df['contextual_score'] = df['contextual_anomaly'].astype(float)
            
            # Add neighbor deviation scores
            if 'speed_vs_neighbors' in df.columns:
                speed_context = self.normalize_scores(df['speed_vs_neighbors'].fillna(0))
                df['contextual_score'] = df['contextual_score'] * 0.5 + speed_context * 0.5
        else:
            df['contextual_score'] = 0.0
        
        # Combined score
        df['combined_anomaly_score'] = (
            self.trajectory_weight * df['trajectory_score'] +
            self.prediction_weight * df['prediction_score'] +
            self.contextual_weight * df['contextual_score']
        )
        
        # Normalize final score
        df['combined_anomaly_score'] = self.normalize_scores(df['combined_anomaly_score'])
        
        # Risk level based on combined score
        df['risk_level'] = pd.cut(
            df['combined_anomaly_score'],
            bins=[-0.01, 0.25, 0.5, 0.75, 1.01],
            labels=['low', 'medium', 'high', 'critical']
        )
        
        return df


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Run the advanced anomaly detection pipeline."""
    
    print("=" * 70)
    print("ADVANCED ANOMALY DETECTION PIPELINE")
    print("=" * 70)
    
    # Check for required files
    if not TRAJECTORY_ANOMALY_PATH.exists():
        print(f"Error: {TRAJECTORY_ANOMALY_PATH} not found.")
        print("Please run trajectory_anomaly_detector.py first.")
        return
    
    # Load trajectory anomalies
    print("\n1. Loading trajectory anomaly data...")
    traj_df = pd.read_csv(TRAJECTORY_ANOMALY_PATH)
    print(f"   Loaded {len(traj_df)} points from {traj_df['mmsi'].nunique()} vessels")
    
    # Load prediction data if available
    pred_df = None
    if PREDICTION_PATH.exists():
        print("\n2. Loading prediction data...")
        pred_df = pd.read_csv(PREDICTION_PATH)
        print(f"   Loaded {len(pred_df)} prediction points")
    else:
        print("\n2. No prediction data found (skipping)")
    
    # Feature Engineering
    print("\n3. Feature Engineering...")
    fe = FeatureEngineer()
    traj_df = fe.engineer_features(traj_df)
    print("   Added temporal features (hour, day_of_week, time_period)")
    print("   Added motion features (acceleration, rate_of_turn)")
    
    # Contextual Anomaly Detection
    print("\n4. Contextual Anomaly Detection...")
    ctx_detector = ContextualAnomalyDetector(
        spatial_radius_km=5.0,
        temporal_window_minutes=30,
        min_neighbors=2
    )
    traj_df = ctx_detector.detect_contextual_anomalies(traj_df)
    ctx_count = traj_df['contextual_anomaly'].sum()
    print(f"   Found {ctx_count} contextual anomalies")
    
    # Temporal Clustering
    print("\n5. Temporal Clustering...")
    clusterer = TemporalClusterer(
        max_gap_seconds=300,
        min_event_points=2
    )
    traj_df, events_df = clusterer.cluster_anomalies(traj_df)
    print(f"   Grouped anomalies into {len(events_df)} events")
    
    # Combined Scoring
    print("\n6. Combined Scoring...")
    scorer = CombinedScorer(
        trajectory_weight=0.5,
        prediction_weight=0.3,
        contextual_weight=0.2
    )
    traj_df = scorer.calculate_combined_score(traj_df, pred_df)
    print("   Calculated combined anomaly scores")
    
    # Save results
    print("\n7. Saving results...")
    
    # Select output columns
    output_cols = [
        'mmsi', 'timestamp', 'latitude', 'longitude', 'sog', 'cog', 'heading',
        'ship_category', 'hour', 'day_of_week', 'time_period', 'is_weekend',
        'speed_change', 'course_change', 'acceleration', 'rate_of_turn',
        'anomaly_type', 'is_anomaly', 'severity',
        'contextual_anomaly', 'contextual_anomaly_type', 'neighbor_count',
        'event_id',
        'trajectory_score', 'prediction_score', 'contextual_score',
        'combined_anomaly_score', 'risk_level'
    ]
    
    # Only include columns that exist
    output_cols = [c for c in output_cols if c in traj_df.columns]
    traj_df[output_cols].to_csv(OUTPUT_PATH, index=False)
    print(f"   Saved advanced anomalies to {OUTPUT_PATH}")
    
    if not events_df.empty:
        events_df.to_csv(EVENTS_PATH, index=False)
        print(f"   Saved anomaly events to {EVENTS_PATH}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal data points: {len(traj_df)}")
    print(f"Trajectory anomalies: {(traj_df['is_anomaly'] == 1).sum()}")
    print(f"Contextual anomalies: {traj_df['contextual_anomaly'].sum()}")
    print(f"Anomaly events: {len(events_df)}")
    
    if 'risk_level' in traj_df.columns:
        print("\nRisk Level Distribution:")
        print(traj_df['risk_level'].value_counts())
    
    if not events_df.empty:
        print("\nTop 5 Longest Events:")
        print(events_df.nlargest(5, 'duration_seconds')[
            ['event_id', 'mmsi', 'duration_seconds', 'point_count', 'dominant_anomaly_type']
        ])


if __name__ == "__main__":
    main()

