"""
Temporal clustering of consecutive anomalies into events.

Shared by the trajectory and advanced anomaly detectors so both can collapse
per-point flags into higher-level behavioral incidents instead of reporting
hundreds of anomalies per vessel.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd


class TemporalClusterer:
    """Group consecutive anomalies into events, separated by time gaps."""

    def __init__(
        self,
        max_gap_seconds: int = 300,
        min_event_points: int = 2,
    ):
        self.max_gap_seconds = max_gap_seconds
        self.min_event_points = min_event_points

    def cluster_anomalies(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Assign an event_id to consecutive anomaly rows separated by no more than
        max_gap_seconds. Single isolated flags (below min_event_points) are not
        promoted into events and keep event_id = None.

        Args:
            df: rows with at least mmsi, timestamp, is_anomaly columns.

        Returns:
            df: input with a new event_id column.
            events_df: one row per event with start/end/duration/center.
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['event_id'] = None

        events = []
        event_counter = 0

        for mmsi in df['mmsi'].unique():
            vessel_df = df[df['mmsi'] == mmsi].sort_values('timestamp')
            anomaly_mask = vessel_df['is_anomaly'] == 1

            if not anomaly_mask.any():
                continue

            anomaly_indices = vessel_df[anomaly_mask].index.tolist()
            if not anomaly_indices:
                continue

            current_event = [anomaly_indices[0]]

            for i in range(1, len(anomaly_indices)):
                prev_idx = anomaly_indices[i - 1]
                curr_idx = anomaly_indices[i]

                prev_time = vessel_df.loc[prev_idx, 'timestamp']
                curr_time = vessel_df.loc[curr_idx, 'timestamp']
                time_gap = (curr_time - prev_time).total_seconds()

                if time_gap <= self.max_gap_seconds:
                    current_event.append(curr_idx)
                    continue

                if len(current_event) >= self.min_event_points:
                    event_counter += 1
                    event_id = f"{mmsi}_{event_counter}"
                    for idx in current_event:
                        df.loc[idx, 'event_id'] = event_id
                    events.append(self._create_event_summary(
                        event_id, mmsi, vessel_df.loc[current_event]
                    ))

                current_event = [curr_idx]

            if len(current_event) >= self.min_event_points:
                event_counter += 1
                event_id = f"{mmsi}_{event_counter}"
                for idx in current_event:
                    df.loc[idx, 'event_id'] = event_id
                events.append(self._create_event_summary(
                    event_id, mmsi, vessel_df.loc[current_event]
                ))

        events_df = pd.DataFrame(events)
        return df, events_df

    @staticmethod
    def _create_event_summary(event_id, mmsi, event_data: pd.DataFrame) -> dict:
        start_time = event_data['timestamp'].min()
        end_time = event_data['timestamp'].max()
        duration = (end_time - start_time).total_seconds()

        if 'anomaly_type' in event_data.columns and len(event_data) > 0:
            anomaly_types = event_data['anomaly_type'].value_counts().to_dict()
            dominant_type = event_data['anomaly_type'].mode().iloc[0]
        else:
            anomaly_types = {}
            dominant_type = 'unknown'

        return {
            'event_id': event_id,
            'mmsi': mmsi,
            'start_time': start_time,
            'end_time': end_time,
            'duration_seconds': duration,
            'point_count': len(event_data),
            'dominant_anomaly_type': dominant_type,
            'anomaly_types': anomaly_types,
            'center_latitude': event_data['latitude'].mean()
                if 'latitude' in event_data.columns else None,
            'center_longitude': event_data['longitude'].mean()
                if 'longitude' in event_data.columns else None,
            'max_severity': int(event_data['severity'].max())
                if 'severity' in event_data.columns else 0,
            'max_speed_change': event_data['speed_change'].max()
                if 'speed_change' in event_data.columns else None,
            'max_course_change': event_data['course_change'].max()
                if 'course_change' in event_data.columns else None,
        }
