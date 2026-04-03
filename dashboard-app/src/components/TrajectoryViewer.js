import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import TrajectoryMap from './TrajectoryMap';

const API_BASE_URL = 'http://127.0.0.1:8000';

const TrajectoryViewer = () => {
  const [trajectories, setTrajectories] = useState([]);
  const [mmsiOptions, setMmsiOptions] = useState([]);
  const [mmsiFilter, setMmsiFilter] = useState('');
  const [selectedTrajectory, setSelectedTrajectory] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [anomalyQueue, setAnomalyQueue] = useState([]);
  const [anomalyQueueError, setAnomalyQueueError] = useState(null);
  const [predictionTracks, setPredictionTracks] = useState([]);
  const [anomalyPredictions, setAnomalyPredictions] = useState([]);
  const [predictionDataError, setPredictionDataError] = useState(null);

  useEffect(() => {
    fetchMmsiOptions();
    fetchTrajectories('');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    fetchAnomalyQueue();
    const interval = setInterval(fetchAnomalyQueue, 10000);
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    fetchPredictionInsights();
  }, []);

  const fetchMmsiOptions = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/trajectories/mmsi`, {
        params: { limit: 500 }
      });
      setMmsiOptions(response.data.mmsi || []);
    } catch (err) {
      console.error('Error fetching MMSI list:', err);
      setMmsiOptions([]);
    }
  };

  const fetchTrajectories = async (overrideMmsi) => {
    setLoading(true);
    setError(null);
    try {
      const params = {};
      const filterValue =
        typeof overrideMmsi === 'string' || typeof overrideMmsi === 'number'
          ? overrideMmsi
          : mmsiFilter;
      if (filterValue) {
        const parsed = Number(filterValue);
        if (!Number.isNaN(parsed)) {
          params.mmsi = parsed;
        }
      }
      params.limit = 50; // Limit for performance
      params.include_coordinates = true; // Request coordinates for map
      
      const response = await axios.get(`${API_BASE_URL}/trajectories`, { params });
      setTrajectories(response.data.trajectories || []);
    } catch (err) {
      setError(err.response?.data?.detail || 'Error loading trajectories');
      console.error('Error fetching trajectories:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchAnomalyQueue = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/anomalies/queue`, {
        params: { limit: 25 }
      });
      setAnomalyQueue(response.data.alerts || []);
      setAnomalyQueueError(null);
    } catch (err) {
      const detail = err.response?.data?.detail;
      if (detail && detail.includes('not trained')) {
        setAnomalyQueue([]);
        setAnomalyQueueError('Anomaly detector not trained yet.');
      } else {
        setAnomalyQueueError('Unable to load anomaly alerts.');
      }
    }
  };

  const fetchPredictionInsights = async () => {
    setPredictionDataError(null);
    try {
      const [tracksRes, anomalyRes] = await Promise.allSettled([
        axios.get(`${API_BASE_URL}/predictions/vessels`, {
          params: { limit_per_vessel: 150 }
        }),
        axios.get(`${API_BASE_URL}/anomalies/predictions`, {
          params: { limit: 250 }
        })
      ]);

      const errors = [];

      if (tracksRes.status === 'fulfilled') {
        setPredictionTracks(tracksRes.value.data.tracks || []);
      } else {
        setPredictionTracks([]);
        errors.push('vessel predictions');
      }

      if (anomalyRes.status === 'fulfilled') {
        setAnomalyPredictions(anomalyRes.value.data.predictions || []);
      } else {
        setAnomalyPredictions([]);
        errors.push('anomaly predictions');
      }

      if (errors.length) {
        setPredictionDataError(`Unavailable: ${errors.join(', ')}`);
      }
    } catch (err) {
      console.error('Error fetching prediction insights:', err);
      setPredictionTracks([]);
      setAnomalyPredictions([]);
      setPredictionDataError('Unable to load prediction data.');
    }
  };

  const handleTrajectoryClick = (trajectory) => {
    setSelectedTrajectory(trajectory);
  };

  const prepareChartData = (points) => {
    return points.map((point, idx) => ({
      index: idx,
      row_index: point.row_index,
      sog: point.sog,
      cog: point.cog,
      heading: point.heading,
      sog_normalized: point.sog_normalized,
      cog_normalized: point.cog_normalized,
      heading_normalized: point.heading_normalized,
    }));
  };

  if (loading) {
    return <div className="loading">Loading trajectories...</div>;
  }

  if (error) {
    return <div className="error">Error: {error}</div>;
  }

  return (
    <div className="trajectory-viewer">
      <div className="controls">
        <h2>AIS Trajectory Viewer</h2>
        <div className="filter-section">
          <label htmlFor="mmsi-filter">Filter by MMSI: </label>
          <div className="mmsi-filter">
            <input
              id="mmsi-filter"
              list="mmsi-options"
              placeholder="Enter MMSI or leave empty for all"
              value={mmsiFilter}
              onChange={(e) => setMmsiFilter(e.target.value)}
            />
            <datalist id="mmsi-options">
              {mmsiOptions.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.id} ({option.count})
                </option>
              ))}
            </datalist>
            <button onClick={() => fetchTrajectories(mmsiFilter)} className="refresh-btn">
              Apply
            </button>
            {mmsiFilter && (
              <button
                onClick={() => {
                  setMmsiFilter('');
                  fetchTrajectories('');
                }}
                className="refresh-btn secondary"
              >
                Clear
              </button>
            )}
          </div>
        </div>
        <div className="info">
          <p>Showing {trajectories.length} trajectories</p>
        </div>

        <div className="anomaly-alerts-panel">
          <div className="anomaly-alerts-header">
            <h3>Anomaly Alerts</h3>
            <button onClick={fetchAnomalyQueue} className="refresh-btn small">
              Refresh
            </button>
          </div>
          {anomalyQueueError ? (
            <div className="anomaly-alert-error">{anomalyQueueError}</div>
          ) : anomalyQueue.length === 0 ? (
            <p className="anomaly-alert-empty">No active anomalies.</p>
          ) : (
            <ul className="anomaly-alert-list">
              {anomalyQueue.map((alert) => (
                <li
                  key={alert.alert_id}
                  className="anomaly-alert-item"
                  onClick={() => {
                    const match = trajectories.find(
                      (traj) => traj.segment_id === alert.segment_id
                    );
                    if (match) {
                      setSelectedTrajectory(match);
                    }
                  }}
                >
                  <div className="anomaly-alert-main">
                    <span className="anomaly-badge">Alert</span>
                    <span>
                      Segment {alert.segment_id}{' '}
                      {alert.mmsi && `· MMSI ${alert.mmsi}`}
                    </span>
                  </div>
                  <div className="anomaly-alert-meta">
                    <span>Score: {alert.score?.toFixed(3)}</span>
                    <span>Threshold: {alert.threshold?.toFixed(2)}</span>
                    <span>
                      {alert.detected_at
                        ? new Date(alert.detected_at).toLocaleString()
                        : 'Recently'}
                    </span>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <div className="map-view-section" style={{ marginBottom: '20px' }}>
        <div className="map-header">
          <h3>Vessel Trajectories Map</h3>
          <button onClick={fetchPredictionInsights} className="refresh-btn small">
            Refresh prediction overlays
          </button>
        </div>
        {predictionDataError && (
          <div className="warning" style={{ marginBottom: '10px' }}>
            {predictionDataError}
          </div>
        )}
        <p style={{ fontSize: '12px', color: '#666', marginBottom: '10px' }}>
          Real vs predicted trajectories per vessel with anomaly markers. Green lines = actual, blue dashed = predicted.
        </p>
        <div style={{ height: '600px', width: '100%' }}>
          <TrajectoryMap
            trajectories={trajectories}
            selectedTrajectory={selectedTrajectory}
            onTrajectoryClick={handleTrajectoryClick}
            predictionTracks={predictionTracks}
            anomalyPredictions={anomalyPredictions}
          />
        </div>
      </div>

      <div className="content">
        <div className="trajectory-list">
          <h3>Trajectories</h3>
          <div className="trajectory-items">
            {trajectories.map((traj) => (
              <div
                key={traj.segment_id}
                className={`trajectory-item ${
                  selectedTrajectory?.segment_id === traj.segment_id ? 'selected' : ''
                } ${traj.anomaly?.is_anomalous ? 'anomalous' : ''}`}
                onClick={() => handleTrajectoryClick(traj)}
              >
                <div className="trajectory-header">
                  <strong>Segment {traj.segment_id}</strong>
                  {traj.anomaly?.is_anomalous && (
                    <span className="anomaly-badge">Anomaly</span>
                  )}
                </div>
                <div className="trajectory-info">
                  <span>MMSI: {traj.mmsi || 'N/A'}</span>
                  <span>Points: {traj.points.length}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="trajectory-detail">
          {selectedTrajectory ? (
            <div>
              <h3>Trajectory {selectedTrajectory.segment_id}</h3>

              {selectedTrajectory.anomaly?.is_anomalous && (
                <div className="anomaly-banner">
                  <strong>Anomalous behaviour detected</strong>
                  <span>
                    Score:{' '}
                    {selectedTrajectory.anomaly.score !== undefined
                      ? selectedTrajectory.anomaly.score.toFixed(3)
                      : 'N/A'}{' '}
                    | Threshold:{' '}
                    {selectedTrajectory.anomaly.threshold !== undefined
                      ? selectedTrajectory.anomaly.threshold.toFixed(2)
                      : 'N/A'}
                  </span>
                </div>
              )}
              {selectedTrajectory.anomaly && !selectedTrajectory.anomaly.is_anomalous && (
                <div className="anomaly-banner neutral">
                  <span>
                    Recent anomaly score:{' '}
                    {selectedTrajectory.anomaly.score !== undefined
                      ? selectedTrajectory.anomaly.score.toFixed(3)
                      : 'N/A'}{' '}
                    (below alert threshold{' '}
                    {selectedTrajectory.anomaly.threshold !== undefined
                      ? selectedTrajectory.anomaly.threshold.toFixed(2)
                      : 'N/A'}
                    )
                  </span>
                </div>
              )}
              
              <div className="charts-container">
                <div className="chart-section">
                  <h4>Speed Over Ground (SOG)</h4>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={prepareChartData(selectedTrajectory.points)}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="index" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="sog"
                        stroke="#8884d8"
                        name="SOG (knots)"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                <div className="chart-section">
                  <h4>Course Over Ground (COG)</h4>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={prepareChartData(selectedTrajectory.points)}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="index" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="cog"
                        stroke="#82ca9d"
                        name="COG (degrees)"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          ) : (
            <div className="no-selection">
              <p>Select a trajectory from the list to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TrajectoryViewer;

