import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import {
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  ComposedChart
} from 'recharts';
import { MapContainer, TileLayer, Polyline, CircleMarker, Popup, useMap, Tooltip as LeafletTooltip } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

const API_BASE_URL = 'http://127.0.0.1:8000';

// Anomaly type colors for visualization
const ANOMALY_COLORS = {
  speed_change_critical: '#ff3b5c',
  course_change_critical: '#ff2d4f',
  speed_change_warning: '#ff8c42',
  course_change_warning: '#ffb347',
  speed_too_high: '#a855f7',
  speed_too_low: '#60a5fa',
  loitering: '#34d399',
  drift_detected: '#4ade80',
  zigzag_pattern: '#facc15',
  speed_deviation_from_traffic: '#c084fc',
  course_deviation_from_traffic: '#a78bfa',
  combined_traffic_deviation: '#f43f5e',
  none: '#6b7280'
};

// Risk level colors
const RISK_COLORS = {
  critical: '#ff3b5c',
  high: '#ff8c42',
  medium: '#facc15',
  low: '#4ade80'
};

// Gradient colors for trajectory
const TRAJECTORY_GRADIENT = [
  '#22d3ee', '#06b6d4', '#0891b2', '#0e7490', '#155e75'
];

// Helper to check valid coordinates
const isValidCoord = (lat, lon) => {
  return Number.isFinite(lat) && Number.isFinite(lon) && 
         lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180;
};

// Downsample array for cleaner visualization
const downsample = (arr, maxPoints = 200) => {
  if (!arr || arr.length <= maxPoints) return arr;
  const step = Math.ceil(arr.length / maxPoints);
  const result = [];
  for (let i = 0; i < arr.length; i += step) {
    result.push(arr[i]);
  }
  // Always include last point
  if (result.length > 0 && result[result.length - 1] !== arr[arr.length - 1]) {
    result.push(arr[arr.length - 1]);
  }
  return result;
};

// Fix for default marker icons
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Component to fit map bounds with smooth animation
function FitBounds({ positions }) {
  const map = useMap();

  useEffect(() => {
    if (positions.length > 0) {
      const validPositions = positions.filter(pos => 
        Array.isArray(pos) && pos.length === 2 && 
        Number.isFinite(pos[0]) && Number.isFinite(pos[1]) &&
        pos[0] >= -90 && pos[0] <= 90 && pos[1] >= -180 && pos[1] <= 180
      );
      
      if (validPositions.length > 0) {
        const bounds = L.latLngBounds(validPositions);
        map.fitBounds(bounds, { 
          padding: [50, 50], 
          maxZoom: 15,
          animate: true,
          duration: 0.8
        });
      }
    }
  }, [positions, map]);

  return null;
}

// Vessel direction arrow marker
function DirectionMarker({ position, cog, color = '#00d4ff' }) {
  const map = useMap();
  
  useEffect(() => {
    if (!position || !Number.isFinite(cog)) return;
    
    const arrowSize = 14;
    const arrow = L.divIcon({
      className: 'direction-arrow-icon',
      html: `<div style="
        width: ${arrowSize}px; height: ${arrowSize}px;
        transform: rotate(${cog}deg);
        color: ${color};
        font-size: ${arrowSize}px;
        line-height: 1;
        text-align: center;
      ">▲</div>`,
      iconSize: [arrowSize, arrowSize],
      iconAnchor: [arrowSize / 2, arrowSize / 2],
    });
    
    const marker = L.marker(position, { icon: arrow, interactive: false });
    marker.addTo(map);
    return () => marker.remove();
  }, [position, cog, color, map]);
  
  return null;
}

// Custom tooltip for charts
const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="custom-tooltip">
        <p className="tooltip-label">Point {label}</p>
        <div className="tooltip-values">
          {payload.filter(e => e.name !== 'SOG Area' && e.name !== 'COG Area').map((entry, index) => (
            <p key={index} style={{ color: entry.color }}>
              <span className="tooltip-dot" style={{ background: entry.color }} />
              {entry.name}: {typeof entry.value === 'number' ? entry.value.toFixed(4) : entry.value}
            </p>
          ))}
        </div>
      </div>
    );
  }
  return null;
};

const AnomalyDashboard = () => {
  // Advanced anomaly data (includes everything)
  const [advancedVessels, setAdvancedVessels] = useState([]);
  // Prediction-based data
  const [predictionVessels, setPredictionVessels] = useState([]);
  // Anomaly events
  const [events, setEvents] = useState([]);
  
  const [selectedVessel, setSelectedVessel] = useState(null);
  const [selectedChartIndex, setSelectedChartIndex] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filterAnomaliesOnly, setFilterAnomaliesOnly] = useState(true);
  const [explanation, setExplanation] = useState(null);
  const [explanationLoading, setExplanationLoading] = useState(false);
  const [anomalyTypes, setAnomalyTypes] = useState([]);

  // Fetch all data
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        // Try advanced anomalies first, fall back to trajectory anomalies
        let advancedRes;
        try {
          advancedRes = await axios.get(`${API_BASE_URL}/advanced-anomalies/vessels`, { params: { limit_per_vessel: 500 } });
        } catch {
          // Fall back to basic trajectory anomalies
          advancedRes = await axios.get(`${API_BASE_URL}/trajectory-anomalies/vessels`, { params: { limit_per_vessel: 500 } });
        }
        
        const [predRes, typesRes, eventsRes] = await Promise.all([
          axios.get(`${API_BASE_URL}/predictions/vessels`, { params: { limit_per_vessel: 500 } }),
          axios.get(`${API_BASE_URL}/trajectory-anomalies/types`),
          axios.get(`${API_BASE_URL}/advanced-anomalies/events`, { params: { limit: 100 } }).catch(() => ({ data: { events: [] } }))
        ]);
        
        setAdvancedVessels(advancedRes.data.tracks || []);
        setPredictionVessels(predRes.data.tracks || []);
        setAnomalyTypes(typesRes.data.anomaly_types || []);
        setEvents(eventsRes.data.events || []);
      } catch (err) {
        console.error('Error fetching data:', err);
        setError('Failed to load data. Make sure the backend is running.');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Reset chart selection when vessel changes
  useEffect(() => {
    setSelectedChartIndex(null);
  }, [selectedVessel?.mmsi]);

  // Fetch explanation when vessel is selected
  useEffect(() => {
    const fetchExplanation = async () => {
      if (!selectedVessel?.mmsi) {
        setExplanation(null);
        return;
      }

      setExplanationLoading(true);
      try {
        const res = await axios.get(`${API_BASE_URL}/vessel/${selectedVessel.mmsi}/trajectory-explanation`);
        setExplanation(res.data);
      } catch (err) {
        try {
          const res = await axios.get(`${API_BASE_URL}/vessel/${selectedVessel.mmsi}/explanation`);
          setExplanation(res.data);
        } catch {
          setExplanation(null);
        }
      } finally {
        setExplanationLoading(false);
      }
    };

    fetchExplanation();
  }, [selectedVessel?.mmsi]);

  // Combine vessels from both sources
  const combinedVessels = useMemo(() => {
    const vesselMap = new Map();
    
    // Add advanced/trajectory vessels
    advancedVessels.forEach(v => {
      vesselMap.set(v.mmsi, {
        mmsi: v.mmsi,
        advancedData: v,
        predictionData: null,
        anomalyCount: v.anomaly_count || 0,
        contextualAnomalyCount: v.contextual_anomaly_count || 0,
        eventCount: v.event_count || 0,
        hasAnomalies: (v.anomaly_count || 0) > 0,
        anomalyTypes: v.anomaly_types || {},
        shipCategory: v.ship_category || 'Unknown'
      });
    });
    
    // Merge prediction vessels
    predictionVessels.forEach(v => {
      const points = v.points || [];
      const predAnomalyCount = points.filter(p => p.is_anomaly === 1).length;
      
      if (vesselMap.has(v.mmsi)) {
        const existing = vesselMap.get(v.mmsi);
        existing.predictionData = v;
        existing.predictionAnomalyCount = predAnomalyCount;
      } else {
        vesselMap.set(v.mmsi, {
          mmsi: v.mmsi,
          advancedData: null,
          predictionData: v,
          anomalyCount: predAnomalyCount,
          predictionAnomalyCount: predAnomalyCount,
          hasAnomalies: predAnomalyCount > 0,
          anomalyTypes: {},
          shipCategory: 'Unknown'
        });
      }
    });
    
    return Array.from(vesselMap.values()).sort((a, b) => b.anomalyCount - a.anomalyCount);
  }, [advancedVessels, predictionVessels]);

  // Get events for selected vessel
  const vesselEvents = useMemo(() => {
    if (!selectedVessel?.mmsi) return [];
    return events.filter(e => e.mmsi === selectedVessel.mmsi);
  }, [selectedVessel?.mmsi, events]);

  // Filtered vessel list
  const filteredVessels = useMemo(() => {
    if (filterAnomaliesOnly) {
      return combinedVessels.filter(v => v.hasAnomalies);
    }
    return combinedVessels;
  }, [combinedVessels, filterAnomaliesOnly]);

  // Prepare chart data for selected vessel
  const chartData = useMemo(() => {
    if (!selectedVessel) return [];
    
    const predData = selectedVessel.predictionData;
    const advData = selectedVessel.advancedData;
    
    // Prefer prediction data for real vs predicted comparison
    if (predData && predData.points) {
      return predData.points.map((point, idx) => {
        const advPoint = advData?.points?.[idx];
        return {
          index: idx,
          timestamp: point.timestamp,
          pred_sog: point.pred_sog,
          true_sog: point.true_sog,
          pred_cog: point.pred_cog,
          true_cog: point.true_cog,
          combined_error: point.combined_error,
          is_anomaly: point.is_anomaly || advPoint?.is_anomaly || 0,
          anomaly_type: advPoint?.anomaly_type || 'none',
          combined_anomaly_score: advPoint?.combined_anomaly_score,
          risk_level: advPoint?.risk_level || 'low',
          contextual_anomaly: advPoint?.contextual_anomaly || false
        };
      });
    }
    
    // Fall back to advanced data
    if (advData && advData.points) {
      return advData.points.map((point, idx) => ({
        index: idx,
        timestamp: point.timestamp,
        true_sog: point.sog,
        true_cog: point.cog,
        anomaly_type: point.anomaly_type,
        is_anomaly: point.is_anomaly,
        combined_anomaly_score: point.combined_anomaly_score,
        risk_level: point.risk_level || 'low',
        contextual_anomaly: point.contextual_anomaly || false
      }));
    }
    
    return [];
  }, [selectedVessel]);

  // Get map positions for selected vessel
  const mapPositions = useMemo(() => {
    if (!selectedVessel) return { real: [], predicted: [], anomalies: [] };
    
    const predData = selectedVessel.predictionData;
    const advData = selectedVessel.advancedData;
    
    let real = [];
    let predicted = [];
    let anomalies = [];
    
    // First, try to get trajectory from advanced data (trajectory anomalies)
    if (advData && advData.points && advData.points.length > 0) {
      const validAdvPoints = advData.points
        .map((p, i) => ({ ...p, _origIdx: i }))
        .filter(p => isValidCoord(p.latitude, p.longitude));
      console.log(`Vessel ${selectedVessel.mmsi}: ${validAdvPoints.length} valid points from advData`);

      if (validAdvPoints.length > 0) {
        real = downsample(validAdvPoints, 200).map(p => [p.latitude, p.longitude]);

        // Get anomaly markers, preserving original index for chart linking
        const anomalyPoints = validAdvPoints.filter(p => p.is_anomaly === 1 || p.contextual_anomaly);
        anomalies = anomalyPoints.slice(0, 20).map(p => ({
          position: [p.latitude, p.longitude],
          anomaly_type: p.contextual_anomaly ? p.contextual_anomaly_type : p.anomaly_type,
          combined_score: p.combined_anomaly_score,
          risk_level: p.risk_level,
          contextual: p.contextual_anomaly,
          timestamp: p.timestamp,
          dataIndex: p._origIdx
        }));
      }
    }
    
    // If no real trajectory yet, try prediction data
    if (real.length === 0 && predData && predData.points && predData.points.length > 0) {
      const validRealPoints = predData.points.filter(p => isValidCoord(p.true_latitude, p.true_longitude));
      console.log(`Vessel ${selectedVessel.mmsi}: ${validRealPoints.length} valid points from predData`);
      
      if (validRealPoints.length > 0) {
        real = downsample(validRealPoints, 200).map(p => [p.true_latitude, p.true_longitude]);
      }
      
      const validPredPoints = predData.points.filter(p => isValidCoord(p.pred_latitude, p.pred_longitude));
      if (validPredPoints.length > 0) {
        predicted = downsample(validPredPoints, 200).map(p => [p.pred_latitude, p.pred_longitude]);
      }
    }
    
    console.log(`MapPositions: real=${real.length}, predicted=${predicted.length}, anomalies=${anomalies.length}`);
    if (real.length > 0) {
      console.log(`First point: [${real[0][0]}, ${real[0][1]}]`);
    }
    
    return { real, predicted, anomalies };
  }, [selectedVessel]);

  // Get anomaly type label
  const getAnomalyLabel = (type) => {
    const found = anomalyTypes.find(t => t.type === type);
    return found ? found.label : type;
  };

  // Map a clicked anomaly marker to the correct chartData index.
  // Anomaly markers always come from advData, but chartData may be built from
  // predData (different length/order). Timestamp is the reliable shared key.
  // Falls back to dataIndex (advData position) when timestamps are unavailable.
  const findChartIndex = (anomaly) => {
    if (anomaly.timestamp) {
      const match = chartData.findIndex(d => d.timestamp === anomaly.timestamp);
      if (match !== -1) return chartData[match].index;
    }
    // Fallback: advData-only path — indices are directly aligned
    return anomaly.dataIndex;
  };

  if (loading) {
    return (
      <div className="anomaly-dashboard loading-state">
        <div className="loading-spinner"></div>
        <p>Loading vessel data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="anomaly-dashboard error-state">
        <div className="error-icon">⚠️</div>
        <p>{error}</p>
        <button onClick={() => window.location.reload()} className="retry-btn">
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="anomaly-dashboard">
      {/* Sidebar - Vessel List */}
      <aside className="vessel-sidebar">
        <div className="sidebar-header">
          <h2>🚢 Vessels</h2>
          <div className="filter-toggle">
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={filterAnomaliesOnly}
                onChange={(e) => setFilterAnomaliesOnly(e.target.checked)}
              />
              <span className="toggle-slider" />
              <span className="toggle-label">Anomalies only</span>
            </label>
          </div>
          <div className="vessel-count">
            {filteredVessels.length} vessel{filteredVessels.length !== 1 ? 's' : ''}
            {filterAnomaliesOnly && ` with anomalies`}
          </div>
        </div>
        
        <div className="vessel-list">
          {filteredVessels.map((vessel) => {
            const riskLevel = vessel.anomalyCount >= 10 ? 'critical' : vessel.anomalyCount >= 5 ? 'high' : vessel.anomalyCount >= 2 ? 'medium' : 'low';
            return (
              <div
                key={vessel.mmsi}
                className={`vessel-card ${selectedVessel?.mmsi === vessel.mmsi ? 'selected' : ''} ${vessel.hasAnomalies ? 'has-anomaly' : ''}`}
                onClick={() => setSelectedVessel(vessel)}
              >
                <div className="vessel-card-header">
                  <span className="vessel-mmsi">MMSI: {vessel.mmsi}</span>
                  {vessel.hasAnomalies && (
                    <span className={`anomaly-badge badge-${riskLevel}`}>{vessel.anomalyCount}</span>
                  )}
                </div>
                {vessel.shipCategory !== 'Unknown' && (
                  <div className="vessel-category">{vessel.shipCategory}</div>
                )}
                {vessel.hasAnomalies && (
                  <div className="vessel-anomaly-types">
                    {Object.entries(vessel.anomalyTypes || {}).slice(0, 3).map(([type, count]) => (
                      <span key={type} className="mini-type-tag" style={{ borderColor: ANOMALY_COLORS[type] || '#6b7280' }}>
                        {count}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        {selectedVessel ? (
          <>
            {/* Vessel Header with Stats */}
            <div className="vessel-header">
              <div className="vessel-header-row">
                <h2>Vessel {selectedVessel.mmsi}</h2>
                {selectedVessel.shipCategory !== 'Unknown' && (
                  <span className="ship-category-badge">{selectedVessel.shipCategory}</span>
                )}
              </div>
              <div className="vessel-stats-bar">
                <div className="stat-chip">
                  <span className="stat-value">{selectedVessel.advancedData?.sample_count ?? chartData.length}</span>
                  <span className="stat-label">Data Points</span>
                </div>
                <div className={`stat-chip ${selectedVessel.anomalyCount > 0 ? 'stat-alert' : ''}`}>
                  <span className="stat-value">{selectedVessel.anomalyCount}</span>
                  <span className="stat-label">Anomalies</span>
                </div>
                {selectedVessel.contextualAnomalyCount > 0 && (
                  <div className="stat-chip stat-contextual">
                    <span className="stat-value">{selectedVessel.contextualAnomalyCount}</span>
                    <span className="stat-label">Contextual</span>
                  </div>
                )}
                {vesselEvents.length > 0 && (
                  <div className="stat-chip">
                    <span className="stat-value">{vesselEvents.length}</span>
                    <span className="stat-label">Events</span>
                  </div>
                )}
              </div>
            </div>

            {/* Content Layout: Map/Charts on left, Explanation on right */}
            <div className="content-layout">
              {/* Left Side - Map and Charts */}
              <div className="visualization-area">
                {/* Interactive Map */}
                <div className="map-section">
                  <h3>📍 Trajectory Map</h3>
                  {mapPositions.real.length >= 2 ? (
                    <>
                      <div className="map-legend-bar">
                        <span className="legend-chip real"><span className="legend-line solid" />Actual Track</span>
                        {mapPositions.predicted.length >= 2 && (
                          <span className="legend-chip predicted"><span className="legend-line dashed" />LSTM Predicted</span>
                        )}
                        {mapPositions.anomalies.length > 0 && (
                          <span className="legend-chip anomaly"><span className="legend-dot" />{mapPositions.anomalies.length} Anomalies</span>
                        )}
                        <span className="legend-chip start-chip"><span className="legend-dot start" />Start</span>
                        <span className="legend-chip end-chip"><span className="legend-dot end" />End</span>
                      </div>
                      <div className="map-container">
                        <MapContainer
                          center={mapPositions.real[0] || [30, -80]}
                          zoom={12}
                          maxZoom={18}
                          style={{ height: '100%', width: '100%' }}
                          scrollWheelZoom={true}
                          zoomSnap={0.5}
                        >
                          {/* CartoDB Voyager - high zoom support with water detail */}
                          <TileLayer
                            attribution='&copy; <a href="https://carto.com/">CARTO</a> &copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>'
                            url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
                            maxZoom={20}
                            subdomains="abcd"
                          />
                          
                          <FitBounds positions={mapPositions.real} />

                          {/* Real trajectory - chunked gradient effect */}
                          {mapPositions.real.length >= 2 && (() => {
                            const pts = mapPositions.real;
                            const segments = [];
                            for (let i = 0; i < pts.length - 1; i++) {
                              const ratio = i / Math.max(1, pts.length - 2);
                              const colorIdx = Math.min(
                                Math.floor(ratio * TRAJECTORY_GRADIENT.length),
                                TRAJECTORY_GRADIENT.length - 1
                              );
                              segments.push(
                                <Polyline
                                  key={`traj-${i}`}
                                  positions={[pts[i], pts[i + 1]]}
                                  color={TRAJECTORY_GRADIENT[colorIdx]}
                                  weight={4}
                                  opacity={0.9}
                                  lineCap="round"
                                  lineJoin="round"
                                />
                              );
                            }
                            return segments;
                          })()}
                          
                          {/* Trajectory glow effect (wider, more transparent line behind) */}
                          <Polyline
                            positions={mapPositions.real}
                            color="#22d3ee"
                            weight={10}
                            opacity={0.15}
                            lineCap="round"
                          />

                          {/* Predicted trajectory - dashed cyan-blue */}
                          {mapPositions.predicted.length >= 2 && (
                            <>
                              <Polyline 
                                positions={mapPositions.predicted} 
                                color="#818cf8" 
                                weight={3} 
                                opacity={0.9} 
                                dashArray="10 8"
                                lineCap="round"
                              >
                                <Popup>
                                  <div className="map-popup">
                                    <strong className="popup-title predicted">📊 LSTM Predicted Trajectory</strong>
                                    <div className="popup-stat">Points: {mapPositions.predicted.length}</div>
                                  </div>
                                </Popup>
                              </Polyline>
                              {/* Predicted glow */}
                              <Polyline
                                positions={mapPositions.predicted}
                                color="#818cf8"
                                weight={8}
                                opacity={0.1}
                              />
                            </>
                          )}

                          {/* Anomaly markers - pulsing red circles with detailed popups */}
                          {mapPositions.anomalies.map((anomaly, idx) => {
                            const markerColor = anomaly.risk_level
                              ? RISK_COLORS[anomaly.risk_level] || '#ff3b5c'
                              : ANOMALY_COLORS[anomaly.anomaly_type] || '#ff3b5c';
                            const isSelected = selectedChartIndex === anomaly.dataIndex;
                            return (
                              <CircleMarker
                                key={`anomaly-${idx}`}
                                center={anomaly.position}
                                radius={isSelected ? 14 : 10}
                                fillColor={markerColor}
                                color={isSelected ? '#ffffff' : 'rgba(255,255,255,0.8)'}
                                weight={isSelected ? 3 : 2}
                                opacity={1}
                                fillOpacity={0.85}
                                eventHandlers={{ click: () => setSelectedChartIndex(findChartIndex(anomaly)) }}
                              >
                                <LeafletTooltip direction="top" offset={[0, -10]} opacity={0.95}>
                                  <span style={{ fontSize: '11px', fontWeight: 600 }}>
                                    ⚠ {getAnomalyLabel(anomaly.anomaly_type)}
                                  </span>
                                </LeafletTooltip>
                                <Popup>
                                  <div className="map-popup">
                                    <strong className="popup-title anomaly">⚠️ {getAnomalyLabel(anomaly.anomaly_type)}</strong>
                                    {anomaly.risk_level && (
                                      <div className={`popup-risk risk-${anomaly.risk_level}`}>
                                        {anomaly.risk_level.toUpperCase()} RISK
                                      </div>
                                    )}
                                    {anomaly.combined_score != null && (
                                      <div className="popup-stat">Score: <strong>{anomaly.combined_score.toFixed(3)}</strong></div>
                                    )}
                                    {anomaly.contextual && (
                                      <div className="popup-stat contextual">Contextual anomaly (vs nearby traffic)</div>
                                    )}
                                    {anomaly.timestamp && (
                                      <div className="popup-stat time">{anomaly.timestamp}</div>
                                    )}
                                  </div>
                                </Popup>
                              </CircleMarker>
                            );
                          })}

                          {/* Anomaly halo rings for emphasis */}
                          {mapPositions.anomalies.map((anomaly, idx) => (
                            <CircleMarker
                              key={`halo-${idx}`}
                              center={anomaly.position}
                              radius={18}
                              fillColor="transparent"
                              color={RISK_COLORS[anomaly.risk_level] || '#ff3b5c'}
                              weight={1.5}
                              opacity={0.4}
                              fillOpacity={0}
                              interactive={false}
                            />
                          ))}

                          {/* Start marker - green with label */}
                          <CircleMarker 
                            center={mapPositions.real[0]} 
                            radius={9} 
                            fillColor="#22c55e" 
                            color="#ffffff" 
                            weight={3} 
                            opacity={1} 
                            fillOpacity={1}
                          >
                            <LeafletTooltip permanent direction="right" offset={[10, 0]} className="marker-label">
                              Start
                            </LeafletTooltip>
                            <Popup><div className="map-popup"><strong className="popup-title start">🟢 Trajectory Start</strong></div></Popup>
                          </CircleMarker>

                          {/* End marker - violet */}
                          <CircleMarker 
                            center={mapPositions.real[mapPositions.real.length - 1]} 
                            radius={9} 
                            fillColor="#a855f7" 
                            color="#ffffff" 
                            weight={3} 
                            opacity={1} 
                            fillOpacity={1}
                          >
                            <LeafletTooltip permanent direction="right" offset={[10, 0]} className="marker-label">
                              End
                            </LeafletTooltip>
                            <Popup><div className="map-popup"><strong className="popup-title end">🟣 Trajectory End</strong></div></Popup>
                          </CircleMarker>

                          {/* Direction arrows along trajectory */}
                          {(() => {
                            const pts = mapPositions.real;
                            if (pts.length < 4) return null;
                            const step = Math.max(1, Math.floor(pts.length / 6));
                            const arrows = [];
                            for (let i = step; i < pts.length - 1; i += step) {
                              const dlat = pts[i+1 < pts.length ? i+1 : i][0] - pts[i-1][0];
                              const dlon = pts[i+1 < pts.length ? i+1 : i][1] - pts[i-1][1];
                              const bearing = (Math.atan2(dlon, dlat) * 180 / Math.PI + 360) % 360;
                              arrows.push(
                                <DirectionMarker key={`arrow-${i}`} position={pts[i]} cog={bearing} color="#22d3ee" />
                              );
                            }
                            return arrows;
                          })()}
                        </MapContainer>
                      </div>
                    </>
                  ) : (
                    <div className="map-container map-empty">
                      <div className="empty-map-content">
                        <div className="empty-icon">🗺️</div>
                        <p className="empty-title">No trajectory data available</p>
                        <p className="empty-subtitle">At least 2 valid position points required for visualization.</p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Charts Section */}
                <div className="charts-section">
                  <div className="chart-panel">
                    <h3>⚡ Speed Over Ground (SOG)</h3>
                    <ResponsiveContainer width="100%" height={170}>
                      <ComposedChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                        <defs>
                          <linearGradient id="sogGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.3}/>
                            <stop offset="95%" stopColor="#22d3ee" stopOpacity={0.02}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                        <XAxis dataKey="index" tick={{ fontSize: 10, fill: '#6b7280' }} axisLine={{ stroke: '#374151' }} />
                        <YAxis tick={{ fontSize: 10, fill: '#6b7280' }} axisLine={{ stroke: '#374151' }} />
                        <Tooltip content={<CustomTooltip />} />
                        <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '4px' }} />
                        <Area type="monotone" dataKey="true_sog" fill="url(#sogGradient)" stroke="none" name="SOG Area" legendType="none" />
                        <Line type="monotone" dataKey="true_sog" stroke="#22d3ee" strokeWidth={2} name="Actual SOG"
                          dot={(p) => p.index === selectedChartIndex
                            ? <circle key="sel-sog" cx={p.cx} cy={p.cy} r={7} fill="#22d3ee" stroke="white" strokeWidth={2} />
                            : null}
                          activeDot={{ r: 5 }}
                        />
                        {chartData.some(d => d.pred_sog !== undefined) && (
                          <Line type="monotone" dataKey="pred_sog" stroke="#818cf8" strokeWidth={2} strokeDasharray="6 4" name="Predicted SOG"
                            dot={(p) => p.index === selectedChartIndex
                              ? <circle key="sel-pred-sog" cx={p.cx} cy={p.cy} r={7} fill="#818cf8" stroke="white" strokeWidth={2} />
                              : null}
                            activeDot={{ r: 5 }}
                          />
                        )}
                        {chartData.filter(d => d.is_anomaly === 1 || d.contextual_anomaly).map((d, idx) => (
                          <ReferenceLine key={idx} x={d.index} stroke={d.risk_level ? RISK_COLORS[d.risk_level] : '#ff3b5c'} strokeWidth={1.5} strokeOpacity={0.5} />
                        ))}
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="chart-panel">
                    <h3>🧭 Course Over Ground (COG)</h3>
                    <ResponsiveContainer width="100%" height={170}>
                      <ComposedChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                        <defs>
                          <linearGradient id="cogGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#a78bfa" stopOpacity={0.25}/>
                            <stop offset="95%" stopColor="#a78bfa" stopOpacity={0.02}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                        <XAxis dataKey="index" tick={{ fontSize: 10, fill: '#6b7280' }} axisLine={{ stroke: '#374151' }} />
                        <YAxis tick={{ fontSize: 10, fill: '#6b7280' }} domain={[0, 360]} axisLine={{ stroke: '#374151' }} />
                        <Tooltip content={<CustomTooltip />} />
                        <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '4px' }} />
                        <Area type="monotone" dataKey="true_cog" fill="url(#cogGradient)" stroke="none" name="COG Area" legendType="none" />
                        <Line type="monotone" dataKey="true_cog" stroke="#a78bfa" strokeWidth={2} name="Actual COG"
                          dot={(p) => p.index === selectedChartIndex
                            ? <circle key="sel-cog" cx={p.cx} cy={p.cy} r={7} fill="#a78bfa" stroke="white" strokeWidth={2} />
                            : null}
                          activeDot={{ r: 5 }}
                        />
                        {chartData.some(d => d.pred_cog !== undefined) && (
                          <Line type="monotone" dataKey="pred_cog" stroke="#f472b6" strokeWidth={2} strokeDasharray="6 4" name="Predicted COG"
                            dot={(p) => p.index === selectedChartIndex
                              ? <circle key="sel-pred-cog" cx={p.cx} cy={p.cy} r={7} fill="#f472b6" stroke="white" strokeWidth={2} />
                              : null}
                            activeDot={{ r: 5 }}
                          />
                        )}
                        {chartData.filter(d => d.is_anomaly === 1 || d.contextual_anomaly).map((d, idx) => (
                          <ReferenceLine key={idx} x={d.index} stroke={d.risk_level ? RISK_COLORS[d.risk_level] : '#ff3b5c'} strokeWidth={1.5} strokeOpacity={0.5} />
                        ))}
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

            </div>
          </>
        ) : (
          <div className="no-selection">
            <div className="no-selection-visual">
              <div className="no-selection-icon">🚢</div>
              <div className="radar-ring ring-1"></div>
              <div className="radar-ring ring-2"></div>
              <div className="radar-ring ring-3"></div>
            </div>
            <h2>Select a Vessel</h2>
            <p>Choose a vessel from the sidebar to view its trajectory, predictions, anomaly analysis, and AI-generated explanations.</p>
            <div className="no-selection-stats">
              <div className="quick-stat">
                <span className="qs-value">{combinedVessels.length}</span>
                <span className="qs-label">Total Vessels</span>
              </div>
              <div className="quick-stat">
                <span className="qs-value">{combinedVessels.filter(v => v.hasAnomalies).length}</span>
                <span className="qs-label">With Anomalies</span>
              </div>
              <div className="quick-stat">
                <span className="qs-value">{events.length}</span>
                <span className="qs-label">Events</span>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* AI Explanation Sidebar — sibling of main-content */}
      <aside className="explanation-sidebar">
        <div className="explanation-panel">
          <h3>🤖 AI Anomaly Explanation</h3>
          {explanationLoading ? (
            <div className="explanation-loading">
              <div className="loading-spinner small"></div>
              <span>Generating explanation...</span>
            </div>
          ) : explanation ? (
            <div className="explanation-content">
              <div className={`risk-badge risk-${explanation.risk_level}`}>
                {explanation.risk_level.toUpperCase()} RISK
              </div>

              <div className="explanation-summary">
                <p>{explanation.summary}</p>
              </div>

              <div className="explanation-detail">
                <h4>Analysis</h4>
                <p>{explanation.detailed_explanation}</p>
              </div>

              {vesselEvents.length > 0 && (
                <div className="events-section">
                  <h4>📅 Anomaly Events ({vesselEvents.length})</h4>
                  <div className="events-list">
                    {vesselEvents.slice(0, 3).map((event, idx) => (
                      <div key={idx} className="event-item">
                        <span className="event-type">{event.dominant_anomaly_type}</span>
                        <span className="event-duration">{Math.round(event.duration_seconds / 60)} min</span>
                        <span className="event-points">{event.point_count} pts</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {explanation.anomaly_breakdown && explanation.anomaly_breakdown.length > 0 && (
                <div className="anomaly-breakdown">
                  <h4>Anomaly Breakdown</h4>
                  <div className="breakdown-list">
                    {explanation.anomaly_breakdown.map((item, idx) => (
                      <div key={idx} className={`breakdown-item severity-${item.severity}`}>
                        <div className="breakdown-header">
                          <span className="breakdown-type">{item.label}</span>
                          <span className="breakdown-count">{item.count}</span>
                        </div>
                        <p className="breakdown-desc">{item.description}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {explanation.recommendations && explanation.recommendations.length > 0 && (
                <div className="recommendations">
                  <h4>Recommendations</h4>
                  <ul>
                    {explanation.recommendations.map((rec, idx) => (
                      <li key={idx}>{rec}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            <div className="no-explanation">
              <p>No anomalies detected for this vessel, or explanation not available.</p>
            </div>
          )}
        </div>
      </aside>
    </div>
  );
};

export default AnomalyDashboard;
