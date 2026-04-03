import React, { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Polyline, Marker, CircleMarker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix for default marker icons in react-leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

const trajectoryColors = [
  '#0d47a1',
  '#1b5e20',
  '#e65100',
  '#4a148c',
  '#006064',
  '#880e4f',
  '#33691e',
  '#3e2723',
  '#01579b',
  '#ff6f00',
  '#c62828',
  '#2e7d32'
];

const getTrajectoryColor = (mmsi) => {
  if (mmsi === undefined || mmsi === null || Number.isNaN(Number(mmsi))) {
    return '#808080';
  }
  const idx = Math.abs(Number(mmsi)) % trajectoryColors.length;
  return trajectoryColors[idx];
};

const hasCoords = (lat, lon) => Number.isFinite(lat) && Number.isFinite(lon);

// Downsample points for cleaner visualization
const downsamplePoints = (points, maxPoints = 50) => {
  if (!points || points.length <= maxPoints) return points;
  const step = Math.ceil(points.length / maxPoints);
  const sampled = [];
  for (let i = 0; i < points.length; i += step) {
    sampled.push(points[i]);
  }
  // Always include the last point
  if (sampled[sampled.length - 1] !== points[points.length - 1]) {
    sampled.push(points[points.length - 1]);
  }
  return sampled;
};

// Component to fit map bounds to trajectories with better zoom control
function FitBounds({ trajectories, predictionTracks = [], anomalyPredictions = [], selectedTrajectory }) {
  const map = useMap();

  useEffect(() => {
    const bounds = [];
    const pushBounds = (lat, lon) => {
      if (hasCoords(lat, lon)) {
        bounds.push([lat, lon]);
      }
    };

    // If there's a selected trajectory, focus on it
    if (selectedTrajectory && selectedTrajectory.points && selectedTrajectory.points.length > 0) {
      selectedTrajectory.points.forEach(point => {
        pushBounds(point.lat, point.lon);
      });
      if (bounds.length > 0) {
        map.fitBounds(bounds, { 
          padding: [60, 60], 
          maxZoom: 14,
          animate: true,
          duration: 0.5
        });
      }
      return;
    }

    // Otherwise, fit to all data
    trajectories.forEach(traj => {
      if (traj.points && traj.points.length > 0) {
        // Only use first, middle, and last points for bounds calculation
        const pts = traj.points.filter(p => hasCoords(p.lat, p.lon));
        if (pts.length > 0) {
          pushBounds(pts[0].lat, pts[0].lon);
          if (pts.length > 2) {
            const mid = Math.floor(pts.length / 2);
            pushBounds(pts[mid].lat, pts[mid].lon);
          }
          if (pts.length > 1) {
            pushBounds(pts[pts.length - 1].lat, pts[pts.length - 1].lon);
          }
        }
      }
    });

    predictionTracks.forEach((track) => {
      const pts = (track.points || []);
      if (pts.length > 0) {
        pushBounds(pts[0].pred_latitude, pts[0].pred_longitude);
        pushBounds(pts[0].true_latitude, pts[0].true_longitude);
        if (pts.length > 1) {
          pushBounds(pts[pts.length - 1].pred_latitude, pts[pts.length - 1].pred_longitude);
          pushBounds(pts[pts.length - 1].true_latitude, pts[pts.length - 1].true_longitude);
        }
      }
    });

    anomalyPredictions.forEach((point) => {
      pushBounds(point.lat_true, point.lon_true);
      pushBounds(point.lat_pred, point.lon_pred);
    });

    if (bounds.length > 0) {
      map.fitBounds(bounds, { 
        padding: [40, 40], 
        maxZoom: 13,
        animate: true
      });
    }
  }, [trajectories, predictionTracks, anomalyPredictions, selectedTrajectory, map]);

  return null;
}

const TrajectoryMap = ({
  trajectories,
  selectedTrajectory,
  onTrajectoryClick,
  predictionTracks = [],
  anomalyPredictions = []
}) => {
  const mapRef = useRef(null);

  // Calculate center point from all trajectories
  const getCenter = () => {
    let latSum = 0;
    let lonSum = 0;
    let count = 0;

    trajectories.forEach(traj => {
      if (traj.points && traj.points.length > 0) {
        traj.points.forEach(point => {
          if (hasCoords(point.lat, point.lon)) {
            latSum += point.lat;
            lonSum += point.lon;
            count++;
          }
        });
      }
    });

    if (count === 0) {
      predictionTracks.forEach((track) => {
        (track.points || []).forEach((point) => {
          if (hasCoords(point.pred_latitude, point.pred_longitude)) {
            latSum += point.pred_latitude;
            lonSum += point.pred_longitude;
            count++;
          }
        });
      });
    }

    if (count === 0) {
      anomalyPredictions.forEach((point) => {
        if (hasCoords(point.lat_true, point.lon_true)) {
          latSum += point.lat_true;
          lonSum += point.lon_true;
          count++;
        } else if (hasCoords(point.lat_pred, point.lon_pred)) {
          latSum += point.lat_pred;
          lonSum += point.lon_pred;
          count++;
        }
      });
    }

    if (count > 0) {
      return [latSum / count, lonSum / count];
    }
    return [57.7, 11.8];
  };

  const center = getCenter();

  // Debug: Log trajectory data
  useEffect(() => {
    console.log('TrajectoryMap - trajectories:', trajectories.length);
    if (trajectories.length > 0) {
      console.log('First trajectory points:', trajectories[0].points?.length);
      console.log('First point with coords:', trajectories[0].points?.[0]);
    }
    if (predictionTracks.length > 0) {
      console.log('Prediction tracks available:', predictionTracks.length);
    }
  }, [trajectories, predictionTracks]);

  // Count valid trajectories with coordinates
  const validTrajectories = trajectories.filter(traj => {
    if (!traj.points || traj.points.length < 2) return false;
    const positions = traj.points.filter(p => hasCoords(p.lat, p.lon));
    return positions.length >= 2;
  });

  const hasPredictionLines = predictionTracks.some((track) => {
    const predictedPositions = (track.points || []).filter((point) => hasCoords(point.pred_latitude, point.pred_longitude));
    const actualPositions = (track.points || []).filter((point) => hasCoords(point.true_latitude, point.true_longitude));
    return predictedPositions.length >= 2 || actualPositions.length >= 2;
  });

  const hasAnomalyMarkers = anomalyPredictions.some(
    (point) => hasCoords(point.lat_true, point.lon_true) || hasCoords(point.lat_pred, point.lon_pred)
  );

  if (validTrajectories.length === 0 && !hasPredictionLines && !hasAnomalyMarkers) {
    return (
      <div style={{ padding: '20px', textAlign: 'center', color: '#666' }}>
        <p>No spatial data available to display.</p>
        <p>Load trajectories, prediction tracks, or anomaly predictions.</p>
      </div>
    );
  }

  // Compact legend component
  const MapLegend = () => (
    <div style={{
      position: 'absolute',
      bottom: '10px',
      right: '10px',
      zIndex: 1000,
      background: 'rgba(13, 13, 35, 0.92)',
      padding: '10px 14px',
      borderRadius: '6px',
      boxShadow: '0 2px 8px rgba(0,0,0,0.4)',
      fontFamily: 'system-ui',
      fontSize: '11px',
      color: '#e0e0e0',
      border: '1px solid rgba(0, 212, 255, 0.3)'
    }}>
      <div style={{ fontWeight: '600', marginBottom: '6px', color: '#00d4ff', fontSize: '12px' }}>
        Legend
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '18px', height: '3px', background: '#00d4ff', borderRadius: '2px' }}></div>
          <span>Selected</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '18px', height: '3px', background: '#2ecc71', borderRadius: '2px' }}></div>
          <span>Actual</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '18px', height: '3px', background: '#3498db', borderRadius: '2px', borderTop: '2px dashed #3498db' }}></div>
          <span>Predicted</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#e74c3c' }}></div>
          <span>Anomaly</span>
        </div>
      </div>
    </div>
  );

  return (
    <div className="trajectory-map-container" style={{ height: '100%', width: '100%', minHeight: '400px', position: 'relative' }}>
      <MapContainer
        center={center}
        zoom={10}
        style={{ height: '100%', width: '100%', minHeight: '400px' }}
        ref={mapRef}
      >
        {/* Clean nautical-style map - Esri Ocean Basemap */}
        <TileLayer
          attribution='Tiles &copy; Esri &mdash; Sources: GEBCO, NOAA, CHS, OSU, UNH, CSUMB, National Geographic, DeLorme, NAVTEQ, and Esri'
          url="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}"
          maxZoom={16}
        />
        {/* Add ocean labels */}
        <TileLayer
          attribution=''
          url="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Reference/MapServer/tile/{z}/{y}/{x}"
          maxZoom={16}
        />

        <FitBounds
          trajectories={validTrajectories}
          predictionTracks={predictionTracks}
          anomalyPredictions={anomalyPredictions}
          selectedTrajectory={selectedTrajectory}
        />

        {/* Render all trajectories - downsampled for cleaner view */}
        {validTrajectories.map((traj) => {
          if (!traj.points || traj.points.length < 2) return null;

          // Downsample and extract coordinates for cleaner lines
          const sampledPoints = downsamplePoints(traj.points, 30);
          const positions = sampledPoints
            .filter(p => hasCoords(p.lat, p.lon))
            .map(p => [p.lat, p.lon]);

          if (positions.length < 2) return null;

          let color = getTrajectoryColor(traj.mmsi);
          const isSelected = selectedTrajectory?.segment_id === traj.segment_id;
          const isAnomalous = traj.anomaly?.is_anomalous;
          const weight = isSelected ? 4 : isAnomalous ? 4 : 3;
          const opacity = isSelected ? 1.0 : isAnomalous ? 0.9 : 0.75;

          if (isAnomalous) {
            color = '#e74c3c';
          }
          if (isSelected) {
            color = '#00d4ff';
          }

          return (
            <Polyline
              key={traj.segment_id}
              positions={positions}
              color={color}
              weight={weight}
              opacity={opacity}
              eventHandlers={{
                click: () => {
                  if (onTrajectoryClick) {
                    onTrajectoryClick(traj);
                  }
                },
                mouseover: (e) => {
                  const layer = e.target;
                  layer.setStyle({
                    weight: 5,
                    opacity: 1.0
                  });
                },
                mouseout: (e) => {
                  const layer = e.target;
                  layer.setStyle({
                    weight: weight,
                    opacity: opacity
                  });
                }
              }}
            >
              <Popup>
                <div style={{ minWidth: '200px', fontFamily: 'system-ui' }}>
                  <div style={{ borderBottom: '2px solid #00d4ff', paddingBottom: '8px', marginBottom: '8px' }}>
                    <strong style={{ fontSize: '14px', color: '#00d4ff' }}>🚢 Vessel Trajectory</strong>
                  </div>
                  <div style={{ fontSize: '12px', lineHeight: '1.6' }}>
                    <div><strong>MMSI:</strong> {traj.mmsi || 'N/A'}</div>
                    <div><strong>Segment ID:</strong> {traj.segment_id}</div>
                    <div><strong>Data Points:</strong> {traj.points.length}</div>
                    {traj.points && traj.points.length > 0 && (
                      <>
                        <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid #ddd' }}>
                          <strong>First Point:</strong>
                          <div style={{ marginLeft: '10px', fontSize: '11px', color: '#666' }}>
                            Lat: {traj.points[0].lat?.toFixed(6) || 'N/A'}°<br />
                            Lon: {traj.points[0].lon?.toFixed(6) || 'N/A'}°
                            {traj.points[0].sog && <><br />Speed: {traj.points[0].sog.toFixed(2)} knots</>}
                            {traj.points[0].cog && <><br />Course: {traj.points[0].cog.toFixed(1)}°</>}
                          </div>
                        </div>
                        {traj.points.length > 1 && (
                          <div style={{ marginTop: '4px' }}>
                            <strong>Last Point:</strong>
                            <div style={{ marginLeft: '10px', fontSize: '11px', color: '#666' }}>
                              Lat: {traj.points[traj.points.length - 1].lat?.toFixed(6) || 'N/A'}°<br />
                              Lon: {traj.points[traj.points.length - 1].lon?.toFixed(6) || 'N/A'}°
                              {traj.points[traj.points.length - 1].sog && <><br />Speed: {traj.points[traj.points.length - 1].sog.toFixed(2)} knots</>}
                              {traj.points[traj.points.length - 1].cog && <><br />Course: {traj.points[traj.points.length - 1].cog.toFixed(1)}°</>}
                            </div>
                          </div>
                        )}
                      </>
                    )}
                    {traj.anomaly?.is_anomalous && (
                      <div style={{ marginTop: '8px', padding: '6px', background: '#ffe6e6', borderRadius: '4px', border: '1px solid #ff9999' }}>
                        <strong style={{ color: '#d32f2f' }}>⚠️ Anomaly Detected</strong>
                        {traj.anomaly.score && <div>Score: {traj.anomaly.score.toFixed(3)}</div>}
                        {traj.anomaly.threshold && <div>Threshold: {traj.anomaly.threshold.toFixed(2)}</div>}
                      </div>
                    )}
                  </div>
                </div>
              </Popup>
            </Polyline>
          );
        })}

        {/* Prediction tracks - downsampled */}
        {predictionTracks.map((track) => {
          const sampledTrackPoints = downsamplePoints(track.points || [], 30);
          const predictedPositions = sampledTrackPoints
            .filter((point) => hasCoords(point.pred_latitude, point.pred_longitude))
            .map((point) => [point.pred_latitude, point.pred_longitude]);
          const actualPositions = sampledTrackPoints
            .filter((point) => hasCoords(point.true_latitude, point.true_longitude))
            .map((point) => [point.true_latitude, point.true_longitude]);

          const trackLabel = track.mmsi || 'Unknown MMSI';
          const keyFragment = `${trackLabel}-${track.sample_count || predictedPositions.length}`;
          const showActual = actualPositions.length >= 2;
          const showPredicted = predictedPositions.length >= 2;

          return (
            <React.Fragment key={`prediction-track-${keyFragment}`}>
              {showActual && (
                <Polyline positions={actualPositions} color="#2ecc71" weight={3.5} opacity={0.8}>
                  <Popup>
                    <div style={{ minWidth: '200px', fontFamily: 'system-ui' }}>
                      <div style={{ borderBottom: '2px solid #2ecc71', paddingBottom: '8px', marginBottom: '8px' }}>
                        <strong style={{ fontSize: '14px', color: '#2ecc71' }}>📍 Actual Vessel Track</strong>
                      </div>
                      <div style={{ fontSize: '12px', lineHeight: '1.6' }}>
                        <div><strong>MMSI:</strong> {trackLabel}</div>
                        <div><strong>Track Points:</strong> {track.sample_count || actualPositions.length}</div>
                        {track.points && track.points.length > 0 && (
                          <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid #ddd' }}>
                            <strong>Current Status:</strong>
                            <div style={{ marginLeft: '10px', fontSize: '11px', color: '#666' }}>
                              {track.points[track.points.length - 1].true_sog && (
                                <>Speed: {track.points[track.points.length - 1].true_sog.toFixed(2)} knots<br /></>
                              )}
                              {track.points[track.points.length - 1].true_cog && (
                                <>Course: {track.points[track.points.length - 1].true_cog.toFixed(1)}°<br /></>
                              )}
                              {track.points[track.points.length - 1].true_latitude && (
                                <>Position: {track.points[track.points.length - 1].true_latitude.toFixed(6)}°N, {track.points[track.points.length - 1].true_longitude.toFixed(6)}°E</>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </Popup>
                </Polyline>
              )}
              {showPredicted && (
                <Polyline
                  positions={predictedPositions}
                  color="#3498db"
                  weight={3}
                  opacity={0.85}
                  dashArray="8 6"
                >
                  <Popup>
                    <div style={{ minWidth: '200px', fontFamily: 'system-ui' }}>
                      <div style={{ borderBottom: '2px solid #3498db', paddingBottom: '8px', marginBottom: '8px' }}>
                        <strong style={{ fontSize: '14px', color: '#3498db' }}>🔮 Predicted Track (LSTM)</strong>
                      </div>
                      <div style={{ fontSize: '12px', lineHeight: '1.6' }}>
                        <div><strong>MMSI:</strong> {trackLabel}</div>
                        <div><strong>Prediction Points:</strong> {track.sample_count || predictedPositions.length}</div>
                        {track.points && track.points.length > 0 && (
                          <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid #ddd' }}>
                            <strong>Predicted Status:</strong>
                            <div style={{ marginLeft: '10px', fontSize: '11px', color: '#666' }}>
                              {track.points[track.points.length - 1].pred_sog && (
                                <>Speed: {track.points[track.points.length - 1].pred_sog.toFixed(2)} knots<br /></>
                              )}
                              {track.points[track.points.length - 1].pred_cog && (
                                <>Course: {track.points[track.points.length - 1].pred_cog.toFixed(1)}°<br /></>
                              )}
                              {track.points[track.points.length - 1].pred_latitude && (
                                <>Position: {track.points[track.points.length - 1].pred_latitude.toFixed(6)}°N, {track.points[track.points.length - 1].pred_longitude.toFixed(6)}°E</>
                              )}
                            </div>
                            {track.points[track.points.length - 1].combined_error && (
                              <div style={{ marginTop: '6px', padding: '4px', background: '#fff3cd', borderRadius: '4px', fontSize: '11px' }}>
                                <strong>Prediction Error:</strong> {track.points[track.points.length - 1].combined_error.toFixed(3)} km
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </Popup>
                </Polyline>
              )}
            </React.Fragment>
          );
        })}

        {/* Anomaly markers - using CircleMarker for cleaner look */}
        {validTrajectories.map((traj) => {
          if (!traj.anomaly?.is_anomalous || !traj.points || traj.points.length === 0) {
            return null;
          }

          const coords = traj.points.filter((p) => hasCoords(p.lat, p.lon));
          if (coords.length === 0) {
            return null;
          }
          const lastPoint = coords[coords.length - 1];

          return (
            <CircleMarker
              key={`anomaly-${traj.segment_id}`}
              center={[lastPoint.lat, lastPoint.lon]}
              radius={10}
              fillColor="#e74c3c"
              color="#fff"
              weight={3}
              opacity={1}
              fillOpacity={0.9}
            >
              <Popup>
                <div style={{ minWidth: '180px', fontFamily: 'system-ui' }}>
                  <div style={{ borderBottom: '2px solid #e74c3c', paddingBottom: '6px', marginBottom: '6px' }}>
                    <strong style={{ fontSize: '13px', color: '#e74c3c' }}>⚠️ Anomaly Detected</strong>
                  </div>
                  <div style={{ fontSize: '12px', lineHeight: '1.5' }}>
                    <div><strong>MMSI:</strong> {traj.mmsi || 'N/A'}</div>
                    <div><strong>Score:</strong> {traj.anomaly.score?.toFixed(3) || 'N/A'}</div>
                    <div style={{ fontSize: '11px', color: '#666', marginTop: '4px' }}>
                      Position: {lastPoint.lat?.toFixed(4)}°, {lastPoint.lon?.toFixed(4)}°
                    </div>
                  </div>
                </div>
              </Popup>
            </CircleMarker>
          );
        })}

        {/* Prediction anomaly markers - only show flagged anomalies, limited count */}
        {anomalyPredictions
          .filter(point => point.anomaly)
          .slice(0, 3)
          .map((point, idx) => {
            const trueCoords = hasCoords(point.lat_true, point.lon_true) ? [point.lat_true, point.lon_true] : null;
            if (!trueCoords) {
              return null;
            }

            return (
              <CircleMarker
                key={`prediction-anomaly-${point.mmsi || 'na'}-${idx}`}
                center={trueCoords}
                radius={7}
                fillColor="#f39c12"
                color="#fff"
                weight={2}
                opacity={1}
                fillOpacity={0.85}
              >
                <Popup>
                  <div style={{ fontFamily: 'system-ui', fontSize: '12px' }}>
                    <strong style={{ color: '#f39c12' }}>Prediction Anomaly</strong>
                    <div style={{ marginTop: '4px', color: '#666' }}>
                      Error: {point.error ? `${point.error.toFixed(3)} km` : 'N/A'}
                    </div>
                  </div>
                </Popup>
              </CircleMarker>
            );
          })}

        {/* Add markers for start/end points of selected trajectory */}
        {selectedTrajectory && selectedTrajectory.points && selectedTrajectory.points.length > 0 && (
          <>
            {/* Start marker */}
            {hasCoords(selectedTrajectory.points[0].lat, selectedTrajectory.points[0].lon) && (
              <Marker
                position={[
                  selectedTrajectory.points[0].lat,
                  selectedTrajectory.points[0].lon
                ]}
                icon={L.icon({
                  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-green.png',
                  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
                  iconSize: [25, 41],
                  iconAnchor: [12, 41],
                  popupAnchor: [1, -34],
                  shadowSize: [41, 41]
                })}
              >
                <Popup>
                  <div style={{ fontFamily: 'system-ui', fontSize: '12px' }}>
                    <strong style={{ color: '#2ecc71' }}>🟢 Start Point</strong>
                    {selectedTrajectory.points[0].sog && (
                      <div style={{ marginTop: '4px', fontSize: '11px', color: '#666' }}>
                        Speed: {selectedTrajectory.points[0].sog.toFixed(2)} knots<br />
                        {selectedTrajectory.points[0].cog && <>Course: {selectedTrajectory.points[0].cog.toFixed(1)}°</>}
                      </div>
                    )}
                  </div>
                </Popup>
              </Marker>
            )}

            {/* End marker */}
            {hasCoords(
              selectedTrajectory.points[selectedTrajectory.points.length - 1].lat,
              selectedTrajectory.points[selectedTrajectory.points.length - 1].lon
            ) && (
              <Marker
                position={[
                  selectedTrajectory.points[selectedTrajectory.points.length - 1].lat,
                  selectedTrajectory.points[selectedTrajectory.points.length - 1].lon
                ]}
                icon={L.icon({
                  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png',
                  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
                  iconSize: [25, 41],
                  iconAnchor: [12, 41],
                  popupAnchor: [1, -34],
                  shadowSize: [41, 41]
                })}
              >
                <Popup>
                  <div style={{ fontFamily: 'system-ui', fontSize: '12px' }}>
                    <strong style={{ color: '#e74c3c' }}>🔴 End Point</strong>
                    {selectedTrajectory.points[selectedTrajectory.points.length - 1].sog && (
                      <div style={{ marginTop: '4px', fontSize: '11px', color: '#666' }}>
                        Speed: {selectedTrajectory.points[selectedTrajectory.points.length - 1].sog.toFixed(2)} knots<br />
                        {selectedTrajectory.points[selectedTrajectory.points.length - 1].cog && <>Course: {selectedTrajectory.points[selectedTrajectory.points.length - 1].cog.toFixed(1)}°</>}
                      </div>
                    )}
                  </div>
                </Popup>
              </Marker>
            )}
          </>
        )}
      </MapContainer>
      <MapLegend />
    </div>
  );
};

export default TrajectoryMap;

