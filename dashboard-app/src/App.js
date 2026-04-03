import React, { useState, useEffect } from 'react';
import AnomalyDashboard from './components/AnomalyDashboard';
import './components/AnomalyDashboard.css';
import './App.css';

function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'dark');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  return (
    <div className="App">
      <header className="App-header">
        <div className="header-title">
          {/* <h1>🚢 AIS Vessel Intelligence</h1> */}
          <h1> 🚢 LOMAD </h1>
          {/* <p>LSTM trajectory prediction · Anomaly detection · AI-powered analysis</p> */}
          <p> Local Anomaly Detection in Maritime Trajectories Using LSTM Prediction and Visual Analytics </p>
        </div>
        <div className="theme-toggle-wrapper">
          <span className={`theme-label${theme === 'dark' ? ' active' : ''}`}>Dark</span>
          <label className="theme-slider-label">
            <input
              type="checkbox"
              checked={theme === 'light'}
              onChange={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}
            />
            <span className="theme-slider-track">
              <span className="theme-slider-thumb" />
            </span>
          </label>
          <span className={`theme-label${theme === 'light' ? ' active' : ''}`}>Light</span>
        </div>
      </header>
      <main>
        <AnomalyDashboard />
      </main>
    </div>
  );
}

export default App;
