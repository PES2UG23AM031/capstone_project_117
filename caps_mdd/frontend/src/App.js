import React, { useState } from "react";
import axios from "axios";
import "./App.css";

const API_BASE_URL = "http://127.0.0.1:5000";

function App() {
  const [file, setFile]       = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
  };

  const handleUpload = async () => {
    if (!file) return alert("Upload image first!");
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await axios.post(`${API_BASE_URL}/predict`, formData);
      setResult(res.data);
    } catch (err) {
      alert("Error connecting to backend. Make sure app.py is running.");
    }
    setLoading(false);
  };

  const severityColor = (label) => ({
    Healthy:  "#4CAF50", Mild: "#8BC34A",
    Moderate: "#FFC107", Severe: "#FF5722", Critical: "#D32F2F",
  }[label] || "#888");

  const urgencyColor = (note) => {
    if (!note) return "var(--color-text-secondary)";
    if (note.includes("Low"))      return "#4CAF50";
    if (note.includes("Moderate")) return "#FFC107";
    if (note.includes("High"))     return "#FF5722";
    if (note.includes("CRITICAL")) return "#D32F2F";
    return "var(--color-text-secondary)";
  };

  return (
    <div className="app">
      <h1>🌿 Leaf Disease Detector</h1>

      <div className="card">
        <input type="file" accept="image/*" onChange={handleFileChange} />
        {preview && <img src={preview} alt="preview" className="image" />}
        <button onClick={handleUpload} className="btn">
          {loading ? "Analyzing..." : "Predict"}
        </button>
      </div>

      {loading && <div className="loader"></div>}

      {result && (
        <>
          <div className="result-card">
            <h2>{result.disease}</h2>
            <div className="bar">
              <div className="confidence" style={{width:`${result.confidence}%`}}>
                Confidence {result.confidence.toFixed(1)}%
              </div>
            </div>

            <p className="severity">
              Severity:{" "}
              <span style={{color: severityColor(result.severity_label)}}>
                {result.severity_label} ({result.severity}%)
              </span>
            </p>

            <p className="detail">
              Infected area: <strong>{result.infected_pct}%</strong>
              &nbsp;|&nbsp;
              Spots: <strong>{result.spot_count}</strong>
              &nbsp;|&nbsp;
              Zones: <strong>{result.zones_infected}/9</strong>
            </p>

            <div className="severity-visuals">
              <div className="visual-panel">
                <p className="visual-title">Uploaded image</p>
                {preview && <img src={preview} alt="Uploaded leaf" className="analysis-image" />}
              </div>

              {result.severity_image_url && (
                <div className="visual-panel">
                  <p className="visual-title">Stage 2 severity estimation</p>
                  <img
                    src={`${API_BASE_URL}${result.severity_image_url}`}
                    alt="Severity estimation overlay"
                    className="analysis-image"
                  />
                  <p className="visual-caption">
                    Green shows detected leaf tissue. Red shows infected regions. Grid cells indicate spread across zones.
                  </p>
                </div>
              )}
            </div>

            {result.breakdown && (
              <div className="breakdown">
                <p className="breakdown-title">Severity score breakdown</p>
                <p className="formula-text">
                  Final score = 0.40 × Area + 0.25 × Color + 0.20 × Spread + 0.15 × Zone
                </p>
                {[
                  ["Area (×0.40)",   result.breakdown.area_score,   "#E07B54"],
                  ["Color (×0.25)",  result.breakdown.color_score,  "#E0C454"],
                  ["Spread (×0.20)", result.breakdown.spread_score, "#54A0E0"],
                  ["Zone (×0.15)",   result.breakdown.zone_score,   "#54C054"],
                ].map(([label, val, color]) => (
                  <div className="breakdown-row" key={label}>
                    <span>{label}</span>
                    <div className="mini-bar">
                      <div className="mini-fill" style={{width:`${val}%`, background:color}}/>
                    </div>
                    <span>{val}%</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* ── Treatment Recommendation ── */}
          {result.treatment && result.disease !== "Healthy" && (
            <div className="result-card treatment-card">
              <h3>Treatment Recommendation</h3>

              {result.low_confidence_warning && (
                <div className="warning-box">
                  {result.low_confidence_warning}
                </div>
              )}

              <div className="urgency"
                   style={{color: urgencyColor(result.urgency_note)}}>
                {result.urgency_note}
              </div>

              <table className="treatment-table">
                <tbody>
                  <tr><td>Treatment</td><td><strong>{result.treatment}</strong></td></tr>
                  {result.dose    && <tr><td>Dose</td><td>{result.dose}</td></tr>}
                  {result.method  && <tr><td>Method</td><td>{result.method}</td></tr>}
                  {result.timing  && <tr><td>Timing</td><td>{result.timing}</td></tr>}
                  {result.interval&& <tr><td>Interval</td><td>{result.interval}</td></tr>}
                </tbody>
              </table>

              {result.warning && (
                <div className="warning-box warning-yellow">
                  Warning: {result.warning}
                </div>
              )}

              {result.cultural_advice && (
                <div className="advice-box">
                  <strong>Cultural advice:</strong> {result.cultural_advice}
                </div>
              )}

              {result.monitoring && (
                <div className="advice-box">
                  <strong>Monitoring:</strong> {result.monitoring}
                </div>
              )}

              <p className="reason-text">{result.reason}</p>
            </div>
          )}

          {/* ── Healthy message ── */}
          {result.disease === "Healthy" && (
            <div className="result-card">
              <p style={{textAlign:"center", color:"#4CAF50", fontWeight:600}}>
                Leaf appears healthy — no treatment needed.
              </p>
              {result.cultural_advice && (
                <div className="advice-box" style={{marginTop:12}}>
                  <strong>Maintenance:</strong> {result.cultural_advice}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default App;
