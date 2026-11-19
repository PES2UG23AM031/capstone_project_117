import React, { useState } from "react";
import axios from "axios";
import "./App.css";
import { translations } from "./translations";

const API_BASE_URL = "http://127.0.0.1:5000";

function App() {
  const [file, setFile]       = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [lang, setLang]       = useState("en"); // "en", "kn", "hi"

  const t = translations[lang] || translations.en;

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
  };

  const handleUpload = async () => {
    if (!file) return alert(t.uploadPrompt);
    if (loading) return;
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("lang", lang);
      const res = await axios.post(`${API_BASE_URL}/predict`, formData, {
        timeout: 90000,
      });
      setResult(res.data);
    } catch (err) {
      console.error("Diagnosis error:", err);
      const msg = err.response?.data?.error || err.response?.data?.message || err.message;
      alert(`Backend notification: ${msg || "Make sure app.py is running on http://127.0.0.1:5000"}`);
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

  const getLocalizedDisease = (disease) => {
    if (!disease) return "";
    const cleanKey = disease.replace(/[^a-zA-Z]/g, "");
    return t.diseases?.[disease] || t.diseases?.[cleanKey] || disease;
  };

  const getLocalizedSeverity = (label) => {
    return t.severities?.[label] || label;
  };

  const getLocalizedField = (disease, field, fallback) => {
    if (!disease) return fallback;
    const cleanKey = disease.replace(/[^a-zA-Z]/g, "");
    const recs = t.treatmentRecommendations?.[disease] || t.treatmentRecommendations?.[cleanKey];
    if (recs && recs[field]) {
      return recs[field];
    }
    return fallback;
  };

  const getLocalizedCanopyAction = (actionStatus) => {
    if (!actionStatus) return "";
    if (actionStatus.includes("HEALTHY")) return t.canopyActions?.healthy || actionStatus;
    if (actionStatus.includes("LOCALIZED")) return t.canopyActions?.localized || actionStatus;
    if (actionStatus.includes("CRITICAL")) return t.canopyActions?.critical || actionStatus;
    return actionStatus;
  };

  const getLocalizedUrgency = (note) => {
    if (!note) return "";
    const lower = note.toLowerCase();
    if (lower.includes("healthy") || lower.includes("no treatment")) return t.urgency?.none || note;
    if (lower.includes("critical")) return t.urgency?.critical || note;
    if (lower.includes("high") || lower.includes("urgently") || lower.includes("today")) return t.urgency?.high || note;
    if (lower.includes("moderate") || lower.includes("2-3 days") || lower.includes("2–3 days")) return t.urgency?.moderate || note;
    if (lower.includes("low") || lower.includes("monitor")) return t.urgency?.low || note;
    return note;
  };

  const getLocalizedModelArch = (arch) => {
    if (!arch) return "";
    if (arch.includes("Multi-Leaf") || arch.includes("Two-Stage")) {
      return t.modelArchMulti || arch;
    }
    return t.modelArchSingle || arch;
  };

  return (
    <div className="app">
      <h1>{t.title}</h1>
      <p className="subtitle">{t.subtitle}</p>

      {/* Multilingual Selector */}
      <div className="language-bar">
        <button
          className={`lang-btn ${lang === "en" ? "active" : ""}`}
          onClick={() => setLang("en")}
        >
          English
        </button>
        <button
          className={`lang-btn ${lang === "kn" ? "active" : ""}`}
          onClick={() => setLang("kn")}
        >
          ಕನ್ನಡ (Kannada)
        </button>
        <button
          className={`lang-btn ${lang === "hi" ? "active" : ""}`}
          onClick={() => setLang("hi")}
        >
          हिन्दी (Hindi)
        </button>
        <button
          className={`lang-btn ${lang === "te" ? "active" : ""}`}
          onClick={() => setLang("te")}
        >
          తెలుగు (Telugu)
        </button>
      </div>

      <div className="card">
        <input type="file" accept="image/*" onChange={handleFileChange} />
        {preview && <img src={preview} alt="preview" className="image" />}
        <button onClick={handleUpload} className="btn" disabled={loading} style={{opacity: loading ? 0.7 : 1, cursor: loading ? "not-allowed" : "pointer"}}>
          {loading ? t.analyzingBtn : t.predictBtn}
        </button>
      </div>

      {loading && <div className="loader"></div>}

      {result && (
        <>
          {result.is_citrus === false ? (
            <div className="result-card" style={{ borderLeft: "5px solid #d32f2f", background: "#fffaf9" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", marginBottom: 12 }}>
                <h2 style={{ color: "#d32f2f", margin: 0 }}>{t.nonCitrus?.title || "🚫 Non-Citrus Leaf / Rejected"}</h2>
                <span style={{ background: "#ffebee", color: "#c62828", padding: "4px 12px", borderRadius: 12, fontSize: "0.82rem", fontWeight: 700, border: "1px solid #ffcdd2" }}>
                  {t.nonCitrus?.badge || "Verification Status: REJECTED"}
                </span>
              </div>
              <div style={{ background: "#ffffff", padding: "14px 16px", borderRadius: 8, marginBottom: 16, border: "1px solid #ffcdd2" }}>
                <p style={{ margin: "0 0 8px 0", fontSize: "0.98rem", color: "#b71c1c", fontWeight: 700 }}>
                  {t.nonCitrus?.verdict || result.verdict || "Uploaded leaf does not match authentic Citrus (Rutaceae) foliar characteristics."}
                </p>
                <p style={{ margin: "0 0 8px 0", fontSize: "0.88rem", color: "#444" }}>
                  <strong>{t.warning || "Clinical Note"}:</strong> {t.nonCitrus?.warning || result.warning}
                </p>
                <p style={{ margin: 0, fontSize: "0.88rem", color: "#2e7d32" }}>
                  <strong>{t.culturalAdvice || "Action"}:</strong> {t.nonCitrus?.advice || result.cultural_advice}
                </p>
              </div>
              <div className="visual-panel" style={{ maxWidth: 380, margin: "0 auto", textAlign: "center" }}>
                <p className="visual-title">{t.uploadedImage}</p>
                {preview && <img src={preview} alt="Uploaded non-citrus leaf" className="analysis-image" />}
              </div>
            </div>
          ) : (
            <>
              <div className="result-card">
                <div style={{display:"flex", justifyContent:"space-between", alignItems:"center", flexWrap:"wrap", marginBottom:12}}>
                  <h2>{getLocalizedDisease(result.disease)}</h2>
                  <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
                    <span style={{background:"#e8f5e9", color:"#2e7d32", padding:"4px 10px", borderRadius:12, fontSize:"0.78rem", fontWeight:600, border:"1px solid #c8e6c9"}}>
                      🌿 {t.verifiedCitrus || "Verified Citrus Leaf"}
                    </span>
                    {result.model_architecture && (
                      <span style={{background:"#f1f8e9", color:"#33691e", padding:"4px 10px", borderRadius:12, fontSize:"0.78rem", fontWeight:600, border:"1px solid #dcedc8"}}>
                        ⚡ {getLocalizedModelArch(result.model_architecture)}
                      </span>
                    )}
                  </div>
                </div>
            <div className="bar">
              <div className="confidence" style={{width:`${result.confidence}%`}}>
                {t.confidence} {result.confidence.toFixed(1)}%
              </div>
            </div>

            <p className="severity">
              {t.severity}:{" "}
              <span style={{color: severityColor(result.severity_label)}}>
                {getLocalizedSeverity(result.severity_label)} ({result.severity}%)
              </span>
            </p>

            <p className="detail">
              {t.infectedArea}: <strong>{result.infected_pct}%</strong>
              &nbsp;|&nbsp;
              {t.spots}: <strong>{result.spot_count}</strong>
              &nbsp;|&nbsp;
              {t.zones}: <strong>{result.zones_infected}/9</strong>
            </p>

            <div className="severity-visuals">
              <div className="visual-panel">
                <p className="visual-title">{t.uploadedImage}</p>
                {preview && <img src={preview} alt="Uploaded leaf" className="analysis-image" />}
              </div>

              {result.severity_image_url && (
                <div className="visual-panel">
                  <p className="visual-title">{t.stage2Severity}</p>
                  <img
                    src={`${API_BASE_URL}${result.severity_image_url}`}
                    alt="Severity estimation overlay"
                    className="analysis-image"
                  />
                  <p className="visual-caption">
                    {t.leafTissueNote}
                  </p>
                </div>
              )}
            </div>

            {result.breakdown && (
              <div className="breakdown">
                <p className="breakdown-title">{t.breakdownTitle}</p>
                <p className="formula-text">
                  {t.formulaText}
                </p>
                {[
                  [t.areaLabel,   result.breakdown.area_score,   "#E07B54"],
                  [t.spreadLabel, result.breakdown.spread_score, "#54A0E0"],
                  [t.colorLabel,  result.breakdown.color_score,  "#E0C454"],
                  ...(result.breakdown.zone_score !== undefined ? [[t.zoneLabel, result.breakdown.zone_score, "#54C054"]] : []),
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

          {/* ── Multi-Leaf Canopy Analysis Card ── */}
          {result.is_multi_leaf && (
            <div className="result-card" style={{ borderLeft: "5px solid #2e7d32" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                <h3 style={{ margin: 0, color: "#2e7d32" }}>{t.canopyTitle || "🌿 Plant Canopy Diagnosis (Multi-Leaf)"}</h3>
                <span style={{ background: "#e8f5e9", color: "#2e7d32", padding: "4px 10px", borderRadius: 12, fontSize: "0.82rem", fontWeight: 700 }}>
                  {result.total_leaves_detected} {t.leavesAssessed || "Leaves Assessed"}
                </span>
              </div>

              <div style={{ background: "#f5f5f5", padding: "12px 16px", borderRadius: 8, marginBottom: 16 }}>
                <p style={{ margin: "0 0 6px 0", fontSize: "0.95rem" }}>
                  <strong>{t.canopyInfectionRate || "Canopy Infection Rate"}:</strong>{" "}
                  <span style={{ color: result.canopy_infection_rate > 50 ? "#d32f2f" : "#f57c00", fontWeight: 700 }}>
                    {result.canopy_infection_rate}%
                  </span>
                  &nbsp;({result.healthy_leaves_count} {getLocalizedDisease("Healthy")}, {result.diseased_leaves_count} {t.severities?.Severe || "Diseased"})
                </p>
                <p style={{ margin: 0, fontSize: "0.9rem", color: "#424242" }}>
                  <strong>{t.actionRequired || "Action Required"}:</strong> {getLocalizedCanopyAction(result.canopy_action_status)}
                </p>
              </div>

              <h4 style={{ margin: "0 0 10px 0", fontSize: "0.92rem", color: "#555" }}>{t.individualBreakdown || "Individual Leaf Breakdown:"}</h4>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 10 }}>
                {result.individual_leaves && result.individual_leaves.map((leaf) => (
                  <div key={leaf.leaf_id} style={{
                    border: "1px solid #e0e0e0",
                    borderRadius: 8,
                    padding: "10px 12px",
                    background: leaf.disease.toLowerCase() === "healthy" ? "#f1f8e9" : "#fffde7"
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontWeight: 600, fontSize: "0.88rem" }}>
                      <span>{t.leafNum || "Leaf"} #{leaf.leaf_id}</span>
                      <span style={{ color: leaf.disease.toLowerCase() === "healthy" ? "#388e3c" : "#d32f2f" }}>
                        {getLocalizedDisease(leaf.disease)} ({leaf.confidence}%)
                      </span>
                    </div>
                    <div style={{ fontSize: "0.8rem", color: "#666", marginTop: 4 }}>
                      {t.severity || "Severity"}: <strong>{leaf.severity}%</strong> ({getLocalizedSeverity(leaf.grade_label) || "Grade 0"})
                    </div>
                    <div style={{ fontSize: "0.78rem", color: "#333", marginTop: 6, borderTop: "1px dashed #ccc", paddingTop: 4 }}>
                      <strong>{t.dose || "Rx"}:</strong> {getLocalizedField(leaf.disease, "treatment", leaf.treatment)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── Treatment Recommendation ── */}
          {result.treatment && result.disease !== "Healthy" && (
            <div className="result-card treatment-card">
              <h3>{t.treatmentTitle}</h3>

              {result.low_confidence_warning && (
                <div className="warning-box">
                  {lang === "kn" ? "ಮಾದರಿ ನಿಖರತೆ ಕಡಿಮೆ ಇದೆ — ಚಿಕಿತ್ಸೆ ನೀಡುವ ಮುನ್ನ ರೋಗ ಲಕ್ಷಣಗಳನ್ನು ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ." : (lang === "hi" ? "मॉडल की सटीकता सीमा से कम है — उपचार से पहले लक्षणों का पुनः सत्यापन करें।" : (lang === "te" ? "మోడల్ ఖచ్చితత్వం పరిమితి కంటే తక్కువగా ఉంది — చికిత్స అందించే ముందు తెగులు లక్షణాలను తిరిగి నిర్ధారించుకోండి." : result.low_confidence_warning))}
                </div>
              )}

              <div className="urgency"
                   style={{color: urgencyColor(result.urgency_note)}}>
                {getLocalizedUrgency(result.urgency_note)}
              </div>

              <table className="treatment-table">
                <tbody>
                  <tr><td>{t.treatmentTitle}</td><td><strong>{getLocalizedField(result.disease, "treatment", result.treatment)}</strong></td></tr>
                  {result.dose    && <tr><td>{t.dose}</td><td>{getLocalizedField(result.disease, "dose", result.dose)}</td></tr>}
                  {result.method  && <tr><td>{t.method}</td><td>{getLocalizedField(result.disease, "method", result.method)}</td></tr>}
                  {result.timing  && <tr><td>{t.timing}</td><td>{getLocalizedField(result.disease, "timing", result.timing)}</td></tr>}
                  {result.interval&& <tr><td>{t.interval}</td><td>{getLocalizedField(result.disease, "interval", result.interval)}</td></tr>}
                </tbody>
              </table>

              {result.warning && (
                <div className="warning-box warning-yellow">
                  {t.warning}: {getLocalizedField(result.disease, "warning", result.warning)}
                </div>
              )}

              {result.cultural_advice && (
                <div className="advice-box">
                  <strong>{t.culturalAdvice}:</strong> {getLocalizedField(result.disease, "cultural_advice", result.cultural_advice)}
                </div>
              )}

              {result.monitoring && (
                <div className="advice-box">
                  <strong>{t.monitoring}:</strong> {getLocalizedField(result.disease, "monitoring", result.monitoring)}
                </div>
              )}

              <p className="reason-text">{getLocalizedField(result.disease, "reason", result.reason)}</p>
            </div>
          )}

          {/* ── Healthy message ── */}
          {result.disease === "Healthy" && (
            <div className="result-card">
              <p style={{textAlign:"center", color:"#4CAF50", fontWeight:600}}>
                {t.healthyMsg}
              </p>
              {result.cultural_advice && (
                <div className="advice-box" style={{marginTop:12}}>
                  <strong>{t.maintenance}:</strong> {getLocalizedField("Healthy", "cultural_advice", result.cultural_advice)}
                </div>
              )}
            </div>
          )}
            </>
          )}
        </>
      )}
    </div>
  );
}

export default App;
