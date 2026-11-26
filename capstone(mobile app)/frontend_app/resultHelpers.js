// Pure helpers used by App.js (kept separate so they can be unit-tested).

// Removes a leading emoji so it can be drawn as its own <Text> (avoids the Android clipping bug).
export const stripLeadingEmoji = (s) =>
  String(s || '').replace(/^(?:[\uD83C-\uD83E][\uDC00-\uDFFF]|[\u2600-\u27BF]|\uFE0F|\s)+/, '');

export const isHealthy = (disease) => String(disease || '').trim().toLowerCase() === 'healthy';

// Backend urgency sentence -> key of translations[lang].urgency
export const urgencyKey = (note) => {
  const l = String(note || '').toLowerCase();
  if (!l) return null;
  if (l.includes('critical') || l.includes('emergency')) return 'critical';
  if (l.includes('high')) return 'high';
  if (l.includes('moderate')) return 'moderate';
  if (l.includes('low')) return 'low';
  if (l.includes('no treatment') || l.includes('healthy')) return 'none';
  return null;
};

// Backend canopy action sentence -> key of translations[lang].canopyActions
export const canopyActionKey = (text) => {
  const s = String(text || '').toUpperCase();
  if (s.startsWith('HEALTHY')) return 'healthy';
  if (s.startsWith('LOCALIZED')) return 'localized';
  if (s.startsWith('CRITICAL')) return 'critical';
  return null;
};

export const canonicalDisease = (disease, dict) => {
  const k = String(disease || '').replace(/\s+/g, '');
  if (dict && dict[k]) return k;
  if (k === 'CitrusCanker') return 'Canker';
  if (k === 'Leafminer') return 'LeafMiner';
  return k;
};

export const localizedDisease = (translations, lang, disease) => {
  const d = translations[lang] && translations[lang].diseases;
  if (!d) return disease;
  return d[disease] || d[canonicalDisease(disease, d)] || disease;
};

export const localizedSeverity = (translations, lang, label) => {
  const s = translations[lang] && translations[lang].severities;
  return (s && s[label]) || label;
};

export const localizedUrgency = (translations, lang, note) => {
  if (lang === 'en') return note;
  const key = urgencyKey(note);
  const u = translations[lang] && translations[lang].urgency;
  return (key && u && u[key]) || note;
};

export const localizedCanopyAction = (translations, lang, text) => {
  if (lang === 'en') return text;
  const key = canopyActionKey(text);
  const a = translations[lang] && translations[lang].canopyActions;
  return (key && a && a[key]) || text;
};

const TREATMENT_FIELDS = ['treatment', 'dose', 'method', 'timing', 'interval', 'cultural_advice', 'warning', 'monitoring', 'reason'];

// English: text from the RAG backend.  Other languages: curated dictionary text for that disease,
// falling back to the backend text when the dictionary has no entry.
export const localizedTreatment = (translations, lang, r) => {
  const out = {};
  const all = translations[lang] && translations[lang].treatmentRecommendations;
  const dict = lang !== 'en' && all ? all[canonicalDisease(r.disease, all)] : null;
  TREATMENT_FIELDS.forEach((f) => { out[f] = (dict && dict[f]) || r[f] || ''; });
  return out;
};

export const leafSizeText = (t, category) => {
  const c = String(category || '').toLowerCase();
  if (c.startsWith('small')) return t('leafSizeSmall');
  if (c.startsWith('medium')) return t('leafSizeMedium');
  if (c.startsWith('large')) return t('leafSizeLarge');
  return category || '-';
};

export const severityColor = (label) => {
  const l = String(label || '').toLowerCase();
  if (l.includes('healthy')) return '#4CAF50';
  if (l.includes('mild') || l.includes('trace')) return '#8BC34A';
  if (l.includes('moderate')) return '#FFC107';
  if (l.includes('severe')) return '#FF5722';
  if (l.includes('critical')) return '#D32F2F';
  return '#888';
};

export const urgencyColor = (note) => {
  const k = urgencyKey(note);
  if (k === 'none' || k === 'low') return '#4CAF50';
  if (k === 'moderate') return '#FFC107';
  if (k === 'high') return '#FF5722';
  if (k === 'critical') return '#D32F2F';
  return '#888';
};