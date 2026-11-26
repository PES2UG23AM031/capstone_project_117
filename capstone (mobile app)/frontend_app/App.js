import React, { useState } from 'react';
import {
  StyleSheet,
  Text as RNText,
  View,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  ScrollView,
  Alert,
  Platform,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { translations } from './translations';
import {
  stripLeadingEmoji,
  isHealthy as isHealthyDisease,
  canopyActionKey,
  localizedDisease,
  localizedSeverity,
  localizedUrgency,
  localizedCanopyAction,
  localizedTreatment,
  leafSizeText,
  severityColor,
  urgencyColor,
} from './resultHelpers';

// ============================================================
// API
// ============================================================
const API_URL = 'https://citrus-backend-43363162624.asia-south1.run.app';
const REQUEST_TIMEOUT_MS = 180000; // first request after idle + multi-leaf photos can be slow

// Set by <App/> on every render so the Text component can adapt to the chosen language.
let ACTIVE_LANG = 'en';

const LANGUAGES = [
  { code: 'en', label: 'English', flag: '🇬🇧' },
  { code: 'kn', label: 'ಕನ್ನಡ', flag: '🇮🇳' },
  { code: 'hi', label: 'हिन्दी', flag: '🇮🇳' },
  { code: 'te', label: 'తెలుగు', flag: '🇮🇳' },
];

// ============================================================
// TEXT WRAPPER
// Some phones use a custom system font. Android then computes a wrong line box
// for emoji / Kannada / Hindi / Telugu glyphs and the label gets pushed out of view.
// An explicit lineHeight and no extra font padding fixes it.
// ============================================================
const Text = ({ style, ...props }) => {
  const flat = StyleSheet.flatten(style) || {};
  const size = flat.fontSize;
  const extra = size && !flat.lineHeight ? { lineHeight: Math.round(size * 1.45) } : null;
  // Some phones measure Devanagari / Kannada / Telugu text (especially bold) a little narrower than
  // they draw it. The last letters then wrap onto a second line that is clipped away, e.g. "कैमरा"
  // shows as "कैम". A little free space on the right stops that.
  const slack = size && flat.paddingRight === undefined ? { paddingRight: Math.ceil(size * 0.4) } : null;
  // Hindi: some phones draw synthetic bold wider than they measure it, which clips the last letters.
  // Plain weight is measured and drawn consistently, so bold is switched off for Hindi.
  const w = String(flat.fontWeight || '');
  const noBold = ACTIVE_LANG === 'hi' && w && w !== 'normal' && w !== '400' ? { fontWeight: 'normal' } : null;
  return <RNText {...props} style={[styles.noPad, style, extra, slack, noBold]} />;
};

// Emoji + label as TWO separate Text nodes (never mixed in one string)
const IconLabel = ({ icon, label, textStyle, iconSize = 18 }) => (
  <View style={styles.iconRow}>
    {icon ? <Text style={[styles.emoji, { fontSize: iconSize }]}>{icon}</Text> : null}
    <Text style={textStyle}>{label}</Text>
  </View>
);

const Btn = ({ icon, label, onPress, disabled, style, textStyle = styles.btnText }) => (
  <TouchableOpacity style={style} onPress={onPress} disabled={disabled} activeOpacity={0.8}>
    <IconLabel icon={icon} label={label} textStyle={textStyle} />
  </TouchableOpacity>
);

const canopyColor = (text) => {
  const k = canopyActionKey(text);
  if (k === 'healthy') return '#4CAF50';
  if (k === 'localized') return '#E0A800';
  return '#D32F2F';
};

// ============================================================
// MAIN APP
// ============================================================
export default function App() {
  const [lang, setLang] = useState(null); // 'en' | 'kn' | 'hi' | 'te'
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  // set when the backend says the photo is not (clearly) a citrus leaf:
  //   { status: 'reject' | 'retake' | 'fruit', probability }
  const [gateIssue, setGateIssue] = useState(null);

  ACTIVE_LANG = lang || 'en';

  const t = (key) =>
    (lang && translations[lang] && translations[lang][key]) || translations.en[key] || key;

  const clearOutput = () => {
    setResult(null);
    setError(null);
    setGateIssue(null);
  };

  const selectLanguage = (code) => {
    setLang(code);
    clearOutput();
  };

  // ---------------- PICK IMAGE ----------------
  const pickImage = async () => {
    clearOutput();
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert(t('permissionNeeded'), t('galleryPermission'));
      return;
    }
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaType
        ? ImagePicker.MediaType.Images
        : ImagePicker.MediaTypeOptions.Images,
      allowsEditing: false, // do not crop: a branch photo may contain several leaves
      quality: 0.8,
    });
    if (!res.canceled) setImage(res.assets[0].uri);
  };

  // ---------------- CAMERA ----------------
  const takePhoto = async () => {
    clearOutput();
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert(t('permissionNeeded'), t('cameraPermission'));
      return;
    }
    const res = await ImagePicker.launchCameraAsync({
      allowsEditing: false,
      quality: 0.8,
    });
    if (!res.canceled) setImage(res.assets[0].uri);
  };

  // ---------------- ANALYZE ----------------
  // force = true -> "Analyze anyway" on a borderline ("retake") photo
  const analyze = async (uri = image, force = false) => {
    if (!uri) return;
    setLoading(true);
    clearOutput();

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
      const form = new FormData();
      form.append('file', {
        uri: Platform.OS === 'android' ? uri : uri.replace('file://', ''),
        name: 'leaf.jpg',
        type: 'image/jpeg',
      });
      if (force) form.append('force', '1'); // backend only honours this for "retake", never "reject"

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: form,
        signal: controller.signal,
      });

      if (!response.ok) {
        let message = `Server error: ${response.status}`;
        try {
          const errData = await response.json();
          message = errData.message || errData.error || message;
        } catch {
          // keep default message
        }
        throw new Error(message);
      }

      const data = await response.json();

      // Not a (clear) citrus leaf, or a fruit: show a friendly card instead of a result.
      if (data.status === 'reject' || data.status === 'retake' || data.status === 'fruit') {
        setGateIssue({ status: data.status, probability: data.citrus_probability });
        return;
      }
      if (data.status === 'error' || !data.disease) {
        throw new Error(data.message || data.error || 'Unexpected response from server');
      }

      setResult(data);
    } catch (err) {
      console.log('Prediction error:', err);
      setError(err.name === 'AbortError' ? t('requestTimeout') : err.message);
    } finally {
      clearTimeout(timeout);
      setLoading(false);
    }
  };

  const resetAnalysis = () => {
    setImage(null);
    clearOutput();
  };

  // ============================================================
  // LANGUAGE SELECTION SCREEN
  // ============================================================
  if (!lang) {
    return (
      <View style={styles.languageScreen}>
        <Text style={styles.logo}>🌿</Text>
        <Text style={styles.welcomeTitle}>Welcome</Text>
        <Text style={styles.languageSubtitle}>Select your preferred language</Text>
        {LANGUAGES.map((l) => (
          <Btn
            key={l.code}
            icon={l.flag}
            label={l.label}
            style={styles.languageButton}
            textStyle={styles.languageButtonText}
            onPress={() => selectLanguage(l.code)}
          />
        ))}
      </View>
    );
  }

  // ============================================================
  // DERIVED VALUES FOR THE RESULT SCREEN
  // ============================================================
  const healthy = result ? isHealthyDisease(result.disease) : false;
  const isMulti = !!(result && result.is_multi_leaf);
  const tx = result ? localizedTreatment(translations, lang, result) : null;

  const severityImg = result
    ? result.severity_image_b64
      ? `data:image/jpeg;base64,${result.severity_image_b64}`
      : result.severity_image_url
      ? `${API_URL}${result.severity_image_url}`
      : null
    : null;
  const canopyImg = result
    ? result.annotated_canopy_b64
      ? `data:image/jpeg;base64,${result.annotated_canopy_b64}`
      : result.annotated_canopy_url
      ? `${API_URL}${result.annotated_canopy_url}`
      : null
    : null;

  const gateIcon = gateIssue
    ? gateIssue.status === 'fruit' ? '🍊' : gateIssue.status === 'reject' ? '🚫' : '📸'
    : '';
  const gateTitle = gateIssue
    ? gateIssue.status === 'fruit'
      ? t('fruitTitle')
      : gateIssue.status === 'reject'
      ? t('notCitrusTitle')
      : t('retakeTitle')
    : '';
  const gateMsg = gateIssue
    ? gateIssue.status === 'fruit'
      ? t('fruitMsg')
      : gateIssue.status === 'reject'
      ? t('notCitrusMsg')
      : t('retakeMsg')
    : '';

  // ============================================================
  // MAIN UI
  // ============================================================
  return (
    <ScrollView contentContainerStyle={styles.container}>
      {/* HEADER */}
      <IconLabel icon="🌿" label={t('appTitle')} textStyle={styles.heading} iconSize={26} />

      {/* CHANGE LANGUAGE */}
      <Btn
        icon="🌐"
        label={t('changeLanguage')}
        style={styles.changeLanguageButton}
        textStyle={styles.changeLanguageText}
        onPress={() => {
          setLang(null);
          setImage(null);
          clearOutput();
        }}
      />

      {/* IMAGE CARD */}
      <View style={styles.card}>
        <View style={styles.btnRow}>
          <Btn icon="📷" label={t('camera')} style={styles.btn} onPress={takePhoto} disabled={loading} />
          <Btn icon="🖼️" label={t('gallery')} style={styles.btn} onPress={pickImage} disabled={loading} />
        </View>

        <IconLabel icon="💡" label={t('photoTip')} textStyle={styles.tipText} iconSize={14} />

        {image && (
          <>
            <Text style={styles.imageTitle}>{t('selectedImage')}</Text>
            <Image source={{ uri: image }} style={styles.previewImg} />
            <Btn
              icon="🔍"
              label={loading ? t('analyzing') : t('predict')}
              style={[styles.btn, styles.predictBtn, loading && styles.btnDisabled]}
              onPress={() => analyze(image)}
              disabled={loading}
            />
          </>
        )}
      </View>

      {/* LOADING */}
      {loading && (
        <View style={styles.loaderBox}>
          <ActivityIndicator size="large" color="#4CAF50" />
          <Text style={styles.loaderText}>{t('analyzingLeaf')}</Text>
          <Text style={styles.loaderSub}>{t('wait')}</Text>
        </View>
      )}

      {/* ERROR */}
      {error && (
        <View style={styles.errorBox}>
          <IconLabel icon="⚠️" label={t('error')} textStyle={styles.errorTitle} iconSize={14} />
          <Text style={styles.errorText}>{error}</Text>
          <Btn
            label={t('tryAgain')}
            style={[styles.btn, { marginTop: 10, flex: 0 }]}
            onPress={() => image && analyze(image)}
          />
        </View>
      )}

      {/* NOT A CITRUS LEAF / PLEASE RETAKE / FRUIT */}
      {gateIssue && !loading && (
        <View
          style={[
            styles.gateBox,
            gateIssue.status === 'reject' ? styles.gateReject : styles.gateRetake,
            gateIssue.status === 'fruit' && styles.gateFruit,
          ]}
        >
          <IconLabel icon={gateIcon} label={gateTitle} textStyle={styles.gateTitle} iconSize={18} />
          <Text style={styles.gateText}>{gateMsg}</Text>

          {gateIssue.status !== 'fruit' &&
            [1, 2, 3, 4].map((n) => (
              <View key={n} style={styles.tipRow}>
                <Text style={styles.tipBullet}>•</Text>
                <Text style={styles.gateTip}>{t(`retakeTip${n}`)}</Text>
              </View>
            ))}

          <View style={styles.gateBtnRow}>
            <Btn label={t('tryAnotherPhoto')} style={styles.btn} onPress={resetAnalysis} />
            {gateIssue.status === 'retake' && (
              <Btn
                label={t('analyzeAnyway')}
                style={[styles.btn, styles.btnOutline]}
                textStyle={styles.btnOutlineText}
                onPress={() => image && analyze(image, true)}
              />
            )}
          </View>
        </View>
      )}

      {/* RESULT */}
      {result && !loading && (
        <>
          {result.gate_status === 'unverified' && (
            <View style={[styles.warningBox, styles.warningYellow]}>
              <IconLabel
                icon="⚠️"
                label={t('unverifiedNote')}
                textStyle={[styles.warningYellowText, { flexShrink: 1 }]}
                iconSize={14}
              />
            </View>
          )}

          {/* CANOPY (several leaves in one photo) */}
          {isMulti && (
            <View style={styles.resultCard}>
              <IconLabel
                icon="🌿"
                label={stripLeadingEmoji(t('canopyTitle'))}
                textStyle={styles.resultTitle}
                iconSize={20}
              />
              {canopyImg && <Image source={{ uri: canopyImg }} style={styles.analysisImg} />}

              <View style={styles.statsRow}>
                <StatBox value={result.total_leaves_detected} label={t('leavesAssessed')} />
                <StatBox value={result.healthy_leaves_count} label={t('healthyLabel')} color="#4CAF50" />
                <StatBox value={result.diseased_leaves_count} label={t('diseasedLabel')} color="#D32F2F" />
                <StatBox
                  value={`${Number(result.canopy_infection_rate || 0).toFixed(1)}%`}
                  label={t('canopyInfectionRate')}
                />
              </View>

              <Text style={styles.label}>{t('actionRequired')}</Text>
              <Text style={[styles.canopyAction, { color: canopyColor(result.canopy_action_status) }]}>
                {localizedCanopyAction(translations, lang, result.canopy_action_status)}
              </Text>

              <Text style={[styles.sectionTitle, { marginTop: 14 }]}>{t('individualBreakdown')}</Text>
              {(result.individual_leaves || []).map((leaf) => (
                <View key={leaf.leaf_id} style={styles.leafRow}>
                  <View style={styles.leafHead}>
                    <Text style={styles.leafName}>
                      {t('leafNum')} #{leaf.leaf_id}: {localizedDisease(translations, lang, leaf.disease)}
                    </Text>
                    <Text style={[styles.leafSev, { color: severityColor(leaf.severity_label) }]}>
                      {localizedSeverity(translations, lang, leaf.severity_label)}
                    </Text>
                  </View>
                  <Text style={styles.leafMeta}>
                    {t('confidence')}: {Number(leaf.confidence || 0).toFixed(1)}%{'  |  '}
                    {t('severity')}: {Number(leaf.severity || 0).toFixed(1)}%
                  </Text>
                  {!isHealthyDisease(leaf.disease) && (
                    <Text style={styles.leafTreat}>
                      {localizedTreatment(translations, lang, leaf).treatment}
                    </Text>
                  )}
                </View>
              ))}
            </View>
          )}

          {isMulti && <Text style={styles.primaryHeading}>{t('primaryLeaf')}</Text>}

          {/* PREDICTION */}
          <View style={styles.resultCard}>
            <Text style={styles.resultTitle}>{t('predictionResult')}</Text>

            <Text style={styles.label}>{t('disease')}</Text>
            <Text style={styles.diseaseName}>{localizedDisease(translations, lang, result.disease)}</Text>

            <Text style={styles.label}>{t('confidence')}</Text>
            <View style={styles.bar}>
              <View
                style={[styles.confFill, { width: `${Math.min(Number(result.confidence || 0), 100)}%` }]}
              >
                <Text style={styles.confText}>{Number(result.confidence || 0).toFixed(1)}%</Text>
              </View>
            </View>

            <Text style={styles.severityLine}>
              {t('severity')}:{' '}
              <Text style={{ color: severityColor(result.severity_label), fontWeight: '700' }}>
                {localizedSeverity(translations, lang, result.severity_label)} (
                {Number(result.severity || 0).toFixed(2)}%)
              </Text>
            </Text>

            <Text style={styles.detailLine}>
              {t('infectedArea')}: {Number(result.infected_pct || 0).toFixed(2)}%{'  |  '}
              {t('spots')}: {result.spot_count || 0}
              {'  |  '}
              {t('zones')}: {result.zones_infected || 0}/9
            </Text>

            <Text style={styles.modelText}>
              {t('model')}: {isMulti ? t('modelArchMulti') : t('modelArchSingle')}
            </Text>
          </View>

          {/* IMAGES (single leaf) */}
          {!isMulti && (
            <View style={styles.resultCard}>
              <Text style={styles.sectionTitle}>{t('uploadedImage')}</Text>
              <Image source={{ uri: image }} style={styles.analysisImg} />

              {severityImg && (
                <>
                  <Text style={styles.sectionTitle}>{t('stage2Severity')}</Text>
                  <Image source={{ uri: severityImg }} style={styles.analysisImg} />
                  <Text style={styles.visualCaption}>{t('leafTissueNote')}</Text>
                </>
              )}
            </View>
          )}

          {/* SEVERITY BREAKDOWN */}
          {result.breakdown && !healthy && (
            <View style={styles.breakdown}>
              <Text style={styles.breakdownTitle}>{t('breakdownTitle')}</Text>
              <Text style={styles.formulaText}>{t('formulaText')}</Text>
              <ScoreRow label={t('areaLabel')} value={result.breakdown.area_score} color="#E07B54" />
              <ScoreRow label={t('colorLabel')} value={result.breakdown.color_score} color="#E0C454" />
              <ScoreRow label={t('spreadLabel')} value={result.breakdown.spread_score} color="#54A0E0" />
              <ScoreRow label={t('zoneLabel')} value={result.breakdown.zone_score} color="#54C054" />
            </View>
          )}

          {/* LEAF INFORMATION */}
          <View style={styles.resultCard}>
            <Text style={styles.sectionTitle}>{t('leafInformation')}</Text>
            <TRow label={t('leafSize')} value={leafSizeText(t, result.leaf_size_category)} />
            <TRow
              label={result.is_young_leaf ? t('youngLeaf') : t('matureLeaf')}
              value={result.is_young_leaf ? t('yes') : t('no')}
            />
          </View>

          {/* TREATMENT */}
          {tx.treatment && !healthy && (
            <View style={styles.resultCard}>
              <Text style={styles.treatmentHeading}>{t('treatmentTitle')}</Text>

              {!!result.low_confidence_warning && (
                <View style={styles.warningBox}>
                  <Text style={styles.warningText}>
                    {lang === 'en' ? result.low_confidence_warning : t('lowConfidence')}
                  </Text>
                </View>
              )}

              {!!result.urgency_note && (
                <Text style={[styles.urgency, { color: urgencyColor(result.urgency_note) }]}>
                  {localizedUrgency(translations, lang, result.urgency_note)}
                </Text>
              )}

              <View style={styles.table}>
                <TRow label={t('treatment')} value={tx.treatment} bold />
                {!!tx.dose && <TRow label={t('dose')} value={tx.dose} />}
                {!!tx.method && <TRow label={t('method')} value={tx.method} />}
                {!!tx.timing && <TRow label={t('timing')} value={tx.timing} />}
                {!!tx.interval && <TRow label={t('interval')} value={tx.interval} />}
              </View>

              {!!tx.warning && (
                <View style={[styles.warningBox, styles.warningYellow]}>
                  <Text style={styles.warningYellowText}>
                    ⚠️ {t('warning')}: {tx.warning}
                  </Text>
                </View>
              )}

              {!!tx.cultural_advice && (
                <View style={styles.adviceBox}>
                  <Text style={styles.adviceText}>
                    <Text style={styles.bold}>{t('culturalAdvice')}: </Text>
                    {tx.cultural_advice}
                  </Text>
                </View>
              )}

              {!!tx.monitoring && (
                <View style={styles.adviceBox}>
                  <Text style={styles.adviceText}>
                    <Text style={styles.bold}>{t('monitoring')}: </Text>
                    {tx.monitoring}
                  </Text>
                </View>
              )}

              {!!tx.reason && (
                <Text style={styles.reasonText}>
                  <Text style={styles.bold}>{t('recommendationReason')}: </Text>
                  {tx.reason}
                </Text>
              )}
            </View>
          )}

          {/* HEALTHY */}
          {healthy && (
            <View style={styles.resultCard}>
              <IconLabel
                icon="✅"
                label={t('healthyMsg')}
                textStyle={[styles.diseaseName, { color: '#4CAF50', flexShrink: 1, marginBottom: 0 }]}
                iconSize={20}
              />
              {!!tx.cultural_advice && (
                <View style={[styles.adviceBox, { marginTop: 12 }]}>
                  <Text style={styles.adviceText}>
                    <Text style={styles.bold}>{t('maintenance')}: </Text>
                    {tx.cultural_advice}
                  </Text>
                </View>
              )}
            </View>
          )}

          {/* NEW ANALYSIS */}
          <Btn
            icon="🔄"
            label={t('newAnalysis')}
            style={styles.newAnalysisButton}
            textStyle={styles.newAnalysisText}
            onPress={resetAnalysis}
          />
        </>
      )}

      {/* INSTRUCTION */}
      {!image && !loading && (
        <View style={styles.instructionBox}>
          <IconLabel
            icon="📌"
            label={t('instruction')}
            textStyle={[styles.instructionText, { flexShrink: 1 }]}
            iconSize={14}
          />
          <Text style={[styles.instructionText, { marginTop: 8 }]}>{t('branchTip')}</Text>
        </View>
      )}
    </ScrollView>
  );
}

// ============================================================
// SMALL COMPONENTS
// ============================================================
const ScoreRow = ({ label, value, color }) => {
  const n = Number(value || 0);
  return (
    <View style={styles.breakdownRow}>
      <Text style={styles.breakdownLabel}>{label}</Text>
      <View style={styles.miniBar}>
        <View style={[styles.miniFill, { width: `${Math.min(n, 100)}%`, backgroundColor: color }]} />
      </View>
      <Text style={styles.breakdownVal}>{n.toFixed(1)}%</Text>
    </View>
  );
};

const TRow = ({ label, value, bold }) => (
  <View style={styles.tableRow}>
    <Text style={styles.tableLabel}>{label}</Text>
    <Text style={[styles.tableValue, bold && { fontWeight: '700' }]}>{value}</Text>
  </View>
);

const StatBox = ({ value, label, color }) => (
  <View style={styles.statBox}>
    <Text style={[styles.statValue, color ? { color } : null]}>{String(value ?? 0)}</Text>
    <Text style={styles.statLabel}>{label}</Text>
  </View>
);

// ============================================================
// STYLES
// ============================================================
const styles = StyleSheet.create({
  // helpers
  noPad: { includeFontPadding: false },
  iconRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center' },
  emoji: { marginRight: 8, textAlign: 'center' },
  bold: { fontWeight: '700' },

  // language screen
  languageScreen: {
    flex: 1,
    backgroundColor: '#f0f4f0',
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 30,
  },
  logo: { fontSize: 65, marginBottom: 10 },
  welcomeTitle: { fontSize: 30, fontWeight: '700', color: '#2e7d32', marginBottom: 10, textAlign: 'center' },
  languageSubtitle: { fontSize: 17, color: '#666', marginBottom: 30, textAlign: 'center' },
  languageButton: {
    width: '90%',
    backgroundColor: '#4CAF50',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    marginVertical: 7,
    minHeight: 56,
    elevation: 3,
  },
  languageButtonText: { color: '#fff', fontWeight: '700', fontSize: 18 },

  // main
  container: { flexGrow: 1, backgroundColor: '#f0f4f0', padding: 16, paddingTop: 56 },
  heading: { fontSize: 26, fontWeight: '700', color: '#2e7d32', textAlign: 'center', flexShrink: 1 },
  changeLanguageButton: {
    alignSelf: 'flex-end',
    backgroundColor: '#e8f5e9',
    paddingHorizontal: 13,
    paddingVertical: 8,
    borderRadius: 20,
    marginTop: 12,
    marginBottom: 12,
  },
  changeLanguageText: { color: '#2e7d32', fontWeight: '600', fontSize: 13 },

  // cards
  card: { backgroundColor: '#fff', borderRadius: 16, padding: 20, marginBottom: 14, elevation: 3 },
  resultCard: { backgroundColor: '#fff', borderRadius: 16, padding: 20, marginBottom: 14, elevation: 3 },

  // buttons
  btnRow: { flexDirection: 'row', gap: 12, marginBottom: 14 },
  btn: {
    flex: 1,
    backgroundColor: '#4CAF50',
    paddingVertical: 14,
    paddingHorizontal: 10,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 50,
  },
  predictBtn: { marginTop: 12, flex: 0 },
  btnDisabled: { backgroundColor: '#aaa' },
  btnText: { color: '#fff', fontWeight: '600', fontSize: 15, textAlign: 'center' },
  imageTitle: { fontSize: 14, fontWeight: '600', color: '#555', marginBottom: 5 },
  previewImg: { width: '100%', height: 280, borderRadius: 10, resizeMode: 'contain', marginVertical: 12 },

  // [GATE] camera tip
  tipText: { fontSize: 12, color: '#5c6b5c', flexShrink: 1, textAlign: 'center' },

  // loading
  loaderBox: { alignItems: 'center', padding: 20 },
  loaderText: { marginTop: 10, color: '#2e7d32', fontWeight: '600', fontSize: 15 },
  loaderSub: { marginTop: 4, color: '#888', fontSize: 12 },

  // error
  errorBox: {
    backgroundColor: '#ffebee',
    borderRadius: 10,
    padding: 14,
    marginBottom: 14,
    borderLeftWidth: 4,
    borderLeftColor: '#f44336',
  },
  errorTitle: { fontWeight: '700', fontSize: 14, color: '#c62828' },
  errorText: { color: '#b71c1c', fontSize: 13, marginTop: 4 },

  // [GATE] not-citrus / retake card
  gateBox: { borderRadius: 12, padding: 16, marginBottom: 14, borderLeftWidth: 4, elevation: 2 },
  gateReject: { backgroundColor: '#ffebee', borderLeftColor: '#f44336' },
  gateRetake: { backgroundColor: '#fff8e1', borderLeftColor: '#FFC107' },
  gateTitle: { fontWeight: '700', fontSize: 16, color: '#333', flexShrink: 1 },
  gateText: { fontSize: 13, color: '#444', marginTop: 8, textAlign: 'center' },
  tipRow: { flexDirection: 'row', marginTop: 6, paddingRight: 8 },
  tipBullet: { fontSize: 13, color: '#444', marginRight: 8 },
  gateTip: { flex: 1, fontSize: 13, color: '#444' },
  gateBtnRow: { flexDirection: 'row', gap: 10, marginTop: 14 },
  btnOutline: { backgroundColor: '#fff', borderWidth: 1.5, borderColor: '#4CAF50' },
  btnOutlineText: { color: '#2e7d32', fontWeight: '600', fontSize: 15, textAlign: 'center' },

  // result
  resultTitle: { fontSize: 20, fontWeight: '700', color: '#2e7d32', textAlign: 'center', marginBottom: 15 },
  label: { fontSize: 13, color: '#777', fontWeight: '600', marginTop: 8, marginBottom: 4 },
  diseaseName: { fontSize: 22, fontWeight: '700', color: '#1b5e20', textAlign: 'center', marginBottom: 14 },
  bar: { backgroundColor: '#e8f5e9', borderRadius: 30, height: 36, overflow: 'hidden', marginBottom: 12 },
  confFill: {
    height: '100%',
    backgroundColor: '#4CAF50',
    borderRadius: 30,
    justifyContent: 'center',
    alignItems: 'center',
    minWidth: 60,
  },
  confText: { color: '#fff', fontWeight: '600', fontSize: 13 },
  severityLine: { fontSize: 15, fontWeight: '600', textAlign: 'center', marginBottom: 6 },
  detailLine: { fontSize: 12, color: '#555', textAlign: 'center', marginBottom: 14 },
  modelText: { fontSize: 11, color: '#777', textAlign: 'center', marginTop: 5 },

  // images
  analysisImg: {
    width: '100%',
    height: 240,
    borderRadius: 10,
    resizeMode: 'contain',
    backgroundColor: '#f7faf7',
    marginBottom: 14,
  },
  sectionTitle: { fontSize: 15, fontWeight: '700', color: '#1b5e20', marginBottom: 10 },
  visualCaption: { fontSize: 10, color: '#5c6b5c', marginBottom: 10 },

  // breakdown
  breakdown: { backgroundColor: '#fff', borderRadius: 16, padding: 16, marginBottom: 14, elevation: 2 },
  breakdownTitle: { fontWeight: '700', fontSize: 15, marginBottom: 8, color: '#333' },
  formulaText: { fontSize: 11, color: '#666', marginBottom: 12 },
  breakdownRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 8 },
  breakdownLabel: { width: 110, fontSize: 11, color: '#555' },
  miniBar: { flex: 1, height: 10, backgroundColor: '#e0e0e0', borderRadius: 5, overflow: 'hidden' },
  miniFill: { height: '100%', borderRadius: 5 },
  breakdownVal: { width: 40, fontSize: 11, fontWeight: '700', textAlign: 'right' },

  // leaf / treatment
  table: { marginBottom: 12 },
  tableRow: { flexDirection: 'row', paddingVertical: 8, borderBottomWidth: 1, borderBottomColor: '#f0f0f0' },
  tableLabel: { width: 90, fontSize: 13, color: '#666' },
  tableValue: { flex: 1, fontSize: 13, color: '#222' },
  treatmentHeading: { fontSize: 17, fontWeight: '600', color: '#1b5e20', marginBottom: 12 },
  urgency: { fontWeight: '600', fontSize: 13, textAlign: 'center', marginBottom: 12 },

  // warnings
  warningBox: { backgroundColor: '#fff3e0', borderLeftWidth: 3, borderLeftColor: '#FF9800', padding: 10, marginBottom: 10 },
  warningText: { fontSize: 12, color: '#e65100' },
  warningYellow: { backgroundColor: '#fff8e1', borderLeftColor: '#FFC107' },
  warningYellowText: { fontSize: 12, color: '#856404' },

  // advice
  adviceBox: { backgroundColor: '#f1f8e9', borderLeftWidth: 3, borderLeftColor: '#8BC34A', padding: 10, marginBottom: 8 },
  adviceText: { fontSize: 12, color: '#33691e' },
  reasonText: { fontSize: 11, color: '#888', marginTop: 8 },

  // new analysis
  newAnalysisButton: {
    backgroundColor: '#2e7d32',
    paddingVertical: 15,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
    minHeight: 52,
  },
  newAnalysisText: { color: '#fff', fontSize: 16, fontWeight: '700' },

  // instruction
  instructionBox: { backgroundColor: '#e8f5e9', borderRadius: 12, padding: 16, marginTop: 10 },
  instructionText: { color: '#2e7d32', fontSize: 13, textAlign: 'center' },

  // ---- merged: canopy / multi-leaf ----
  primaryHeading: { fontSize: 15, fontWeight: '700', color: '#2e7d32', marginBottom: 8, marginTop: 4, textAlign: 'center' },
  statsRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 4, marginBottom: 6 },
  statBox: { flexGrow: 1, flexBasis: '45%', backgroundColor: '#f1f8e9', borderRadius: 10, padding: 10, alignItems: 'center' },
  statValue: { fontSize: 20, fontWeight: '700', color: '#1b5e20' },
  statLabel: { fontSize: 11, color: '#5c6b5c', textAlign: 'center', marginTop: 2 },
  canopyAction: { fontSize: 13, fontWeight: '700', textAlign: 'center', marginBottom: 4 },
  leafRow: { borderTopWidth: 1, borderTopColor: '#eee', paddingVertical: 10 },
  leafHead: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: 8 },
  leafName: { fontSize: 14, fontWeight: '700', color: '#1b5e20', flexShrink: 1 },
  leafSev: { fontSize: 12, fontWeight: '700' },
  leafMeta: { fontSize: 12, color: '#555', marginTop: 2 },
  leafTreat: { fontSize: 12, color: '#333', marginTop: 4 },
  gateFruit: { backgroundColor: '#fff3e0', borderLeftColor: '#FF9800' },
});