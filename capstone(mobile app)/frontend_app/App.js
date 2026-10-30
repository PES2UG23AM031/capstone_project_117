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
import { I18n } from 'i18n-js';

// ============================================================
// API
// ============================================================
const API_URL = 'https://citrus-backend-43363162624.asia-south1.run.app';

// ============================================================
// I18N  (emojis are NOT inside these strings any more;
//        they are rendered separately by <Btn> / <IconLabel>)
// ============================================================
const translations = {
  en: {
    welcome: 'Welcome',
    selectLanguage: 'Select your preferred language',
    title: 'Leaf Disease Detector',
    camera: 'Camera',
    gallery: 'Gallery',
    predict: 'Predict',
    analyzing: 'Analyzing...',
    analyzingLeaf: 'Analyzing leaf...',
    wait: 'This may take 15-30 seconds',
    selectedImage: 'Selected Image',
    predictionResult: 'Prediction Result',
    confidence: 'Confidence',
    disease: 'Disease',
    severity: 'Severity',
    infectedArea: 'Infected area',
    spots: 'Spots',
    zones: 'Zones',
    uploadedImage: 'Uploaded image',
    severityEstimation: 'Stage 2 severity estimation',
    severityCaption: 'Green: leaf tissue. Red: infected regions. Grid: zone spread.',
    severityBreakdown: 'Severity score breakdown',
    formula: 'Final score = 0.45×Area + 0.20×Color + 0.20×Spread + 0.15×Zone',
    area: 'Area',
    color: 'Color',
    spread: 'Spread',
    zone: 'Zone',
    treatmentRecommendation: 'Treatment Recommendation',
    treatment: 'Treatment',
    dose: 'Dose',
    method: 'Method',
    timing: 'Timing',
    interval: 'Interval',
    warning: 'Warning',
    culturalAdvice: 'Cultural advice',
    monitoring: 'Monitoring',
    tryAgain: 'Try Again',
    error: 'Error',
    permissionNeeded: 'Permission needed',
    cameraPermission: 'Camera permission needed',
    galleryPermission: 'Gallery permission needed',
    requestTimeout: 'Request timed out. Server is busy. Please try again.',
    healthyMessage: 'Leaf appears healthy — no treatment needed.',
    maintenance: 'Maintenance',
    instruction: 'Take a photo or upload a leaf image to detect diseases and get treatment recommendations.',
    changeLanguage: 'Change Language',
    newAnalysis: 'Analyze Another Leaf',
    recommendationReason: 'Recommendation reason',
    urgency: 'Urgency',
    leafInformation: 'Leaf Information',
    leafSize: 'Leaf size',
    youngLeaf: 'Young leaf',
    matureLeaf: 'Mature leaf',
    yes: 'Yes',
    no: 'No',
    model: 'Model',
    // --- citrus gate ---
    photoTip: 'Tip: place ONE leaf on a plain surface and fill the frame.',
    notCitrusTitle: "This doesn't look like a citrus leaf",
    notCitrusMsg: 'Please take a clear photo of a single citrus leaf.',
    retakeTitle: 'Please retake the photo',
    retakeMsg: "I'm not sure this is a citrus leaf. Please retake the photo using these tips:",
    retakeTip1: 'Photograph ONE leaf only.',
    retakeTip2: 'Fill the frame with the leaf.',
    retakeTip3: 'Use good daylight and hold the camera steady.',
    retakeTip4: 'Hold the leaf in your hand or place it on a plain surface.',
    analyzeAnyway: 'Analyze anyway',
    tryAnotherPhoto: 'Try another photo',
    unverifiedNote: 'This photo was not confirmed as a citrus leaf, so the result may be unreliable.',
  },

  kn: {
    welcome: 'ಸ್ವಾಗತ',
    selectLanguage: 'ನಿಮ್ಮ ಇಷ್ಟದ ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ',
    title: 'ಎಲೆ ರೋಗ ಪತ್ತೆ ವ್ಯವಸ್ಥೆ',
    camera: 'ಕ್ಯಾಮೆರಾ',
    gallery: 'ಗ್ಯಾಲರಿ',
    predict: 'ಪತ್ತೆಹಚ್ಚಿ',
    analyzing: 'ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ...',
    analyzingLeaf: 'ಎಲೆಯನ್ನು ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ...',
    wait: 'ಇದಕ್ಕೆ 15-30 ಸೆಕೆಂಡುಗಳು ಬೇಕಾಗಬಹುದು',
    selectedImage: 'ಆಯ್ಕೆ ಮಾಡಿದ ಚಿತ್ರ',
    predictionResult: 'ಭವಿಷ್ಯವಾಣಿ ಫಲಿತಾಂಶ',
    confidence: 'ವಿಶ್ವಾಸಾರ್ಹತೆ',
    disease: 'ರೋಗ',
    severity: 'ತೀವ್ರತೆ',
    infectedArea: 'ಸೋಂಕಿತ ಪ್ರದೇಶ',
    spots: 'ಕಲೆಗಳು',
    zones: 'ವಲಯಗಳು',
    uploadedImage: 'ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ಚಿತ್ರ',
    severityEstimation: 'ಹಂತ 2 ತೀವ್ರತೆ ಅಂದಾಜು',
    severityCaption: 'ಹಸಿರು: ಎಲೆಯ ಭಾಗ. ಕೆಂಪು: ಸೋಂಕಿತ ಪ್ರದೇಶಗಳು. ಗ್ರಿಡ್: ವಲಯ ಹರಡುವಿಕೆ.',
    severityBreakdown: 'ತೀವ್ರತೆ ಅಂಕಗಳ ವಿವರ',
    formula: 'ಅಂತಿಮ ಅಂಕ = 0.45×ಪ್ರದೇಶ + 0.20×ಬಣ್ಣ + 0.20×ಹರಡುವಿಕೆ + 0.15×ವಲಯ',
    area: 'ಪ್ರದೇಶ',
    color: 'ಬಣ್ಣ',
    spread: 'ಹರಡುವಿಕೆ',
    zone: 'ವಲಯ',
    treatmentRecommendation: 'ಚಿಕಿತ್ಸೆಯ ಸಲಹೆ',
    treatment: 'ಚಿಕಿತ್ಸೆ',
    dose: 'ಪ್ರಮಾಣ',
    method: 'ವಿಧಾನ',
    timing: 'ಸಮಯ',
    interval: 'ಅಂತರ',
    warning: 'ಎಚ್ಚರಿಕೆ',
    culturalAdvice: 'ಬೆಳೆ ನಿರ್ವಹಣೆ ಸಲಹೆ',
    monitoring: 'ಮೇಲ್ವಿಚಾರಣೆ',
    tryAgain: 'ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ',
    error: 'ದೋಷ',
    permissionNeeded: 'ಅನುಮತಿ ಅಗತ್ಯವಿದೆ',
    cameraPermission: 'ಕ್ಯಾಮೆರಾ ಅನುಮತಿ ಅಗತ್ಯವಿದೆ',
    galleryPermission: 'ಗ್ಯಾಲರಿ ಅನುಮತಿ ಅಗತ್ಯವಿದೆ',
    requestTimeout: 'ವಿನಂತಿಯ ಸಮಯ ಮುಗಿದಿದೆ. ಸರ್ವರ್ ಕಾರ್ಯನಿರತವಾಗಿದೆ. ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.',
    healthyMessage: 'ಎಲೆ ಆರೋಗ್ಯಕರವಾಗಿ ಕಾಣುತ್ತದೆ. ಚಿಕಿತ್ಸೆಯ ಅಗತ್ಯವಿಲ್ಲ.',
    maintenance: 'ನಿರ್ವಹಣೆ',
    instruction: 'ರೋಗವನ್ನು ಪತ್ತೆಹಚ್ಚಲು ಮತ್ತು ಚಿಕಿತ್ಸೆಯ ಸಲಹೆಗಳನ್ನು ಪಡೆಯಲು ಎಲೆಯ ಫೋಟೋ ತೆಗೆದುಕೊಳ್ಳಿ ಅಥವಾ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.',
    changeLanguage: 'ಭಾಷೆ ಬದಲಾಯಿಸಿ',
    newAnalysis: 'ಮತ್ತೊಂದು ಎಲೆಯನ್ನು ವಿಶ್ಲೇಷಿಸಿ',
    recommendationReason: 'ಸಲಹೆಯ ಕಾರಣ',
    urgency: 'ತುರ್ತು ಮಟ್ಟ',
    leafInformation: 'ಎಲೆಯ ಮಾಹಿತಿ',
    leafSize: 'ಎಲೆಯ ಗಾತ್ರ',
    youngLeaf: 'ಎಳೆಯ ಎಲೆ',
    matureLeaf: 'ಬೆಳೆದ ಎಲೆ',
    yes: 'ಹೌದು',
    no: 'ಇಲ್ಲ',
    model: 'ಮಾದರಿ',
    // --- citrus gate (please have a Kannada speaker check the wording) ---
    photoTip: 'ಸಲಹೆ: ಸರಳ ಹಿನ್ನೆಲೆಯ ಮೇಲೆ ಒಂದೇ ಎಲೆಯನ್ನು ಇರಿಸಿ ಮತ್ತು ಚೌಕಟ್ಟು ತುಂಬುವಂತೆ ಫೋಟೋ ತೆಗೆಯಿರಿ.',
    notCitrusTitle: 'ಇದು ಸಿಟ್ರಸ್ ಎಲೆಯಂತೆ ಕಾಣುತ್ತಿಲ್ಲ',
    notCitrusMsg: 'ದಯವಿಟ್ಟು ಒಂದು ಸಿಟ್ರಸ್ ಎಲೆಯ ಸ್ಪಷ್ಟ ಫೋಟೋ ತೆಗೆಯಿರಿ.',
    retakeTitle: 'ದಯವಿಟ್ಟು ಮತ್ತೊಮ್ಮೆ ಫೋಟೋ ತೆಗೆಯಿರಿ',
    retakeMsg: 'ಇದು ಸಿಟ್ರಸ್ ಎಲೆ ಎಂದು ಖಚಿತವಿಲ್ಲ. ದಯವಿಟ್ಟು ಈ ಸಲಹೆಗಳನ್ನು ಅನುಸರಿಸಿ ಮತ್ತೊಮ್ಮೆ ಫೋಟೋ ತೆಗೆಯಿರಿ:',
    retakeTip1: 'ಒಂದೇ ಎಲೆಯ ಫೋಟೋ ತೆಗೆಯಿರಿ.',
    retakeTip2: 'ಎಲೆಯು ಚೌಕಟ್ಟನ್ನು ತುಂಬುವಂತೆ ಇರಲಿ.',
    retakeTip3: 'ಉತ್ತಮ ಹಗಲು ಬೆಳಕನ್ನು ಬಳಸಿ ಮತ್ತು ಕ್ಯಾಮೆರಾವನ್ನು ಸ್ಥಿರವಾಗಿ ಹಿಡಿಯಿರಿ.',
    retakeTip4: 'ಎಲೆಯನ್ನು ಕೈಯಲ್ಲಿ ಹಿಡಿಯಿರಿ ಅಥವಾ ಸರಳ ಮೇಲ್ಮೈಯ ಮೇಲೆ ಇರಿಸಿ.',
    analyzeAnyway: 'ಹೇಗಾದರೂ ವಿಶ್ಲೇಷಿಸಿ',
    tryAnotherPhoto: 'ಬೇರೆ ಫೋಟೋ ಪ್ರಯತ್ನಿಸಿ',
    unverifiedNote: 'ಇದು ಸಿಟ್ರಸ್ ಎಲೆ ಎಂದು ದೃಢಪಟ್ಟಿಲ್ಲ, ಆದ್ದರಿಂದ ಫಲಿತಾಂಶ ನಂಬಲರ್ಹವಲ್ಲದಿರಬಹುದು.',
  },

  hi: {
    welcome: 'स्वागत है',
    selectLanguage: 'अपनी पसंदीदा भाषा चुनें',
    title: 'पत्ती रोग पहचान प्रणाली',
    camera: 'कैमरा',
    gallery: 'गैलरी',
    predict: 'पहचानें',
    analyzing: 'विश्लेषण किया जा रहा है...',
    analyzingLeaf: 'पत्ती का विश्लेषण किया जा रहा है...',
    wait: 'इसमें 15-30 सेकंड लग सकते हैं',
    selectedImage: 'चयनित चित्र',
    predictionResult: 'पूर्वानुमान परिणाम',
    confidence: 'विश्वास',
    disease: 'रोग',
    severity: 'गंभीरता',
    infectedArea: 'संक्रमित क्षेत्र',
    spots: 'धब्बे',
    zones: 'क्षेत्र',
    uploadedImage: 'अपलोड किया गया चित्र',
    severityEstimation: 'चरण 2 गंभीरता अनुमान',
    severityCaption: 'हरा: पत्ती का ऊतक। लाल: संक्रमित क्षेत्र। ग्रिड: क्षेत्र का फैलाव।',
    severityBreakdown: 'गंभीरता स्कोर विवरण',
    formula: 'अंतिम स्कोर = 0.45×क्षेत्र + 0.20×रंग + 0.20×फैलाव + 0.15×भाग',
    area: 'क्षेत्र',
    color: 'रंग',
    spread: 'फैलाव',
    zone: 'क्षेत्र',
    treatmentRecommendation: 'उपचार की सलाह',
    treatment: 'उपचार',
    dose: 'मात्रा',
    method: 'विधि',
    timing: 'समय',
    interval: 'अंतराल',
    warning: 'चेतावनी',
    culturalAdvice: 'फसल प्रबंधन सलाह',
    monitoring: 'निगरानी',
    tryAgain: 'फिर से प्रयास करें',
    error: 'त्रुटि',
    permissionNeeded: 'अनुमति आवश्यक',
    cameraPermission: 'कैमरा अनुमति आवश्यक है',
    galleryPermission: 'गैलरी अनुमति आवश्यक है',
    requestTimeout: 'अनुरोध का समय समाप्त हो गया। सर्वर व्यस्त है। फिर से प्रयास करें।',
    healthyMessage: 'पत्ती स्वस्थ दिखाई देती है। उपचार की आवश्यकता नहीं है।',
    maintenance: 'रखरखाव',
    instruction: 'रोग का पता लगाने और उपचार की सलाह प्राप्त करने के लिए पत्ती की तस्वीर लें या चित्र अपलोड करें।',
    changeLanguage: 'भाषा बदलें',
    newAnalysis: 'दूसरी पत्ती का विश्लेषण करें',
    recommendationReason: 'सलाह का कारण',
    urgency: 'तत्कालता',
    leafInformation: 'पत्ती की जानकारी',
    leafSize: 'पत्ती का आकार',
    youngLeaf: 'नई पत्ती',
    matureLeaf: 'परिपक्व पत्ती',
    yes: 'हाँ',
    no: 'नहीं',
    model: 'मॉडल',
    // --- citrus gate (please have a Hindi speaker check the wording) ---
    photoTip: 'सुझाव: सादी सतह पर एक ही पत्ती रखें और फ्रेम भरकर फोटो लें।',
    notCitrusTitle: 'यह सिट्रस (नींबू वर्गीय) पत्ती नहीं लगती',
    notCitrusMsg: 'कृपया सिट्रस की एक पत्ती की साफ़ फोटो लें।',
    retakeTitle: 'कृपया फिर से फोटो लें',
    retakeMsg: 'पक्का नहीं है कि यह सिट्रस की पत्ती है। कृपया इन सुझावों के अनुसार फिर से फोटो लें:',
    retakeTip1: 'केवल एक पत्ती की फोटो लें।',
    retakeTip2: 'पत्ती से पूरा फ्रेम भरें।',
    retakeTip3: 'अच्छी दिन की रोशनी में फोटो लें और कैमरा स्थिर रखें।',
    retakeTip4: 'पत्ती को हाथ में पकड़ें या सादी सतह पर रखें।',
    analyzeAnyway: 'फिर भी विश्लेषण करें',
    tryAnotherPhoto: 'दूसरी फोटो आज़माएँ',
    unverifiedNote: 'यह पुष्टि नहीं हुई कि यह सिट्रस की पत्ती है, इसलिए परिणाम भरोसेमंद नहीं हो सकता।',
  },
};

const i18n = new I18n(translations);
i18n.enableFallback = true;
i18n.defaultLocale = 'en';

const LANGUAGE_CODES = { English: 'en', Kannada: 'kn', Hindi: 'hi' };

// ============================================================
// TEXT WRAPPER  (THE FIX)
// ------------------------------------------------------------
// Some phones use a custom system font. Android then computes a
// wrong line box when text has emoji / Kannada / Hindi glyphs and
// the label gets pushed out of view. Giving every Text an explicit
// lineHeight and turning off extra font padding fixes it.
// ============================================================
const Text = ({ style, ...props }) => {
  const flat = StyleSheet.flatten(style) || {};
  const extra =
    flat.fontSize && !flat.lineHeight
      ? { lineHeight: Math.round(flat.fontSize * 1.45) }
      : null;
  return <RNText {...props} style={[styles.noPad, style, extra]} />;
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

// ============================================================
// AUTOMATIC TRANSLATION OF AI OUTPUT
// ============================================================
const translateText = async (text, targetLanguage) => {
  if (!text || typeof text !== 'string' || targetLanguage === 'en') return text;
  try {
    const response = await fetch(
      `https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=${targetLanguage}&dt=t&q=${encodeURIComponent(text)}`
    );
    if (!response.ok) return text;
    const data = await response.json();
    if (Array.isArray(data) && Array.isArray(data[0])) {
      return data[0].map((item) => item[0]).join('');
    }
    return text;
  } catch (error) {
    console.log('Translation failed:', error.message);
    return text; // fall back to English
  }
};

const TEXT_FIELDS = [
  'disease',
  'severity_label',
  'grade_label',
  'pathological_symptoms',
  'leaf_size_category',
  'treatment',
  'dose',
  'method',
  'timing',
  'interval',
  'warning',
  'low_confidence_warning',
  'cultural_advice',
  'monitoring',
  'urgency_note',
  'reason',
];

// Keeps the ORIGINAL English values in _raw so that logic
// (healthy check, colours) still works after translation.
const translateResult = async (data, languageCode) => {
  const out = {
    ...data,
    _raw: {
      isHealthy: data.disease === 'Healthy',
      severity_label: data.severity_label,
      urgency_note: data.urgency_note,
    },
  };
  if (languageCode === 'en') return out;

  // All fields translated IN PARALLEL (much faster than one by one)
  await Promise.all(
    TEXT_FIELDS.map(async (field) => {
      if (data[field]) out[field] = await translateText(data[field], languageCode);
    })
  );
  return out;
};

// ============================================================
// MAIN APP
// ============================================================
export default function App() {
  const [language, setLanguage] = useState(null);
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  // [GATE] set when the backend says the photo is not (clearly) a citrus leaf:
  //        { status: 'reject' | 'retake', probability }
  const [gateIssue, setGateIssue] = useState(null);

  const t = (key) => i18n.t(key);

  const selectLanguage = (selected) => {
    i18n.locale = LANGUAGE_CODES[selected];
    setLanguage(selected);
    setResult(null);
    setError(null);
    setGateIssue(null);
  };

  // ---------------- PICK IMAGE ----------------
  const pickImage = async () => {
    setError(null);
    setResult(null);
    setGateIssue(null);
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert(t('permissionNeeded'), t('galleryPermission'));
      return;
    }
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaType
        ? ImagePicker.MediaType.Images
        : ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 0.8,
    });
    if (!res.canceled) setImage(res.assets[0].uri);
  };

  // ---------------- CAMERA ----------------
  const takePhoto = async () => {
    setError(null);
    setResult(null);
    setGateIssue(null);
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert(t('permissionNeeded'), t('cameraPermission'));
      return;
    }
    const res = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      quality: 0.8,
    });
    if (!res.canceled) setImage(res.assets[0].uri);
  };

  // ---------------- ANALYZE ----------------
  // force = true  ->  "Analyze anyway" on a borderline ("retake") photo
  const analyze = async (uri = image, force = false) => {
    if (!uri) return;
    setLoading(true);
    setResult(null);
    setError(null);
    setGateIssue(null);

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 120000);

    try {
      const form = new FormData();
      form.append('file', {
        uri: Platform.OS === 'android' ? uri : uri.replace('file://', ''),
        name: 'leaf.jpg',
        type: 'image/jpeg',
      });
      if (force) form.append('force', '1');   // [GATE] backend only honours this for "retake", never for "reject"

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: form,
        signal: controller.signal,
        headers: {
          // Free ngrok tunnels show an interstitial "Visit Site" warning page
          // to non-browser clients unless this header is present, which would
          // otherwise show up here as a generic network/parse failure.
          'ngrok-skip-browser-warning': 'true',
        },
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

      // [GATE] The backend answers HTTP 200 with no disease fields when the photo
      // is not a (clear) citrus leaf. Show a friendly card instead of a result.
      if (data.status === 'reject' || data.status === 'retake') {
        console.log('Citrus gate:', data.status, data.citrus_probability);
        setGateIssue({ status: data.status, probability: data.citrus_probability });
        return;
      }
      // Any other reply without a disease (e.g. unreadable image, "No file uploaded")
      if (!data.disease) {
        throw new Error(data.message || data.error || 'Unexpected response from server');
      }

      const translated = await translateResult(data, LANGUAGE_CODES[language] || 'en');
      setResult(translated);
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
    setResult(null);
    setError(null);
    setGateIssue(null);
  };

  // ---------------- COLOURS (use ORIGINAL English text) ----------------
  const severityColor = (label) => {
    const l = String(label || '').toLowerCase();
    if (l.includes('healthy')) return '#4CAF50';
    if (l.includes('mild') || l.includes('trace')) return '#8BC34A';
    if (l.includes('moderate')) return '#FFC107';
    if (l.includes('severe')) return '#FF5722';
    if (l.includes('critical')) return '#D32F2F';
    return '#888';
  };

  const urgencyColor = (note) => {
    if (!note) return '#888';
    const l = String(note).toLowerCase();
    if (l.includes('low')) return '#4CAF50';
    if (l.includes('moderate')) return '#FFC107';
    if (l.includes('high')) return '#FF5722';
    if (l.includes('critical')) return '#D32F2F';
    return '#888';
  };

  // ============================================================
  // LANGUAGE SELECTION SCREEN
  // ============================================================
  if (!language) {
    return (
      <View style={styles.languageScreen}>
        <Text style={styles.logo}>🌿</Text>
        <Text style={styles.welcomeTitle}>Welcome</Text>
        <Text style={styles.languageSubtitle}>Select your preferred language</Text>

        <Btn
          icon="🇬🇧"
          label="English"
          style={styles.languageButton}
          textStyle={styles.languageButtonText}
          onPress={() => selectLanguage('English')}
        />
        <Btn
          icon="🇮🇳"
          label="ಕನ್ನಡ"
          style={styles.languageButton}
          textStyle={styles.languageButtonText}
          onPress={() => selectLanguage('Kannada')}
        />
        <Btn
          icon="🇮🇳"
          label="हिन्दी"
          style={styles.languageButton}
          textStyle={styles.languageButtonText}
          onPress={() => selectLanguage('Hindi')}
        />
      </View>
    );
  }

  // ============================================================
  // MAIN UI
  // ============================================================
  const isHealthy = result?._raw?.isHealthy;

  return (
    <ScrollView contentContainerStyle={styles.container}>
      {/* HEADER */}
      <IconLabel icon="🌿" label={t('title')} textStyle={styles.heading} iconSize={26} />

      {/* CHANGE LANGUAGE */}
      <Btn
        icon="🌐"
        label={t('changeLanguage')}
        style={styles.changeLanguageButton}
        textStyle={styles.changeLanguageText}
        onPress={() => {
          setLanguage(null);
          setImage(null);
          setResult(null);
          setError(null);
          setGateIssue(null);
        }}
      />

      {/* IMAGE CARD */}
      <View style={styles.card}>
        <View style={styles.btnRow}>
          <Btn icon="📷" label={t('camera')} style={styles.btn} onPress={takePhoto} disabled={loading} />
          <Btn icon="🖼️" label={t('gallery')} style={styles.btn} onPress={pickImage} disabled={loading} />
        </View>

        {/* [GATE] camera guidance: the gate works best on one leaf filling the frame */}
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

      {/* [GATE] NOT A CITRUS LEAF / PLEASE RETAKE (with photo tips) */}
      {gateIssue && !loading && (
        <View
          style={[
            styles.gateBox,
            gateIssue.status === 'reject' ? styles.gateReject : styles.gateRetake,
          ]}
        >
          <IconLabel
            icon={gateIssue.status === 'reject' ? '🚫' : '📸'}
            label={gateIssue.status === 'reject' ? t('notCitrusTitle') : t('retakeTitle')}
            textStyle={styles.gateTitle}
            iconSize={18}
          />
          <Text style={styles.gateText}>
            {gateIssue.status === 'reject' ? t('notCitrusMsg') : t('retakeMsg')}
          </Text>

          {/* how to take a good photo */}
          {[1, 2, 3, 4].map((n) => (
            <View key={n} style={styles.tipRow}>
              <Text style={styles.tipBullet}>•</Text>
              <Text style={styles.gateTip}>{t(`retakeTip${n}`)}</Text>
            </View>
          ))}

          <View style={styles.gateBtnRow}>
            <Btn
              label={t('tryAnotherPhoto')}
              style={styles.btn}
              onPress={resetAnalysis}
            />
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
          {/* [GATE] shown only when the user forced an analysis on a borderline photo */}
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

          <View style={styles.resultCard}>
            <Text style={styles.resultTitle}>{t('predictionResult')}</Text>

            <Text style={styles.label}>{t('disease')}</Text>
            <Text style={styles.diseaseName}>{result.disease}</Text>

            <Text style={styles.label}>{t('confidence')}</Text>
            <View style={styles.bar}>
              <View
                style={[
                  styles.confFill,
                  { width: `${Math.min(Number(result.confidence || 0), 100)}%` },
                ]}
              >
                <Text style={styles.confText}>
                  {Number(result.confidence || 0).toFixed(1)}%
                </Text>
              </View>
            </View>

            <Text style={styles.severityLine}>
              {t('severity')}:{' '}
              <Text
                style={{
                  color: severityColor(result._raw?.severity_label),
                  fontWeight: '700',
                }}
              >
                {result.severity_label} ({Number(result.severity || 0).toFixed(2)}%)
              </Text>
            </Text>

            <Text style={styles.detailLine}>
              {t('infectedArea')}: {Number(result.infected_pct || 0).toFixed(2)}%
              {'  |  '}
              {t('spots')}: {result.spot_count || 0}
              {'  |  '}
              {t('zones')}: {result.zones_infected || 0}/9
            </Text>

            {result.model_architecture && (
              <Text style={styles.modelText}>
                {t('model')}: {result.model_architecture}
              </Text>
            )}
          </View>

          {/* IMAGES */}
          <View style={styles.resultCard}>
            <Text style={styles.sectionTitle}>{t('uploadedImage')}</Text>
            <Image source={{ uri: image }} style={styles.analysisImg} />

            {result.severity_image_url && (
              <>
                <Text style={styles.sectionTitle}>{t('severityEstimation')}</Text>
                <Image
                  source={{
                    uri: `${API_URL}${result.severity_image_url}`,
                    headers: { 'ngrok-skip-browser-warning': 'true' },
                  }}
                  style={styles.analysisImg}
                />
                <Text style={styles.visualCaption}>{t('severityCaption')}</Text>
              </>
            )}
          </View>

          {/* SEVERITY BREAKDOWN */}
          {result.breakdown && (
            <View style={styles.breakdown}>
              <Text style={styles.breakdownTitle}>{t('severityBreakdown')}</Text>
              <Text style={styles.formulaText}>{t('formula')}</Text>
              <ScoreRow label={`${t('area')} (×0.45)`} value={result.breakdown.area_score} color="#E07B54" />
              <ScoreRow label={`${t('color')} (×0.20)`} value={result.breakdown.color_score} color="#E0C454" />
              <ScoreRow label={`${t('spread')} (×0.20)`} value={result.breakdown.spread_score} color="#54A0E0" />
              <ScoreRow label={`${t('zone')} (×0.15)`} value={result.breakdown.zone_score} color="#54C054" />
            </View>
          )}

          {/* LEAF INFORMATION */}
          <View style={styles.resultCard}>
            <Text style={styles.sectionTitle}>{t('leafInformation')}</Text>
            <TRow label={t('leafSize')} value={result.leaf_size_category || '-'} />
            <TRow
              label={result.is_young_leaf ? t('youngLeaf') : t('matureLeaf')}
              value={result.is_young_leaf ? t('yes') : t('no')}
            />
          </View>

          {/* TREATMENT */}
          {result.treatment && !isHealthy && (
            <View style={styles.resultCard}>
              <Text style={styles.treatmentHeading}>{t('treatmentRecommendation')}</Text>

              {result.low_confidence_warning && (
                <View style={styles.warningBox}>
                  <Text style={styles.warningText}>{result.low_confidence_warning}</Text>
                </View>
              )}

              {result.urgency_note && (
                <Text style={[styles.urgency, { color: urgencyColor(result._raw?.urgency_note) }]}>
                  {result.urgency_note}
                </Text>
              )}

              <View style={styles.table}>
                <TRow label={t('treatment')} value={result.treatment} bold />
                {result.dose && <TRow label={t('dose')} value={result.dose} />}
                {result.method && <TRow label={t('method')} value={result.method} />}
                {result.timing && <TRow label={t('timing')} value={result.timing} />}
                {result.interval && <TRow label={t('interval')} value={result.interval} />}
              </View>

              {result.warning && (
                <View style={[styles.warningBox, styles.warningYellow]}>
                  <Text style={styles.warningYellowText}>
                    ⚠️ {t('warning')}: {result.warning}
                  </Text>
                </View>
              )}

              {result.cultural_advice && (
                <View style={styles.adviceBox}>
                  <Text style={styles.adviceText}>
                    <Text style={styles.bold}>{t('culturalAdvice')}: </Text>
                    {result.cultural_advice}
                  </Text>
                </View>
              )}

              {result.monitoring && (
                <View style={styles.adviceBox}>
                  <Text style={styles.adviceText}>
                    <Text style={styles.bold}>{t('monitoring')}: </Text>
                    {result.monitoring}
                  </Text>
                </View>
              )}

              {result.reason && (
                <Text style={styles.reasonText}>
                  <Text style={styles.bold}>{t('recommendationReason')}: </Text>
                  {result.reason}
                </Text>
              )}
            </View>
          )}

          {/* HEALTHY */}
          {isHealthy && (
            <View style={styles.resultCard}>
              <IconLabel
                icon="✅"
                label={t('healthyMessage')}
                textStyle={[styles.diseaseName, { color: '#4CAF50', flexShrink: 1, marginBottom: 0 }]}
                iconSize={20}
              />
              {result.cultural_advice && (
                <View style={[styles.adviceBox, { marginTop: 12 }]}>
                  <Text style={styles.adviceText}>
                    <Text style={styles.bold}>{t('maintenance')}: </Text>
                    {result.cultural_advice}
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
  tipText: { fontSize: 12, color: '#5c6b5c', fontStyle: 'italic', flexShrink: 1, textAlign: 'center' },

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
  breakdownLabel: { width: 95, fontSize: 11, color: '#555' },
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
  reasonText: { fontSize: 11, color: '#888', fontStyle: 'italic', marginTop: 8 },

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
});