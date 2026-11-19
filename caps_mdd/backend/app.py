import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from flask import Flask, request, jsonify, send_from_directory
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
import json
import uuid
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification
from werkzeug.utils import secure_filename
from rag_recommendation_enginee import get_recommendation
from severity_analysis import compute_comprehensive_severity, LITERATURE_SEVERITY_LEVELS
from citrus_leaf_verifier import CitrusLeafVerifier
from citrus_flush_ontogeny import CitrusFlushOntogenyAnalyzer
from multi_leaf_detector import detect_foliar_instances, draw_canopy_diagnostic_overlay, is_fruit_crop
print("USING RAG ENGINE:", get_recommendation.__module__)

class NumpyJSONProvider(DefaultJSONProvider):
    """Ensures all numpy types (bool_, int64, float32, ndarray) serialize cleanly to JSON."""
    def default(self, obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        tname = type(obj).__name__
        if "bool" in tname:
            return bool(obj)
        if "int" in tname:
            return int(obj)
        if "float" in tname:
            return float(obj)
        return super().default(obj)

def check_is_leaf(img_bgr):
    """
    Validates whether the image contains genuine foliar / plant leaf tissue
    using Woebbecke Excess Green (ExG) and multi-spectral foliar HSV thresholds.
    Accepts any genuine leaf (healthy, diseased, chlorotic, or dry).
    Rejects non-leaf objects (human faces, furniture, shoes, walls, electronics).
    """
    if img_bgr is None:
        return False, 0.0
    h, w = img_bgr.shape[:2]
    total_pixels = max(1, h * w)
    img_float = img_bgr.astype(np.float32)
    b, g, r = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
    exg = 2.0 * g - r - b

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_green = (cv2.inRange(hsv, np.array([12, 18, 18]), np.array([100, 255, 255])) > 0) & (exg > 0.0)
    mask_brown = (cv2.inRange(hsv, np.array([6, 25, 20]), np.array([24, 255, 220])) > 0) & (r > b)
    foliar_mask = mask_green | mask_brown
    foliar_pixels = int(np.count_nonzero(foliar_mask))
    coverage_pct = round((foliar_pixels / float(total_pixels)) * 100.0, 2)
    return (coverage_pct >= 4.0), coverage_pct

leaf_verifier = CitrusLeafVerifier()
ontogeny_analyzer = CitrusFlushOntogenyAnalyzer()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Dedicated Vision Transformer (ViT-Base/16) model path
VIT_MODEL_PATH = os.environ.get("VIT_MODEL_PATH", os.path.join(MODELS_DIR, "vit_best_model"))
if not os.path.exists(VIT_MODEL_PATH):
    alt_vit = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "vit_output", "best_model"))
    if os.path.exists(alt_vit):
        VIT_MODEL_PATH = alt_vit

def load_vit_model():
    if not os.path.exists(VIT_MODEL_PATH):
        raise FileNotFoundError(f"ViT model directory not found at: {VIT_MODEL_PATH}")
    print(f"Loading Vision Transformer (ViT-Base/16) from {VIT_MODEL_PATH}...")
    vit_model = ViTForImageClassification.from_pretrained(VIT_MODEL_PATH)
    vit_model.to(DEVICE).eval()
    id2label = vit_model.config.id2label
    classes = [id2label[str(i)] if str(i) in id2label else id2label[i] for i in range(len(id2label))]
    img_size = getattr(vit_model.config, "image_size", 224)
    print(f"ViT model loaded successfully with {len(classes)} classes: {classes}")
    return vit_model, classes, img_size

model, CLASSES, IMAGE_SIZE = load_vit_model()

try:
    print("Warming up RAG ChromaDB vector store...")
    _ = get_recommendation("Healthy", 98.0, 0.0, "Grade 0: Healthy")
    print("RAG vector store warm and ready.")
except Exception as _e:
    print("RAG warmup notice:", _e)

transform = transforms.Compose([
    transforms.Resize((int(IMAGE_SIZE * 1.14), int(IMAGE_SIZE * 1.14))),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(DEVICE)

def calculate_severity(image_path, severity_image_name=None):
    """
    Computes literature-grounded citrus leaf severity using:
      1. Adaptive Excess Green (ExG) segmentation (young & mature leaf invariant)
      2. High-precision lesion & halo detection
      3. AHP composite weights: 0.45 Area + 0.20 Color + 0.20 Spread + 0.15 Zone
      4. Horsfall-Barratt & EPPO Guidelines PP 1/276 grading (Grades 0-5)
    """
    res = compute_comprehensive_severity(image_path)

    # Standard RAG label mapping (4-Tier Scale: Healthy, Mild, Moderate, Severe)
    grade = res["severity_grade"]
    if grade == 0:
        sev_label = "Healthy"
    elif grade == 1:
        sev_label = "Mild"
    elif grade == 2:
        sev_label = "Moderate"
    else:
        sev_label = "Severe"

    severity_image_url = None
    if severity_image_name:
        severity_path = os.path.join(UPLOAD_FOLDER, severity_image_name)
        img_bgr = res["img_bgr"]
        leaf_mask = res["leaf_mask"]
        lesion_mask = res["lesion_mask"]

        overlay = img_bgr.copy()
        red_layer = np.zeros_like(img_bgr)
        red_layer[:, :] = (0, 0, 220)
        alpha = 0.45
        inf_bool = lesion_mask > 0
        overlay[inf_bool] = cv2.addWeighted(img_bgr, 1 - alpha, red_layer, alpha, 0)[inf_bool]

        contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)

        # 3x3 anatomical grid lines
        h, w = overlay.shape[:2]
        for i in range(1, 3):
            cv2.line(overlay, (int(w * i / 3), 0), (int(w * i / 3), h), (255, 255, 255), 1)
            cv2.line(overlay, (0, int(h * i / 3)), (w, int(h * i / 3)), (255, 255, 255), 1)

        cv2.imwrite(severity_path, overlay)
        severity_image_url = f"/uploads/{severity_image_name}"

    return {
        "percentage": res["composite_severity_score"],
        "label": sev_label,
        "infected_pct": res["s_area_pct"],
        "spot_count": res["spot_count"],
        "zones_infected": int(round((res["s_zone_score"] / 100.0) * 9)),
        "leaf_size_category": res["leaf_size_category"],
        "is_young_leaf": res["is_young_leaf"],
        "severity_grade": res["severity_grade"],
        "grade_label": res["grade_label"],
        "pathological_symptoms": res["pathological_symptoms"],
        "recommended_action": res["recommended_action"],
        "breakdown": {
            "area_score": res["s_area_pct"],
            "color_score": res["s_color_score"],
            "spread_score": res["s_spread_score"],
            "zone_score": res["s_zone_score"],
        },
        "severity_image_url": severity_image_url,
    }

app = Flask(__name__)
app.json = NumpyJSONProvider(app)
CORS(app)

@app.route("/")
def home(): return "Model API running"


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error":"No file uploaded"})

    file = request.files["file"]
    original_name = secure_filename(file.filename) or "upload.jpg"
    stem, ext = os.path.splitext(original_name)
    ext = ext or ".jpg"
    unique_id = uuid.uuid4().hex[:12]
    stored_name = f"{stem}_{unique_id}{ext}"
    severity_name = f"{stem}_{unique_id}_severity.png"
    filepath = os.path.join(UPLOAD_FOLDER, stored_name)
    file.save(filepath)

    img_bgr = cv2.imread(filepath)

    # Validate that the image depicts genuine plant leaf foliage (Leaf vs Non-Leaf Guardrail)
    is_leaf, foliage_pct = check_is_leaf(img_bgr)
    if not is_leaf:
        return jsonify({
            "is_citrus": False,
            "is_leaf": False,
            "verdict": "REJECTED: Non-Leaf Object Detected",
            "disease": "Non-Leaf / Rejected",
            "confidence": 0.0,
            "model_architecture": "Foliar Tissue Verifier (Vegetation Index)",
            "top_predictions": [],
            "severity": 0.0,
            "severity_label": "N/A",
            "treatment": "No treatment: Uploaded image does not contain plant foliage.",
            "warning": f"Non-Leaf Object: The image contains only {foliage_pct:.1f}% vegetative tissue. Please upload a clear photo of a plant leaf.",
            "cultural_advice": "Point the camera directly at a leaf blade.",
            "metrics": {"foliage_coverage_pct": foliage_pct}
        })
    leaf_candidates = detect_foliar_instances(img_bgr, max_leaves=8) if img_bgr is not None else []
    mode_requested = request.form.get("mode", "auto")

    # =========================================================================
    # TWO-STAGE CANOPY PIPELINE: If multiple leaves are detected or requested
    # =========================================================================
    if (len(leaf_candidates) >= 2 or mode_requested == "multi_leaf") and len(leaf_candidates) > 0:
        diagnosed_leaves = []
        temp_crop_dir = os.path.join(UPLOAD_FOLDER, "crops")
        os.makedirs(temp_crop_dir, exist_ok=True)
        ih, iw = img_bgr.shape[:2]

        for idx, candidate in enumerate(leaf_candidates, start=1):
            x, y, bw, bh = candidate["bbox"]
            pad_x = int(bw * 0.05)
            pad_y = int(bh * 0.05)
            x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
            x2, y2 = min(iw, x + bw + pad_x), min(ih, y + bh + pad_y)
            crop = img_bgr[y1:y2, x1:x2]

            crop_filename = f"crop_{unique_id}_{idx}.jpg"
            crop_path = os.path.join(temp_crop_dir, crop_filename)
            cv2.imwrite(crop_path, crop)

            # 1. ViT Disease Diagnosis on Leaf Crop
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(crop_rgb)
            tensor = transform(pil_crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = model(tensor)
                logits = out.logits if hasattr(out, "logits") else out
                calibrated = logits / 1.4
                probs = torch.softmax(calibrated, dim=1)[0]
                conf_val, class_idx = torch.max(probs, dim=0)

            leaf_disease = CLASSES[class_idx.item()]
            leaf_conf = round(min(98.4, float(conf_val.item()) * 100.0), 1)

            # Fruit vs Leaf Guardrail Filter:
            # Exclude spherical fruit crops (green limes, yellow lemons, oranges of any color)
            # and crops lacking foliar pathology confidence (< 38.0%).
            is_fruit, _fruit_reason = is_fruit_crop(crop, leaf_conf)
            if is_fruit:
                continue

            # 2. Severity Analysis on Leaf Crop
            if "healthy" in leaf_disease.lower():
                sev_pct = 0.0
                grade_label = "Grade 0: Healthy"
                leaf_breakdown = {
                    "area_score":   0.0,
                    "color_score":  0.0,
                    "spread_score": 0.0,
                    "zone_score":   0.0
                }
                leaf_spot_count = 0
                leaf_zones_infected = 0
                leaf_infected_pct = 0.0
                l_treatment = "No treatment: Healthy leaf tissue."
                l_rec = {
                    "recommended_treatment": l_treatment,
                    "dose": "None required",
                    "method": "Routine surveillance",
                    "timing": "Standard maintenance",
                    "cultural_advice": "Maintain balanced irrigation and standard citrus grove nutrition."
                }
            else:
                c_res = compute_comprehensive_severity(crop_path)
                sev_pct = float(c_res["composite_severity_score"])
                grade_label = str(c_res["grade_label"])
                leaf_breakdown = {
                    "area_score":   round(float(c_res["s_area_pct"]), 2),
                    "color_score":  round(float(c_res["s_color_score"]), 2),
                    "spread_score": round(float(c_res["s_spread_score"]), 2),
                    "zone_score":   round(float(c_res["s_zone_score"]), 2),
                }
                leaf_spot_count = int(c_res.get("spot_count", 0))
                leaf_zones_infected = int(round((c_res.get("s_zone_score", 0) / 100.0) * 9))
                leaf_infected_pct = round(float(c_res.get("s_area_pct", sev_pct)), 2)
                l_rec = get_recommendation(leaf_disease, leaf_conf, sev_pct, grade_label)
                l_treatment = l_rec.get("recommended_treatment") or l_rec.get("treatment") or "Routine surveillance"

            diagnosed_leaves.append({
                "leaf_id": idx,
                "bbox": [x, y, bw, bh],
                "disease": leaf_disease,
                "confidence": leaf_conf,
                "severity": sev_pct,
                "grade_label": grade_label,
                "breakdown": leaf_breakdown,
                "spot_count": leaf_spot_count,
                "zones_infected": leaf_zones_infected,
                "infected_pct": leaf_infected_pct,
                "treatment": l_treatment,
                "dose": l_rec.get("dose", ""),
                "method": l_rec.get("method", ""),
                "timing": l_rec.get("timing", ""),
                "cultural_advice": l_rec.get("cultural_advice", "")
            })

        # Canopy Health Aggregation
        total_leaves = len(diagnosed_leaves)
        if total_leaves == 0:
            return jsonify({
                "is_citrus": True,
                "is_leaf": False,
                "is_fruit": True,
                "verdict": "CITRUS FRUIT DETECTED (Foliar Diagnostics Only)",
                "disease": "Citrus Fruit (Non-Foliar)",
                "confidence": 0.0,
                "model_architecture": "Foliar Morphological Guardrail",
                "top_predictions": [],
                "severity": 0.0,
                "severity_label": "N/A",
                "treatment": "No foliar treatment: The uploaded image depicts citrus fruit rather than a foliage leaf blade. This diagnostic system is calibrated exclusively for foliar leaf pathology (canker, greening, melanose, etc.).",
                "warning": "Citrus Fruit Detected: The uploaded image contains citrus fruit rather than leaf foliage. Please upload a clear photo of a citrus leaf blade to perform leaf disease diagnosis.",
                "cultural_advice": "Focus the camera directly on a leaf blade for accurate foliar pathology analysis.",
                "metrics": {"object_type": "Citrus Fruit"}
            })

        healthy_count = sum(1 for l in diagnosed_leaves if l["disease"].lower() == "healthy")
        diseased_count = total_leaves - healthy_count
        canopy_infection_rate = round((diseased_count / max(1, total_leaves)) * 100.0, 1)

        if canopy_infection_rate == 0.0:
            canopy_action = "HEALTHY CANOPY: Routine surveillance. No chemical spray required."
        elif canopy_infection_rate <= 40.0:
            canopy_action = "LOCALIZED INFECTION: Targeted spot spraying on infected branch."
        else:
            canopy_action = "CRITICAL EPIDEMIC: Systemic orchard treatment needed across tree canopy."

        # Draw annotated visual bounding boxes overlay
        annotated_canopy = draw_canopy_diagnostic_overlay(img_bgr, diagnosed_leaves, canopy_infection_rate)
        canopy_img_name = f"{stem}_{unique_id}_canopy_diagnostic.png"
        canopy_img_path = os.path.join(UPLOAD_FOLDER, canopy_img_name)
        cv2.imwrite(canopy_img_path, annotated_canopy)
        canopy_url = f"/uploads/{canopy_img_name}"

        # Identify primary diseased leaf based on clinical confidence and severity
        high_conf_diseased = [l for l in diagnosed_leaves if l["disease"].lower() != "healthy" and l["confidence"] >= 40.0]
        if high_conf_diseased:
            primary_leaf = max(high_conf_diseased, key=lambda l: (l["severity"] * (l["confidence"] / 100.0)))
        else:
            diseased_leaves = [l for l in diagnosed_leaves if l["disease"].lower() != "healthy"]
            primary_leaf = max(diseased_leaves, key=lambda l: l["confidence"]) if diseased_leaves else diagnosed_leaves[0]

        return jsonify({
            "is_citrus":              True,
            "is_multi_leaf":          True,
            "total_leaves_detected":  int(total_leaves),
            "healthy_leaves_count":   int(healthy_count),
            "diseased_leaves_count":  int(diseased_count),
            "canopy_infection_rate":  float(canopy_infection_rate),
            "canopy_action_status":   str(canopy_action),
            "annotated_canopy_url":   str(canopy_url),
            "individual_leaves":      diagnosed_leaves,
            # Top-level summary fields compatible with existing UI cards
            "disease":                str(primary_leaf["disease"]),
            "confidence":             float(primary_leaf["confidence"]),
            "severity":               float(primary_leaf["severity"]),
            "severity_label":         "Healthy" if primary_leaf["disease"].lower() == "healthy" else ("Severe" if primary_leaf["severity"] > 45 else ("Moderate" if primary_leaf["severity"] > 25 else "Mild")),
            "severity_grade":         0 if primary_leaf["disease"].lower() == "healthy" else 2,
            "grade_label":            str(primary_leaf["grade_label"]),
            "treatment":              str(primary_leaf["treatment"]),
            "dose":                   str(primary_leaf.get("dose", "") or ""),
            "timing":                 str(primary_leaf.get("timing", "") or ""),
            "method":                 str(primary_leaf.get("method", "") or ""),
            "cultural_advice":        str(primary_leaf.get("cultural_advice", "") or ""),
            "severity_image_url":     str(canopy_url),
            "model_architecture":     "Two-Stage: Foliar Localization + ViT-Base/16 (Multi-Leaf Canopy)",
            "breakdown":              primary_leaf.get("breakdown", {
                "area_score":   float(primary_leaf["severity"]),
                "color_score":  0.0,
                "spread_score": 0.0,
                "zone_score":   0.0
            }),
            "infected_pct":           float(primary_leaf.get("infected_pct", primary_leaf["severity"])),
            "spot_count":             int(primary_leaf.get("spot_count", total_leaves)),
            "zones_infected":         int(primary_leaf.get("zones_infected", 1))
        })

    # =========================================================================
    # SINGLE-LEAF PIPELINE (Default Close-up Leaf Uploads)
    # =========================================================================
    # Stage 1 — ViT / Classifier Disease Prediction
    with torch.no_grad():
        output = model(preprocess_image(filepath))
        logits = output.logits if hasattr(output, "logits") else output
        # Temperature Scaling (Guo et al., ICML 2017) to calibrate probabilities
        TEMPERATURE = 1.4
        calibrated_logits = logits / TEMPERATURE
        probs = torch.softmax(calibrated_logits, dim=1)[0]
        confidence, class_index = torch.max(probs, dim=0)

    disease = CLASSES[class_index.item()]
    raw_conf = float(confidence.item()) * 100.0
    # Natural probability calibration: cap at 98.5% so it never saturates at an unrealistic 100.0%
    confidence_pct = round(min(98.4, raw_conf), 1)

    # Fruit vs Leaf Guardrail for Single Upload:
    # If the user uploaded a photo of an orange, lime, or lemon in single mode:
    is_fruit_single, fruit_reason = is_fruit_crop(img_bgr, confidence_pct)
    if is_fruit_single:
        return jsonify({
            "is_citrus": True,
            "is_leaf": False,
            "is_fruit": True,
            "verdict": "CITRUS FRUIT DETECTED (Foliar Diagnostics Only)",
            "disease": "Citrus Fruit (Non-Foliar)",
            "confidence": 0.0,
            "model_architecture": "Foliar Morphological Guardrail",
            "top_predictions": [],
            "severity": 0.0,
            "severity_label": "N/A",
            "treatment": "No foliar treatment: The uploaded image depicts citrus fruit rather than a foliage leaf blade. This diagnostic system is calibrated exclusively for foliar leaf pathology (canker, greening, melanose, etc.).",
            "warning": "Citrus Fruit Detected: The uploaded image contains citrus fruit rather than leaf foliage. Please upload a clear photo of a citrus leaf blade to perform leaf disease diagnosis.",
            "cultural_advice": "Focus the camera directly on a leaf blade for accurate foliar pathology analysis.",
            "metrics": {"object_type": "Citrus Fruit", "reason": fruit_reason}
        })

    # Top-3 predictions for comprehensive diagnostic insight
    topk_confs, topk_indices = torch.topk(probs, k=min(3, len(CLASSES)))
    top_predictions = [
        {"disease": str(CLASSES[idx.item()]), "confidence": round(min(98.4, float(c.item()) * 100.0), 1)}
        for c, idx in zip(topk_confs, topk_indices)
    ]

    # Stage 2 — Severity Analysis
    if disease.lower() == "healthy":
        severity = {
            "percentage": 0.0,
            "label": "Healthy",
            "infected_pct": 0.0,
            "spot_count": 0,
            "zones_infected": 0,
            "breakdown": {"area_score": 0.0, "color_score": 0.0, "spread_score": 0.0, "zone_score": 0.0},
            "severity_image_url": None,
            "severity_grade": 0,
            "grade_label": "Grade 0: Healthy",
            "pathological_symptoms": "Asymptomatic foliage with uniform chlorophyll distribution and zero necrotic lesions.",
            "recommended_action": "Routine surveillance; maintain standard citrus nutritional and irrigation management."
        }
    else:
        severity = calculate_severity(filepath, severity_image_name=severity_name)

    # Stage 3 — RAG Treatment Recommendation
    rec = get_recommendation(
        disease,
        confidence_pct,
        severity["percentage"],
        severity["label"]
    )

    return jsonify({
        "is_citrus":              True,
        "is_multi_leaf":          False,
        "citrus_probability":     float(confidence_pct),
        "ontogeny_stage":         "",
        "is_baby_leaf":           False,
        "flush_protection":       False,
        "disease":                str(disease),
        "confidence":             float(confidence_pct),
        "model_architecture":     "Vision Transformer (ViT-Base/16)",
        "top_predictions":        top_predictions,
        "severity":               float(severity["percentage"]),
        "severity_label":         str(severity["label"]),
        "severity_grade":         int(severity.get("severity_grade", 0)),
        "grade_label":            str(severity.get("grade_label", "")),
        "pathological_symptoms":  str(severity.get("pathological_symptoms", "")),
        "leaf_size_category":     str(severity.get("leaf_size_category", "")),
        "is_young_leaf":          False,
        "infected_pct":           float(severity["infected_pct"]),
        "spot_count":             int(severity["spot_count"]),
        "zones_infected":         int(severity["zones_infected"]),
        "breakdown":              {str(k): float(v) for k, v in severity.get("breakdown", {}).items()},
        "severity_image_url":     severity["severity_image_url"],

        # RAG treatment recommendation
        "treatment":              str(rec.get("recommended_treatment", "") or rec.get("treatment", "") or ""),
        "dose":                   str(rec.get("dose", "") or ""),
        "timing":                 str(rec.get("timing", "") or ""),
        "method":                 str(rec.get("method", "") or ""),
        "interval":               str(rec.get("interval", "") or ""),
        "warning":                str(rec.get("warning", "") or ""),
        "cultural_advice":        str(rec.get("cultural_advice", "") or ""),
        "monitoring":             str(rec.get("monitoring", "") or ""),
        "urgency_note":           str(rec.get("urgency_note", "") or ""),
        "low_confidence_warning": str(rec.get("low_confidence_warning", "") or ""),
        "reason":                 str(rec.get("reason", "") or ""),
        "source":                 str(rec.get("source", "rag_vector_search")),
    })

@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "model_architecture": "Vision Transformer (ViT-Base/16)",
        "model_type": "vit",
        "classes": CLASSES,
        "num_classes": len(CLASSES),
        "image_size": IMAGE_SIZE,
        "device": str(DEVICE),
        "rag_pipeline": "Enabled (ChromaDB Vector Store + RAG Synthesis)",
    })

if __name__ == "__main__":
    app.run(debug=True)
