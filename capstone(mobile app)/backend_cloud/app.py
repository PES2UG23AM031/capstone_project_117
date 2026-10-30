from flask import Flask, request, jsonify, send_from_directory, after_this_request
import os, json
import uuid
from flask_cors import CORS
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification
from werkzeug.utils import secure_filename
from rag_recommendation_enginee import get_recommendation
from severity_analysis import compute_comprehensive_severity, LITERATURE_SEVERITY_LEVELS
from citrus_gate import load_gate, check_citrus_decision   # [GATE] citrus vs not-citrus check
print("USING RAG ENGINE:", get_recommendation.__module__)            

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

# [GATE] Load the citrus gate (MobileNetV3). It runs on CPU: small model, keeps GPU memory for the ViT.
# Set DISABLE_GATE=1 to turn it off for debugging.
GATE_ENABLED = os.environ.get("DISABLE_GATE", "0") != "1"
GATE_MODEL_PATH = os.environ.get("GATE_MODEL_PATH", os.path.join(BASE_DIR, "citrus_gate_v4.pt"))
if GATE_ENABLED:
    load_gate(GATE_MODEL_PATH, device="cpu")
else:
    print("WARNING: citrus gate is DISABLED (DISABLE_GATE=1)")

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

    # Standard RAG label mapping
    grade = res["severity_grade"]
    if grade == 0:
        sev_label = "Healthy"
    elif grade in (1, 2):
        sev_label = "Mild"  # Grade 1 (Trace) & Grade 2 (Mild)
    elif grade == 3:
        sev_label = "Moderate"
    elif grade == 4:
        sev_label = "Severe"
    else:
        sev_label = "Critical"

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
    @after_this_request
    def _cleanup(response):
        try:
            os.remove(filepath)
        except OSError:
            pass
        return response
    # Stage 0 [GATE] — is this a citrus leaf at all? (accept / retake / reject)
    gate_status, citrus_p = "disabled", None
    if GATE_ENABLED:
        try:
            decision, citrus_p = check_citrus_decision(filepath)
        except Exception as e:
            print(f"[gate] could not read {stored_name}: {e}")
            msg = "Could not read this image. Please upload a JPG or PNG photo."
            return jsonify({"success": False, "status": "error", "error": msg, "message": msg})

        citrus_p = round(citrus_p, 3)
        print(f"[gate] {stored_name}: P(citrus)={citrus_p} -> {decision}")

        # Optional: the app may send force=1 to analyse a borderline ("retake") photo anyway.
        # A hard "reject" can never be forced.
        force = request.form.get("force") == "1"

        if decision == "reject" or (decision == "retake" and not force):
            if decision == "reject":
                msg = "This doesn't look like a citrus leaf. Please photograph a single citrus leaf."
            else:
                msg = ("I'm not sure this is a citrus leaf. Place one leaf on a plain surface "
                       "and fill the frame, then try again.")
            return jsonify({
                "success": False,
                "status": decision,            # "reject" or "retake"
                "error": msg,                  # human-readable (shown by apps that display data.error)
                "message": msg,
                "citrus_probability": citrus_p,
            })
        gate_status = "accepted" if decision == "accept" else "unverified"

    # Stage 1 — ViT / Classifier Disease Prediction
    with torch.no_grad():
        output = model(preprocess_image(filepath))
        logits = output.logits if hasattr(output, "logits") else output
        probs = torch.softmax(logits, dim=1)[0]
        confidence, class_index = torch.max(probs, dim=0)

    disease = CLASSES[class_index.item()]
    confidence_pct = round(float(confidence.item()) * 100, 1)

    # Top-3 predictions for comprehensive diagnostic insight
    topk_confs, topk_indices = torch.topk(probs, k=min(3, len(CLASSES)))
    top_predictions = [
        {"disease": CLASSES[idx.item()], "confidence": round(float(c.item()) * 100, 1)}
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
        "status":                 "ok",
        "gate_status":            gate_status,      # "accepted", "unverified" (forced), or "disabled"
        "citrus_probability":     citrus_p,
        "disease":                disease,
        "confidence":             confidence_pct,
        "model_architecture":     "Vision Transformer (ViT-Base/16)",
        "top_predictions":        top_predictions,
        "severity":               severity["percentage"],
        "severity_label":         severity["label"],
        "severity_grade":         severity.get("severity_grade", 0),
        "grade_label":            severity.get("grade_label", ""),
        "pathological_symptoms":  severity.get("pathological_symptoms", ""),
        "leaf_size_category":     severity.get("leaf_size_category", ""),
        "is_young_leaf":          severity.get("is_young_leaf", False),
        "infected_pct":           severity["infected_pct"],
        "spot_count":             severity["spot_count"],
        "zones_infected":         severity["zones_infected"],
        "breakdown":              severity["breakdown"],
        "severity_image_url":     severity["severity_image_url"],

        # RAG treatment recommendation
        "treatment":              rec.get("recommended_treatment", "") or rec.get("treatment", ""),
        "dose":                   rec.get("dose", ""),
        "timing":                 rec.get("timing", ""),
        "method":                 rec.get("method", ""),
        "interval":               rec.get("interval", ""),
        "warning":                rec.get("warning", ""),
        "cultural_advice":        rec.get("cultural_advice", ""),
        "monitoring":             rec.get("monitoring", ""),
        "urgency_note":           rec.get("urgency_note", ""),
        "low_confidence_warning": rec.get("low_confidence_warning", ""),
        "reason":                 rec.get("reason", ""),
        "source":                 rec.get("source", "rag_vector_search"),
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
        "citrus_gate": "MobileNetV3-Large (v4)" if GATE_ENABLED else "disabled",   # [GATE]
    })

if __name__ == "__main__":
    # debug=False: debug mode starts a reloader that loads every model twice (wastes GPU memory).
    app.run(host="0.0.0.0", port=5000, debug=False)