"""
Citrus leaf disease API  (merged version)

Pipeline
  0. Foliage check            : is there plant tissue in the photo at all?
  1. Multi-leaf detection     : 1 leaf -> single-leaf pipeline, 2+ leaves -> canopy pipeline
  2. Citrus gate (MobileNetV3): accept / retake / reject   (single-leaf pipeline)
  3. ViT-Base/16 diagnosis    : 12 classes, temperature-scaled confidence
  4. Fruit / low-confidence guard
  5. Severity (OLS-regression CSI, grades 0-3)
  6. RAG treatment recommendation (ChromaDB + Groq)

Every reply is HTTP 200 JSON with a "status" field:
  "ok"      -> normal result
  "reject"  -> not a citrus leaf (cannot be forced)
  "retake"  -> borderline photo (app may resend with force=1)
  "fruit"   -> citrus fruit, not a leaf
  "error"   -> unreadable image / no file
"""
import os
import time
import uuid
import base64

from flask import Flask, request, jsonify, send_from_directory, after_this_request
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification
from werkzeug.utils import secure_filename

from rag_recommendation_enginee import get_recommendation
from severity_analysis import compute_comprehensive_severity
from citrus_gate import load_gate, check_citrus_decision          # [GATE]
from multi_leaf_detector import (                                  # [CANOPY]
    detect_foliar_instances,
    draw_canopy_diagnostic_overlay,
    is_fruit_crop,
)

print("USING RAG ENGINE:", get_recommendation.__module__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
CROP_FOLDER = os.path.join(UPLOAD_FOLDER, "crops")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROP_FOLDER, exist_ok=True)

# Temperature scaling of the ViT logits (Guo et al., ICML 2017). 1.0 = raw softmax.
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.4"))
# Display cap on the confidence. 100 = no cap (real value shown). Your friend's version used 98.4,
# which made almost every confident answer show exactly 98.4%.
MAX_CONF_SHOWN = float(os.environ.get("MAX_CONF_SHOWN", "100"))
MAX_LEAVES = 8
# The fruit shape rule can mistake a curled leaf (e.g. CurlVirus) for a round fruit. When the citrus
# gate is at least this sure the photo is a citrus LEAF, the shape rule is ignored.
FRUIT_SKIP_GATE_P = float(os.environ.get("FRUIT_SKIP_GATE_P", "0.9"))
KEEP_FILES_SECONDS = 30 * 60


class NumpyJSONProvider(DefaultJSONProvider):
    """Lets jsonify() handle numpy scalars / arrays."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
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

# [GATE] MobileNetV3 citrus gate (CPU). DISABLE_GATE=1 turns it off for debugging.
GATE_ENABLED = os.environ.get("DISABLE_GATE", "0") != "1"
GATE_MODEL_PATH = os.environ.get("GATE_MODEL_PATH", os.path.join(BASE_DIR, "citrus_gate_v4.pt"))
# The gate was tuned on single-leaf photos. In canopy mode it is OFF by default;
# set CANOPY_GATE=1 to also reject non-citrus crops (test on real branch photos first).
CANOPY_GATE = os.environ.get("CANOPY_GATE", "0") == "1"
if GATE_ENABLED:
    load_gate(GATE_MODEL_PATH, device="cpu")
else:
    print("WARNING: citrus gate is DISABLED (DISABLE_GATE=1)")

transform = transforms.Compose([
    transforms.Resize((int(IMAGE_SIZE * 1.14), int(IMAGE_SIZE * 1.14))),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def classify(pil_img):
    """ViT diagnosis of one PIL image -> (disease, confidence_pct, top3)."""
    tensor = transform(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)
        logits = out.logits if hasattr(out, "logits") else out
        probs = torch.softmax(logits / TEMPERATURE, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
        k_conf, k_idx = torch.topk(probs, k=min(3, len(CLASSES)))
    disease = CLASSES[int(idx.item())]
    conf_pct = round(min(MAX_CONF_SHOWN, float(conf.item()) * 100.0), 1)
    top3 = [
        {"disease": str(CLASSES[int(i.item())]),
         "confidence": round(min(MAX_CONF_SHOWN, float(c.item()) * 100.0), 1)}
        for c, i in zip(k_conf, k_idx)
    ]
    return disease, conf_pct, top3


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def check_is_leaf(img_bgr):
    """Cheap foliage check (Excess-Green + HSV). Rejects walls, faces, furniture, etc."""
    if img_bgr is None:
        return False, 0.0
    h, w = img_bgr.shape[:2]
    f = img_bgr.astype(np.float32)
    b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    exg = 2.0 * g - r - b
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    green = (cv2.inRange(hsv, np.array([12, 18, 18]), np.array([100, 255, 255])) > 0) & (exg > 0.0)
    brown = (cv2.inRange(hsv, np.array([6, 25, 20]), np.array([24, 255, 220])) > 0) & (r > b)
    coverage = round(int(np.count_nonzero(green | brown)) / float(max(1, h * w)) * 100.0, 2)
    return coverage >= 4.0, coverage


def severity_label_for(disease, grade):
    """Label passed to the RAG engine.
    A leaf the ViT calls diseased but whose CSI is under the Grade-1 cut-off is treated as
    'Mild' (monitor). Passing 'Healthy' for a diseased class would make the RAG engine
    answer 'Emergency'."""
    if str(disease).lower() == "healthy":
        return "Healthy"
    if grade <= 1:
        return "Mild"
    if grade == 2:
        return "Moderate"
    return "Severe"


def file_to_b64(path, max_w=900):
    """Small JPEG as base64, so the app never needs a second request for the image."""
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        h, w = img.shape[:2]
        if w > max_w:
            img = cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf).decode() if ok else None
    except Exception as e:
        print("file_to_b64 failed:", e)
        return None


def prune_uploads():
    """Cloud Run's disk lives in memory: delete old files so it never fills up."""
    now = time.time()
    for folder in (UPLOAD_FOLDER, CROP_FOLDER):
        try:
            for name in os.listdir(folder):
                p = os.path.join(folder, name)
                if os.path.isfile(p) and now - os.path.getmtime(p) > KEEP_FILES_SECONDS:
                    os.remove(p)
        except OSError:
            pass


def reply(status, message, **extra):
    body = {"success": False, "status": status, "error": message, "message": message}
    body.update(extra)
    return jsonify(body)


def rec_fields(rec):
    """Flatten a RAG recommendation into plain strings."""
    g = lambda k: str(rec.get(k, "") or "")
    return {
        "treatment": str(rec.get("recommended_treatment", "") or rec.get("treatment", "") or ""),
        "dose": g("dose"), "method": g("method"), "timing": g("timing"),
        "interval": g("interval"), "warning": g("warning"),
        "cultural_advice": g("cultural_advice"), "monitoring": g("monitoring"),
        "urgency_note": g("urgency_note"), "reason": g("reason"),
        "low_confidence_warning": g("low_confidence_warning"),
        "source": str(rec.get("source", "rag_vector_search")),
    }


HEALTHY_REC = {
    "treatment": "No treatment: Healthy leaf tissue.",
    "dose": "None required", "method": "Routine surveillance", "timing": "Standard maintenance",
    "interval": "", "warning": "", "monitoring": "", "urgency_note": "", "reason": "",
    "low_confidence_warning": "", "source": "static",
    "cultural_advice": "Maintain balanced irrigation and standard citrus grove nutrition.",
}

ZERO_BREAKDOWN = {"area_score": 0.0, "color_score": 0.0, "spread_score": 0.0, "zone_score": 0.0}


def make_severity_overlay(res, out_path):
    """Red lesion overlay + yellow contours + 3x3 zone grid."""
    img_bgr, lesion_mask = res["img_bgr"], res["lesion_mask"]
    overlay = img_bgr.copy()
    red = np.zeros_like(img_bgr)
    red[:, :] = (0, 0, 220)
    inf = lesion_mask > 0
    overlay[inf] = cv2.addWeighted(img_bgr, 0.55, red, 0.45, 0)[inf]
    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)
    h, w = overlay.shape[:2]
    for i in range(1, 3):
        cv2.line(overlay, (int(w * i / 3), 0), (int(w * i / 3), h), (255, 255, 255), 1)
        cv2.line(overlay, (0, int(h * i / 3)), (w, int(h * i / 3)), (255, 255, 255), 1)
    cv2.imwrite(out_path, overlay)


def severity_dict(res, disease, image_name=None):
    sev_grade = int(res["severity_grade"])
    url = None
    if image_name:
        make_severity_overlay(res, os.path.join(UPLOAD_FOLDER, image_name))
        url = f"/uploads/{image_name}"
    return {
        "percentage": float(res["composite_severity_score"]),
        "label": severity_label_for(disease, sev_grade),
        "infected_pct": float(res["s_area_pct"]),
        "spot_count": int(res["spot_count"]),
        "zones_infected": int(round((res["s_zone_score"] / 100.0) * 9)),
        "leaf_size_category": res["leaf_size_category"],
        "is_young_leaf": bool(res["is_young_leaf"]),
        "severity_grade": sev_grade,
        "grade_label": res["grade_label"],
        "pathological_symptoms": res["pathological_symptoms"],
        "recommended_action": res["recommended_action"],
        "breakdown": {
            "area_score": float(res["s_area_pct"]),
            "color_score": float(res["s_color_score"]),
            "spread_score": float(res["s_spread_score"]),
            "zone_score": float(res["s_zone_score"]),
        },
        "severity_image_url": url,
    }


HEALTHY_SEVERITY = {
    "percentage": 0.0, "label": "Healthy", "infected_pct": 0.0, "spot_count": 0, "zones_infected": 0,
    "leaf_size_category": "", "is_young_leaf": False, "severity_grade": 0,
    "grade_label": "Grade 0: Healthy",
    "pathological_symptoms": "Asymptomatic foliage with uniform chlorophyll distribution and zero necrotic lesions.",
    "recommended_action": "Routine surveillance; maintain standard citrus nutritional and irrigation management.",
    "breakdown": ZERO_BREAKDOWN, "severity_image_url": None,
}


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
app = Flask(__name__)
app.json = NumpyJSONProvider(app)
CORS(app)


@app.route("/")
def home():
    return "Model API running"


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


def canopy_pipeline(img_bgr, leaf_candidates, unique_id, stem, gate_status):
    ih, iw = img_bgr.shape[:2]
    leaves = []
    rec_cache = {}                       # one RAG call per (disease, label), not per leaf

    for idx, cand in enumerate(leaf_candidates, start=1):
        x, y, bw, bh = cand["bbox"]
        px, py = int(bw * 0.05), int(bh * 0.05)
        x1, y1 = max(0, x - px), max(0, y - py)
        x2, y2 = min(iw, x + bw + px), min(ih, y + bh + py)
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        if GATE_ENABLED and CANOPY_GATE:
            try:
                decision, _p = check_citrus_decision(pil_crop)
                if decision == "reject":
                    continue
            except Exception as e:
                print("[gate] crop check failed:", e)

        disease, conf, _top3 = classify(pil_crop)
        is_fruit, _why = is_fruit_crop(crop, conf)
        if is_fruit:
            continue

        if disease.lower() == "healthy":
            sev = dict(HEALTHY_SEVERITY)
            rec = HEALTHY_REC
        else:
            crop_path = os.path.join(CROP_FOLDER, f"crop_{unique_id}_{idx}.jpg")
            cv2.imwrite(crop_path, crop)
            try:
                res = compute_comprehensive_severity(crop_path)
            finally:
                try:
                    os.remove(crop_path)
                except OSError:
                    pass
            sev = severity_dict(res, disease)
            key = (disease, sev["label"])
            if key not in rec_cache:
                rec_cache[key] = rec_fields(
                    get_recommendation(disease, conf, sev["percentage"], sev["label"]))
            rec = rec_cache[key]

        leaves.append({
            "leaf_id": idx, "bbox": [int(x), int(y), int(bw), int(bh)],
            "disease": disease, "confidence": conf,
            "severity": sev["percentage"], "severity_label": sev["label"],
            "severity_grade": sev["severity_grade"], "grade_label": sev["grade_label"],
            "breakdown": sev["breakdown"], "spot_count": sev["spot_count"],
            "zones_infected": sev["zones_infected"], "infected_pct": sev["infected_pct"],
            "leaf_size_category": sev["leaf_size_category"], "is_young_leaf": sev["is_young_leaf"],
            **rec,
        })

    if not leaves:
        return None

    total = len(leaves)
    healthy = sum(1 for l in leaves if l["disease"].lower() == "healthy")
    diseased = total - healthy
    rate = round(diseased / max(1, total) * 100.0, 1)
    if rate == 0.0:
        action = "HEALTHY CANOPY: Routine surveillance. No chemical spray required."
    elif rate <= 40.0:
        action = "LOCALIZED INFECTION: Targeted spot spraying on infected branch."
    else:
        action = "CRITICAL EPIDEMIC: Systemic orchard treatment needed across tree canopy."

    annotated = draw_canopy_diagnostic_overlay(img_bgr, leaves, rate)
    canopy_name = f"{stem}_{unique_id}_canopy_diagnostic.png"
    canopy_path = os.path.join(UPLOAD_FOLDER, canopy_name)
    cv2.imwrite(canopy_path, annotated)

    sick = [l for l in leaves if l["disease"].lower() != "healthy"]
    sure = [l for l in sick if l["confidence"] >= 40.0]
    if sure:
        primary = max(sure, key=lambda l: l["severity"] * (l["confidence"] / 100.0))
    elif sick:
        primary = max(sick, key=lambda l: l["confidence"])
    else:
        primary = leaves[0]

    return {
        "status": "ok", "gate_status": gate_status, "is_multi_leaf": True,
        "total_leaves_detected": total, "healthy_leaves_count": healthy,
        "diseased_leaves_count": diseased, "canopy_infection_rate": rate,
        "canopy_action_status": action,
        "annotated_canopy_url": f"/uploads/{canopy_name}",
        "annotated_canopy_b64": file_to_b64(canopy_path),
        "individual_leaves": leaves,
        # top-level fields describe the most affected leaf (same shape as single-leaf)
        "disease": primary["disease"], "confidence": primary["confidence"],
        "severity": primary["severity"], "severity_label": primary["severity_label"],
        "severity_grade": primary["severity_grade"], "grade_label": primary["grade_label"],
        "leaf_size_category": primary["leaf_size_category"], "is_young_leaf": primary["is_young_leaf"],
        "infected_pct": primary["infected_pct"], "spot_count": primary["spot_count"],
        "zones_infected": primary["zones_infected"], "breakdown": primary["breakdown"],
        "severity_image_url": f"/uploads/{canopy_name}",
        "severity_image_b64": None,
        "model_architecture": "Two-Stage: Foliar Localization + ViT-Base/16 (Multi-Leaf Canopy)",
        "top_predictions": [],
        **{k: primary[k] for k in ("treatment", "dose", "method", "timing", "interval", "warning",
                                   "cultural_advice", "monitoring", "urgency_note", "reason",
                                   "low_confidence_warning", "source")},
    }


@app.route("/predict", methods=["POST"])
def predict():
    prune_uploads()
    if "file" not in request.files:
        return reply("error", "No file uploaded")

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

    force = request.form.get("force") == "1"
    mode = request.form.get("mode", "auto")

    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        return reply("error", "Could not read this image. Please upload a JPG or PNG photo.")

    # Stage 0 - is there any plant tissue at all?
    is_leaf, foliage_pct = check_is_leaf(img_bgr)
    if not is_leaf:
        return reply("reject",
                     "This doesn't look like a citrus leaf. Please photograph a single citrus leaf.",
                     is_leaf=False, foliage_pct=foliage_pct)

    leaf_candidates = detect_foliar_instances(img_bgr, max_leaves=MAX_LEAVES)

    # ---------------- canopy pipeline (2+ leaves, or mode=multi_leaf) ----------------
    if leaf_candidates and (len(leaf_candidates) >= 2 or mode == "multi_leaf"):
        out = canopy_pipeline(img_bgr, leaf_candidates, unique_id, stem, "canopy")
        if out is None:
            return reply("fruit",
                         "This looks like citrus fruit. This system diagnoses leaves only. "
                         "Please photograph a citrus leaf.")
        return jsonify(out)

    # ---------------- single-leaf pipeline ----------------
    gate_status, citrus_p = "disabled", None
    if GATE_ENABLED:
        try:
            decision, citrus_p = check_citrus_decision(filepath)
        except Exception as e:
            print(f"[gate] could not read {stored_name}: {e}")
            return reply("error", "Could not read this image. Please upload a JPG or PNG photo.")
        citrus_p = round(citrus_p, 3)
        print(f"[gate] {stored_name}: P(citrus)={citrus_p} -> {decision}")
        if decision == "reject":
            return reply("reject",
                         "This doesn't look like a citrus leaf. Please photograph a single citrus leaf.",
                         citrus_probability=citrus_p)
        if decision == "retake" and not force:
            return reply("retake",
                         "I'm not sure this is a citrus leaf. Place one leaf on a plain surface "
                         "and fill the frame, then try again.",
                         citrus_probability=citrus_p)
        gate_status = "accepted" if decision == "accept" else "unverified"

    # Stage 1 - ViT diagnosis
    disease, confidence_pct, top_predictions = classify(Image.open(filepath))

    # Fruit / low-confidence guard
    is_fruit, why = is_fruit_crop(img_bgr, confidence_pct)
    gate_sure_leaf = (gate_status == "accepted" and citrus_p is not None
                      and citrus_p >= FRUIT_SKIP_GATE_P)
    if is_fruit and why.startswith("spherical") and gate_sure_leaf:
        print(f"[fruit] shape rule ignored: gate P(citrus)={citrus_p} >= {FRUIT_SKIP_GATE_P} ({why})")
        is_fruit = False
    if is_fruit:
        if why.startswith("spherical"):
            return reply("fruit",
                         "This looks like citrus fruit. This system diagnoses leaves only. "
                         "Please photograph a citrus leaf.",
                         citrus_probability=citrus_p, reason=why, vit_confidence=confidence_pct)
        if not force:
            return reply("retake",
                         "I couldn't recognise a clear leaf in this photo. Place one leaf on a "
                         "plain surface and fill the frame, then try again.",
                         citrus_probability=citrus_p, reason=why, vit_confidence=confidence_pct)
        gate_status = "unverified"

    # Stage 2 - severity
    if disease.lower() == "healthy":
        sev = dict(HEALTHY_SEVERITY)
    else:
        res = compute_comprehensive_severity(filepath)
        sev = severity_dict(res, disease, image_name=severity_name)

    # Stage 3 - RAG treatment
    rec = rec_fields(get_recommendation(disease, confidence_pct, sev["percentage"], sev["label"]))

    return jsonify({
        "status": "ok", "gate_status": gate_status, "is_multi_leaf": False,
        "citrus_probability": citrus_p,
        "disease": disease, "confidence": confidence_pct,
        "model_architecture": "Vision Transformer (ViT-Base/16)",
        "top_predictions": top_predictions,
        "severity": sev["percentage"], "severity_label": sev["label"],
        "severity_grade": sev["severity_grade"], "grade_label": sev["grade_label"],
        "pathological_symptoms": sev["pathological_symptoms"],
        "leaf_size_category": sev["leaf_size_category"], "is_young_leaf": sev["is_young_leaf"],
        "infected_pct": sev["infected_pct"], "spot_count": sev["spot_count"],
        "zones_infected": sev["zones_infected"], "breakdown": sev["breakdown"],
        "severity_image_url": sev["severity_image_url"],
        "severity_image_b64": (file_to_b64(os.path.join(UPLOAD_FOLDER, severity_name))
                               if sev["severity_image_url"] else None),
        **rec,
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
        "citrus_gate": "MobileNetV3-Large (v4)" if GATE_ENABLED else "disabled",
        "canopy_mode": "enabled",
        "canopy_gate": bool(GATE_ENABLED and CANOPY_GATE),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)