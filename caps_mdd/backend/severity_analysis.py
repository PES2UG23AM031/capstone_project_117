"""
=============================================================================
CITRUS LEAF SEVERITY ANALYSIS & CONTINUOUS CSI REGRESSION FRAMEWORK
Capstone Project - Advanced Diagnostic & Severity Estimation Module
=============================================================================
100% UNCALIBRATED DATA-DRIVEN FORMULATION (ORDINARY LEAST SQUARES REGRESSION):
  - Trained on 120 CVAT expert-annotated citrus leaf images across 12 classes
  - CIELAB Chroma Suppression segmentation (L* > 155, Chroma < 24)
  - Scale-invariant physical features:
      A: Lesion Area Coverage (0 to 100%)
      C: Color Deviation from Chlorophyll Baseline (0 to 100%)
      S: Spatial Centroid Dispersion / Equivalent Radius (0 to 100%)
      Z: 3x3 Grid Anatomical Zone Penetration (0 to 100%)
  - Learned Closed-Form Composite Severity Index (weights sum to 1.0):
      CSI = 0.546932 * A + 0.072753 * C + 0.198905 * S + 0.181410 * Z
  - Continuous Grade Mapping:
      G_pred = clip(0.040064 + 0.040617 * CSI, 0, 3)
  - Pure Uncalibrated Discrete Rounding:
      Grade = clip(round(G_pred), 0, 3)
      Grade 0: Healthy (< 0.5)
      Grade 1: Mild (0.5 to 1.5)
      Grade 2: Moderate (1.5 to 2.5)
      Grade 3: Severe (>= 2.5)
=============================================================================
"""

import os
import cv2
import numpy as np

# Literature & Phytopathology Grounded Severity Tiers
LITERATURE_SEVERITY_LEVELS = {
    0: {
        "label": "Grade 0: Healthy",
        "grade_name": "Healthy",
        "range": (0.0, 11.32),
        "continuous_range": (0.0, 0.5),
        "symptoms": "Healthy leaf tissue; zero active infection; intact chlorophyll distribution.",
        "action": "Routine surveillance; preventative organic spray during monsoon or high humidity."
    },
    1: {
        "label": "Grade 1: Mild",
        "grade_name": "Mild",
        "range": (11.32, 35.94),
        "continuous_range": (0.5, 1.5),
        "symptoms": "Early localized micro-lesions (<5% leaf blade coverage), discrete pustules, or faint chlorotic halos.",
        "action": "Targeted organic or copper fungicide/bactericide application; regular monitoring."
    },
    2: {
        "label": "Grade 2: Moderate",
        "grade_name": "Moderate",
        "range": (35.94, 60.56),
        "continuous_range": (1.5, 2.5),
        "symptoms": "Spreading infection foci across multiple sectors (5%-20% area coverage), chlorotic halos expanding.",
        "action": "Curative therapeutic spray schedule; canopy sanitation; prune severely affected twigs."
    },
    3: {
        "label": "Grade 3: Severe",
        "grade_name": "Severe",
        "range": (60.56, 100.0),
        "continuous_range": (2.5, 3.0),
        "symptoms": "Extensive coalescing necrosis (>20% area coverage), severe leaf distortion, photosynthetic disruption.",
        "action": "Aggressive disease suppression; quarantine affected zone; defoliation containment."
    },
    # Backward compatibility
    4: {
        "label": "Grade 3: Severe",
        "grade_name": "Severe",
        "range": (60.56, 100.0),
        "continuous_range": (2.5, 3.0),
        "symptoms": "Extensive coalescing necrosis (>20% area coverage), severe leaf distortion, photosynthetic disruption.",
        "action": "Aggressive disease suppression; quarantine affected zone; defoliation containment."
    },
    5: {
        "label": "Grade 3: Severe",
        "grade_name": "Severe",
        "range": (60.56, 100.0),
        "continuous_range": (2.5, 3.0),
        "symptoms": "Extensive coalescing necrosis (>20% area coverage), severe leaf distortion, photosynthetic disruption.",
        "action": "Aggressive disease suppression; quarantine affected zone; defoliation containment."
    }
}

HEALTHY_CITRUS_LAB = np.array([125.0, 110.0, 142.0], dtype=np.float32)


# =============================================================================
# 1. CIELAB CHROMA SUPPRESSION LEAF SEGMENTATION
# =============================================================================

def segment_leaf_clean(img_bgr):
    """
    Segments citrus leaf blade from white/light paper background
    using CIELAB chroma suppression and HSV saturation.
    Completely eliminates paper wrinkles, paper textures, and cast shadows.
    """
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    L = lab[:, :, 0]
    A_ch = lab[:, :, 1]
    B_ch = lab[:, :, 2]
    S_ch = hsv[:, :, 1]

    chroma = np.sqrt((A_ch.astype(float) - 128.0)**2 + (B_ch.astype(float) - 128.0)**2)
    # Background paper & cast shadows on paper have near-zero chroma & low saturation
    is_paper_bright = (L > 140) & (chroma < 22)
    is_paper_shadow = (chroma < 18) & (S_ch < 35)
    is_bg = is_paper_bright | is_paper_shadow

    init_mask = ((~is_bg) * 255).astype(np.uint8)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    init_mask = cv2.morphologyEx(init_mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    init_mask = cv2.morphologyEx(init_mask, cv2.MORPH_OPEN, k_open, iterations=2)

    cnts, _ = cv2.findContours(init_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf_mask = np.zeros((h, w), dtype=np.uint8)
    if cnts:
        valid_cnts = [c for c in cnts if cv2.contourArea(c) > (h * w * 0.04)]
        if valid_cnts:
            best_c = max(valid_cnts, key=cv2.contourArea)
            cv2.drawContours(leaf_mask, [best_c], -1, 255, -1)
            leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    return leaf_mask


# =============================================================================
# 2. PATHOLOGICAL LESION EXTRACTION (SHADOW-IMMUNE)
# =============================================================================

def detect_lesion_clean(img_bgr, leaf_mask):
    """
    Detects true pathological citrus lesions strictly inside leaf_mask:
    - Sooty mould / true black rot
    - Brown / corky pustules (Canker, Melanose, Black Spot)
    - Necrotic tip blighting
    - Chlorosis (Greening HLB, Deficiency)
    - Leaf Miner trails
    Uses adaptive median leaf baseline contrast to remain immune to illumination,
    leaf shadows, and background lighting.
    """
    if np.count_nonzero(leaf_mask) == 0:
        return np.zeros_like(leaf_mask)

    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    L = lab[:, :, 0]
    A_ch = lab[:, :, 1]
    B_ch = lab[:, :, 2]
    H_ch = hsv[:, :, 0]
    S_ch = hsv[:, :, 1]
    V_ch = hsv[:, :, 2]

    img_f = img_bgr.astype(np.float32)
    b, g, r = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
    exg = 2.0 * g - r - b

    in_leaf = leaf_mask > 0
    med_v = float(np.median(V_ch[in_leaf]))
    med_l = float(np.median(L[in_leaf]))
    med_a = float(np.median(A_ch[in_leaf]))
    med_exg = float(np.median(exg[in_leaf]))

    # 1. Pitch black rot / Sooty mould (dark AND not pure green chlorophyll)
    is_soot = (V_ch < 45) & (L < 45) & (A_ch >= 115)

    # 2. Dark necrosis / rot (significantly darker than median leaf with elevated A_ch)
    is_dark_rot = (V_ch < min(55.0, med_v * 0.70)) & (A_ch >= med_a + 4)

    # 3. Brown / corky pustules (Melanose, Canker, Black Spot)
    is_brown_corky = (V_ch < med_v * 0.85) & (H_ch <= 32) & (A_ch >= med_a + 4) & (r > b + 15)

    # 4. Necrotic tip blighting (dead curled leaf tip)
    is_tip_blight = (A_ch >= 126) & (r > g) & (H_ch <= 30)

    # 5. Chlorosis (Greening HLB, Deficiency)
    is_chlorosis = (H_ch >= 16) & (H_ch <= 34) & (B_ch >= 162) & (A_ch >= med_a + 5) & (exg < max(25.0, med_exg * 0.5))

    # 6. Leaf Miner trails (silvery/translucent tunnels)
    is_miner = (S_ch < 45) & (V_ch > 100) & (A_ch >= med_a + 5) & (L >= 115) & (exg < 15)

    pathology = (is_soot | is_dark_rot | is_brown_corky | is_tip_blight | is_chlorosis | is_miner) & in_leaf

    # Pure healthy green leaf tissue protection (even in shadows)
    is_healthy_green = (H_ch >= 30) & (H_ch <= 90) & (A_ch <= max(112.0, med_a + 2)) & (g >= r + 4) & (exg > 20)

    lesions = pathology & (~is_healthy_green)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_lesions = cv2.morphologyEx((lesions * 255).astype(np.uint8), cv2.MORPH_OPEN, k)
    return clean_lesions


# =============================================================================
# 3. COMPREHENSIVE SEVERITY COMPUTATION
# =============================================================================

def compute_comprehensive_severity(image_path):
    """
    Computes leaf disease severity using empirical OLS regression weights:
    CSI = 0.546932 * A + 0.072753 * C + 0.198905 * S + 0.181410 * Z
    Predicted Grade = clip(round(0.040064 + 0.040617 * CSI), 0, 3)
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    h, w = img_bgr.shape[:2]
    total_pixels = h * w

    # 1. Segment leaf cleanly
    leaf_mask = segment_leaf_clean(img_bgr)
    leaf_pixels = int(np.count_nonzero(leaf_mask))

    if leaf_pixels == 0:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        leaf_mask = (gray < 235).astype(np.uint8) * 255
        leaf_pixels = int(np.count_nonzero(leaf_mask))

    leaf_coverage_ratio = leaf_pixels / max(1, total_pixels)
    if leaf_coverage_ratio < 0.25:
        leaf_size_category = "Small / Distant Leaf"
    elif leaf_coverage_ratio <= 0.60:
        leaf_size_category = "Medium / Standard Leaf"
    else:
        leaf_size_category = "Large / Close-up Leaf"

    # Young leaf check
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    leaf_lab = img_lab[leaf_mask > 0]
    mean_b = float(np.mean(leaf_lab[:, 2])) if len(leaf_lab) > 0 else 140.0
    mean_l = float(np.mean(leaf_lab[:, 0])) if len(leaf_lab) > 0 else 120.0
    is_young_leaf = bool(mean_b > 150.0 and mean_l > 135.0)

    # 2. Detect lesions
    lesion_mask = detect_lesion_clean(img_bgr, leaf_mask)
    lesion_pixels = int(np.count_nonzero(lesion_mask))

    # Feature A: Lesion Area Coverage (0-100)
    feat_A = min(100.0, max(0.0, (lesion_pixels / float(max(1, leaf_pixels))) * 100.0))

    # Feature C: CIELAB Color Deviation from Healthy Green (0-100)
    if lesion_pixels > 0:
        lesion_lab = img_lab[lesion_mask > 0]
        delta_e = np.linalg.norm(lesion_lab - HEALTHY_CITRUS_LAB, axis=1)
        mean_de = float(np.mean(delta_e))
        feat_C = min(100.0, max(0.0, (mean_de / 85.0) * 100.0))
    else:
        feat_C = 5.0

    # Feature S: Spatial Centroid Dispersion (0-100)
    if lesion_pixels > 10:
        ys, xs = np.nonzero(lesion_mask)
        std_y, std_x = np.std(ys), np.std(xs)
        spatial_sigma = float(np.sqrt(std_y**2 + std_x**2))
        r_eq = float(np.sqrt(leaf_pixels / np.pi))
        feat_S = min(100.0, max(0.0, (spatial_sigma / (r_eq * 0.70)) * 100.0))
    else:
        feat_S = 0.0

    # Feature Z: 3x3 Anatomical Zone Penetration (0-100)
    ys_leaf, xs_leaf = np.nonzero(leaf_mask)
    if len(ys_leaf) > 0 and len(xs_leaf) > 0:
        y_min, y_max = int(np.min(ys_leaf)), int(np.max(ys_leaf))
        x_min, x_max = int(np.min(xs_leaf)), int(np.max(xs_leaf))
        box_h = max(1, (y_max - y_min) // 3)
        box_w = max(1, (x_max - x_min) // 3)
        zones_affected = 0
        for r in range(3):
            for c in range(3):
                zy1 = y_min + r * box_h
                zy2 = y_min + (r + 1) * box_h if r < 2 else y_max
                zx1 = x_min + c * box_w
                zx2 = x_min + (c + 1) * box_w if c < 2 else x_max
                zone_lesion = np.count_nonzero(lesion_mask[zy1:zy2, zx1:zx2])
                zone_leaf = np.count_nonzero(leaf_mask[zy1:zy2, zx1:zx2])
                if zone_leaf > 50 and (zone_lesion / float(zone_leaf)) > 0.015:
                    zones_affected += 1
        feat_Z = (zones_affected / 9.0) * 100.0
    else:
        zones_affected = 0
        feat_Z = 0.0

    # Spot count
    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_cnt_area = max(4, int(leaf_pixels * 0.0001))
    valid_cnts = [c for c in contours if cv2.contourArea(c) >= min_cnt_area]
    spot_count = len(valid_cnts)

    # -------------------------------------------------------------------------
    # FINAL CLOSED-FORM CSI REGRESSION FORMULA (Weights sum to 1.0)
    # -------------------------------------------------------------------------
    w_A = 0.546932
    w_C = 0.072753
    w_S = 0.198905
    w_Z = 0.181410
    csi = (w_A * feat_A) + (w_C * feat_C) + (w_S * feat_S) + (w_Z * feat_Z)
    csi = float(np.clip(csi, 0.0, 100.0))
    csi_score = round(csi, 2)

    # Continuous Grade: G_pred = 0.040064 + 0.040617 * CSI
    continuous_grade = 0.040064 + 0.040617 * csi
    continuous_grade = float(np.clip(continuous_grade, 0.0, 3.0))

    # Standard Discrete Rounding (Strictly Uncalibrated)
    assigned_grade = int(np.clip(np.round(continuous_grade), 0, 3))
    grade_info = LITERATURE_SEVERITY_LEVELS[assigned_grade]

    return {
        "image_path": image_path,
        "leaf_size_category": leaf_size_category,
        "is_young_leaf": is_young_leaf,
        "s_area_pct": round(float(feat_A), 2),
        "s_color_score": round(float(feat_C), 2),
        "s_spread_score": round(float(feat_S), 2),
        "s_zone_score": round(float(feat_Z), 2),
        "composite_severity_score": csi_score,
        "continuous_grade": round(continuous_grade, 3),
        "severity_grade": assigned_grade,
        "grade_label": grade_info["label"],
        "pathological_symptoms": grade_info["symptoms"],
        "recommended_action": grade_info["action"],
        "spot_count": spot_count,
        "zones_infected": zones_affected,
        "leaf_pixels": leaf_pixels,
        "lesion_pixels": lesion_pixels,
        "img_bgr": img_bgr,
        "leaf_mask": leaf_mask,
        "lesion_mask": lesion_mask
    }
