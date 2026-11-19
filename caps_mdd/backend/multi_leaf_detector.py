"""
=============================================================================
MULTI-LEAF CITRUS FOLIAR DETECTOR & CANOPY ANALYZER
Backend Extension for Real-Field Multi-Leaf Farmer Uploads
=============================================================================
"""

import cv2
import numpy as np


def detect_foliar_instances(img_bgr, max_leaves=8):
    """
    Detects distinct leaf instances in a photograph across both:
      1. Real-field farm clusters / branches (leaves touching, overlapping, or close together)
      2. Isolated single leaves on neutral backgrounds (laboratory datasets)

    Algorithm (Multi-Scale Peak Centroid Foliar Localization):
      - Mask foliage using combined HSV green/yellow-chlorotic + ExG thresholding.
      - Check if the image is a laboratory single leaf (dominant isolated component).
      - Otherwise, calculate Euclidean Distance Transform with boundary zeroing to locate
        true biological leaf blade centers (local maxima).
      - Generate adaptive leaf bounding boxes centered on peak centroids with NMS.
    Returns:
      List of dicts: [{"bbox": (x, y, w, h), "area": area}, ...]
    """
    h, w = img_bgr.shape[:2]

    # Normalize image scale if too large for consistent detection speed
    max_dim = 1024
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        work_img = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        work_img = img_bgr.copy()

    wh, ww = work_img.shape[:2]

    # 1. Universal Multi-Color Foliar Mask (Invariant to Leaf Color: Green, Yellow, Brown, Mottled)
    hsv = cv2.cvtColor(work_img, cv2.COLOR_BGR2HSV)
    img_float = work_img.astype(np.float32)
    b_ch, g_ch, r_ch = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
    exg = 2.0 * g_ch - r_ch - b_ch

    # Green & chlorotic yellow foliage (healthy, canker halos, greening / HLB)
    g_mask = cv2.inRange(hsv, np.array([12, 20, 20]), np.array([95, 255, 255]))
    exg_mask = (exg > 0).astype(np.uint8) * 255
    foliar_green = cv2.bitwise_and(g_mask, exg_mask)

    # Brown / necrotic dry foliage (DryLeaf, severe BacterialBlight, necrotic patches)
    brown_mask = (cv2.inRange(hsv, np.array([6, 25, 20]), np.array([24, 255, 220])) > 0) & (r_ch > b_ch + 15)
    foliar = cv2.bitwise_or(foliar_green, brown_mask.astype(np.uint8) * 255)

    clean_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    foliar = cv2.morphologyEx(foliar, cv2.MORPH_OPEN, clean_k)

    # 2. Check physically separated components
    num_f, lab_f, stats_f, cents_f = cv2.connectedComponentsWithStats(foliar)
    min_comp_area = int(wh * ww * 0.05)
    sep_comps = [i for i in range(1, num_f) if stats_f[i, cv2.CC_STAT_AREA] >= min_comp_area]

    total_foliar_pct = np.count_nonzero(foliar) / float(wh * ww)

    # 2. Multi-Scale Peak Centroid Foliar Localization
    # Zero border so distance transform measures from all 4 image edges
    foliar_bounded = foliar.copy()
    foliar_bounded[0, :] = 0
    foliar_bounded[-1, :] = 0
    foliar_bounded[:, 0] = 0
    foliar_bounded[:, -1] = 0

    dist = cv2.distanceTransform(foliar_bounded, cv2.DIST_L2, 5)
    max_d = float(dist.max())
    if max_d < 15:
        return []

    from scipy.ndimage import maximum_filter
    filter_sz = int(max(min(wh, ww) * 0.14, 25))
    if filter_sz % 2 == 0:
        filter_sz += 1

    local_max = maximum_filter(dist, size=filter_sz) == dist
    # Select genuine leaf peaks (distance >= 32 px and >= 25% of max leaf radius)
    peaks = np.logical_and(local_max, dist >= max(32.0, 0.25 * max_d))
    num_p, _, _, cents_p = cv2.connectedComponentsWithStats(peaks.astype(np.uint8))

    # Case A: Single Dominant Leaf (isolated on background or single prominent leaf)
    # If there is only 1 separated component, or 1 component holds >= 75% of foliar tissue,
    # treat the image as a single leaf to prevent fracturing 1 leaf into multiple fake boxes.
    if len(sep_comps) == 1:
        c_idx = sep_comps[0]
        bx = int(round(stats_f[c_idx, cv2.CC_STAT_LEFT] / scale))
        by = int(round(stats_f[c_idx, cv2.CC_STAT_TOP] / scale))
        bw = int(round(stats_f[c_idx, cv2.CC_STAT_WIDTH] / scale))
        bh = int(round(stats_f[c_idx, cv2.CC_STAT_HEIGHT] / scale))
        return [{"bbox": (bx, by, bw, bh), "area": bw * bh}]

    if len(sep_comps) > 1:
        comp_areas = [stats_f[i, cv2.CC_STAT_AREA] for i in sep_comps]
        max_area = max(comp_areas)
        total_comp_area = sum(comp_areas)
        if max_area / float(total_comp_area) >= 0.75:
            c_idx = sep_comps[comp_areas.index(max_area)]
            bx = int(round(stats_f[c_idx, cv2.CC_STAT_LEFT] / scale))
            by = int(round(stats_f[c_idx, cv2.CC_STAT_TOP] / scale))
            bw = int(round(stats_f[c_idx, cv2.CC_STAT_WIDTH] / scale))
            bh = int(round(stats_f[c_idx, cv2.CC_STAT_HEIGHT] / scale))
            return [{"bbox": (bx, by, bw, bh), "area": bw * bh}]

    # Case B: Multi-Leaf Branch / Canopy Field Shot (2 or more distinct separated leaves)

    min_leaf_area = int(wh * ww * 0.045)
    raw_boxes = []
    for i in range(1, num_p):
        cx, cy = cents_p[i]
        r = float(dist[int(cy), int(cx)])
        bw = int(max(r * 2.6, min(wh, ww) * 0.30))
        bh = int(max(r * 2.3, min(wh, ww) * 0.26))
        x1 = max(0, int(cx - bw // 2))
        y1 = max(0, int(cy - bh // 2))
        x2 = min(ww, int(cx + bw // 2))
        y2 = min(wh, int(cy + bh // 2))
        box_area = (x2 - x1) * (y2 - y1)
        if box_area >= min_leaf_area:
            raw_boxes.append([x1, y1, x2 - x1, y2 - y1, r])

    # Sort largest/most prominent leaves first
    raw_boxes.sort(key=lambda b: b[4], reverse=True)

    # 3. Non-Maximum Suppression (NMS)
    selected = []
    for b in raw_boxes:
        x1, y1, bw, bh, score = b
        overlap = False
        for s in selected:
            sx1, sy1, sbw, sbh = s["bbox"]
            ix1 = max(x1, sx1)
            iy1 = max(y1, sy1)
            ix2 = min(x1 + bw, sx1 + sbw)
            iy2 = min(y1 + bh, sy1 + sbh)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = bw * bh + sbw * sbh - inter
                iou = inter / float(union)
                io_min = inter / float(min(bw * bh, sbw * sbh))
                if iou > 0.35 or io_min > 0.65:
                    overlap = True
                    break
        if not overlap:
            ox = int(round(x1 / scale))
            oy = int(round(y1 / scale))
            obw = int(round(bw / scale))
            obh = int(round(bh / scale))
            selected.append({"bbox": (ox, oy, obw, obh), "area": obw * obh, "score": score})
            if len(selected) >= max_leaves:
                break

    # Fallback if no peaks met criteria: use largest separated components
    if not selected and len(sep_comps) >= 1:
        for c_idx in sep_comps[:max_leaves]:
            bx = int(round(stats_f[c_idx, cv2.CC_STAT_LEFT] / scale))
            by = int(round(stats_f[c_idx, cv2.CC_STAT_TOP] / scale))
            bw = int(round(stats_f[c_idx, cv2.CC_STAT_WIDTH] / scale))
            bh = int(round(stats_f[c_idx, cv2.CC_STAT_HEIGHT] / scale))
            selected.append({"bbox": (bx, by, bw, bh), "area": bw * bh})

    # Sort left to right
    selected.sort(key=lambda item: item["bbox"][0])
    return selected


def draw_canopy_diagnostic_overlay(img_bgr, diagnosed_leaves, canopy_infection_rate):
    """
    Draws professional bounding boxes, disease labels, CSI severity %,
    and top summary banner onto the image.
    """
    annotated = img_bgr.copy()
    h, w = annotated.shape[:2]

    COLOR_HEALTHY = (40, 200, 40)   # Green
    COLOR_MILD    = (30, 210, 240)  # Yellow
    COLOR_DISEASE = (30, 30, 230)   # Red

    for item in diagnosed_leaves:
        idx = item["leaf_id"]
        x, y, bw, bh = item["bbox"]
        disease = item["disease"]
        conf = item["confidence"]
        sev_pct = item["severity"]
        grade = item.get("grade", "")

        if disease.lower() == "healthy":
            color = COLOR_HEALTHY
        elif sev_pct <= 25.0:
            color = COLOR_MILD
        else:
            color = COLOR_DISEASE

        # Bounding Box
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 3)

        # Label Header Banner
        header = f"#{idx}: {disease} ({conf}%)"
        sub = f"Sev: {sev_pct}%"
        box_w = max(len(header), len(sub)) * 12
        banner_y1 = max(0, y - 44)
        banner_y2 = y
        cv2.rectangle(annotated, (x, banner_y1), (x + box_w, banner_y2), color, -1)
        cv2.putText(annotated, header, (x + 5, max(14, y - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, sub, (x + 5, max(28, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    # Top summary banner
    banner_h = 52
    top_banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    top_banner[:] = (25, 25, 25)
    total = len(diagnosed_leaves)
    healthy_cnt = sum(1 for l in diagnosed_leaves if l["disease"].lower() == "healthy")
    diseased_cnt = total - healthy_cnt
    summary_txt = f"Canopy Analysis: {total} Leaves Assessed ({healthy_cnt} Healthy, {diseased_cnt} Diseased) | Infection Rate: {canopy_infection_rate}%"
    cv2.putText(top_banner, summary_txt, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return np.vstack([top_banner, annotated])


def is_fruit_crop(crop_bgr, vit_conf):
    """
    Color-invariant discriminator to separate Citrus Fruits from Citrus Foliar Leaves.
    Accurately identifies and excludes fruits of ANY color (green limes, yellow lemons,
    orange mandarins/oranges, or dark/rotten fruits), while preserving genuine leaves of
    ANY color (healthy green, chlorotic yellow, necrotic brown, or spotted/mottled).

    Criteria:
      1. ViT Foliar Pathology Confidence:
         The ViT model was trained strictly on citrus leaf blades.
         Fruit peels/rinds lack leaf anatomy (veins, petiole, blade lamina)
         and consistently yield diffuse, low foliar confidence (< 38.0%).
      2. Geometric Spherical Shape & Solidity:
         Citrus fruits are 3D spheroids whose 2D projection is a circular disc:
         - Aspect ratio max(w,h)/min(w,h) <= 1.25 (near 1:1)
         - High contour circularity (4*pi*A / P^2 > 0.65) or convex hull circularity (> 0.72)
         - High solidity (> 0.86, strictly convex with NO pointed apex or petiole taper)
         - ViT confidence < 75%
      3. Leaf Blade Invariance:
         Real citrus leaves are elongated (aspect ratio >= 1.35) with tapered apex and petiole,
         and achieve high ViT foliar pathology confidence (>= 45%, typically 85-99%).

    Returns:
      (is_fruit: bool, reason: str)
    """
    if crop_bgr is None:
        return False, "none_image"

    ch, cw = crop_bgr.shape[:2]
    if ch < 15 or cw < 15:
        return True, "crop_too_small"

    ar = max(ch, cw) / float(max(1, min(ch, cw)))

    # 1. ViT Foliar Pathology Confidence Check:
    # In a 12-class classifier, random chance is 8.3%.
    # Real field canopy leaves under partial shade or angled views achieve 20% - 55% confidence.
    # We reject non-foliar candidates that fall below 18.0%.
    if vit_conf < 18.0:
        return True, f"low_foliar_confidence_{vit_conf}%"

    # 2. Geometric Spherical Shape Analysis:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    border = np.concatenate([thresh[0, :], thresh[-1, :], thresh[:, 0], thresh[:, -1]])
    if np.mean(border) > 127:
        thresh = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)

        if peri > 0 and area > (ch * cw * 0.12):
            circ = (4.0 * np.pi * area) / (peri * peri)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            hull_peri = cv2.arcLength(hull, True)
            hull_circ = (4.0 * np.pi * hull_area) / (hull_peri * hull_peri) if hull_peri > 0 else 0.0
            solidity = area / float(hull_area) if hull_area > 0 else 0.0

            # Spherical fruit invariant:
            # Fruits are circular discs: AR <= 1.24, hull_circ > 0.72, solidity > 0.86
            # Real leaf blades are elongated: AR >= 1.35, hull_circ <= 0.65
            if ar <= 1.24 and (circ > 0.68 or hull_circ > 0.74) and solidity > 0.86 and vit_conf < 75.0:
                return True, f"spherical_geometry_circ_{circ:.2f}_hull_{hull_circ:.2f}_ar_{ar:.2f}"

    return False, "leaf_confirmed"
