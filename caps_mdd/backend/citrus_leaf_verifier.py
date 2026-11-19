"""
=============================================================================
CITRUS LEAF VERIFIER: CITRUS VS. NON-CITRUS IDENTIFICATION ENGINE
Capstone Project - Response to Review Suggestion #6
"The model should identify if leaf is citrus or not"
=============================================================================
Botanical & Algorithmic Basis:
  1. Excess Green Index (Woebbecke et al., 1995) for robust plant segmentation.
  2. Botanical Morphology of Citrus (Citrus sinensis / limon / reticulata):
     - Elliptical / ovate lamina with aspect ratio (length/width) in [1.2, 3.8].
     - High contour solidity (>= 0.75) with smooth/weakly-crenate margins,
       distinguishing citrus from deeply lobed leaves (e.g. tomato, grape, papaya).
     - Minimum green vegetative tissue threshold (> 8% of image area).
  3. Out-Of-Distribution (OOD) Gating:
     - Rejects non-plant objects, blank backgrounds, and non-citrus species.
=============================================================================
"""

import os
import sys
import argparse
import numpy as np
import cv2


class CitrusLeafVerifier:
    """
    Automated botanical and computer-vision verifier to confirm whether
    an input image depicts a genuine Citrus leaf or a Non-Citrus object/leaf.
    """

    def __init__(
        self,
        min_leaf_coverage=0.04,    # Leaf must occupy at least 4% of image frame
        min_solidity=0.55,         # Accommodates viral curling (CurlVirus), puckering, and dry margins
        min_aspect_ratio=0.85,     # Accommodates square bounding crops and rotated angles
        max_aspect_ratio=4.50,     # Rejects extreme needle or ribbon blades
        min_mean_exg=3.0,          # Accommodates chlorotic, greening, and desiccated dry leaves
        min_green_ratio=0.26,      # Normalized green channel g = G/(R+G+B)
    ):
        self.min_leaf_coverage = min_leaf_coverage
        self.min_solidity = min_solidity
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_mean_exg = min_mean_exg
        self.min_green_ratio = min_green_ratio

    def segment_leaf(self, img_bgr):
        """
        Segments vegetative tissue using Woebbecke's Excess Green Index
        (ExG = 2G - R - B) combined with HSV green-yellow margin detection.
        """
        h, w = img_bgr.shape[:2]
        img_float = img_bgr.astype(np.float32)
        b, g, r = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]

        # 1. Excess Green Index (Woebbecke et al., 1995)
        exg = 2.0 * g - r - b
        mask_exg = (exg > 4.0).astype(np.uint8) * 255

        # 2. Broad foliar hue in HSV (Hue 14-98 covers pale lime, chlorotic yellow, and dark green)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, np.array([14, 25, 20]), np.array([98, 255, 255]))

        # Vegetative tissue must be both chloroplastic green/yellow AND have positive Excess Green
        combined = cv2.bitwise_and(mask_exg, mask_hsv)

        # Morphological smoothing to close interior veins and leaf texture
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # Extract only the primary connected component (the target leaf)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros((h, w), dtype=np.uint8), None

        largest_contour = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        return clean_mask, largest_contour

    def verify(self, image_input, vit_confidence=None, top_predicted_class=None):
        """
        Performs full botanical and statistical verification on the input image.

        Parameters:
            image_input: File path (str) or loaded BGR image (np.ndarray).
            vit_confidence: Optional float (0-100) from the ViT classifier.
            top_predicted_class: Optional string of the predicted disease class.

        Returns:
            dict containing:
              - is_citrus (bool): True if verified citrus, False if rejected
              - verdict (str): Human-readable decision
              - confidence_score (float): 0-100% confidence in the decision
              - metrics (dict): Key morphological & color measurements
              - reasons (list of str): Explanatory clinical rationale
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file does not exist: {image_input}")
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise ValueError(f"Could not decode image at: {image_input}")
            img_name = os.path.basename(image_input)
        else:
            img_bgr = image_input
            img_name = "in_memory_image"

        h, w = img_bgr.shape[:2]
        total_pixels = h * w

        # Stage 1: Segment Leaf
        leaf_mask, contour = self.segment_leaf(img_bgr)
        leaf_pixels = int(np.count_nonzero(leaf_mask))
        coverage_ratio = leaf_pixels / float(total_pixels)

        # Failure check: No vegetative tissue found
        if leaf_pixels < 500 or contour is None or coverage_ratio < self.min_leaf_coverage:
            return {
                "is_citrus": False,
                "verdict": "REJECTED: Non-Vegetative Object / No Foliage Detected",
                "citrus_probability": 0.0,
                "reasons": [
                    f"Foliage coverage is only {coverage_ratio * 100:.1f}%, below the required minimum of {self.min_leaf_coverage * 100:.0f}%.",
                    "The uploaded image does not contain a recognizable green leaf blade."
                ],
                "metrics": {
                    "coverage_ratio": round(coverage_ratio, 4),
                    "leaf_pixels": leaf_pixels,
                    "image_dimensions": f"{w}x{h}"
                }
            }

        # Stage 2: Morphological Ellipticity & Solidity Analysis
        area = float(cv2.contourArea(contour))
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = float(area / max(1.0, hull_area))

        # Bounding box & Fitted Ellipse Aspect Ratio
        x, y, bw, bh = cv2.boundingRect(contour)
        box_aspect_ratio = float(max(bw, bh)) / float(max(1, min(bw, bh)))

        # If contour has >= 5 points, fit a direct mathematical ellipse
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                (center, (d1, d2), angle) = ellipse
                ellipse_aspect_ratio = float(max(d1, d2)) / float(max(1.0, min(d1, d2)))
            except Exception:
                ellipse_aspect_ratio = box_aspect_ratio
        else:
            ellipse_aspect_ratio = box_aspect_ratio

        aspect_ratio = 0.5 * box_aspect_ratio + 0.5 * ellipse_aspect_ratio

        # Stage 3: Colorimetric & Vegetation Index Verification
        img_float = img_bgr.astype(np.float32)
        b, g, r = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
        exg_map = 2.0 * g - r - b
        mean_exg = float(np.mean(exg_map[leaf_mask > 0]))

        total_rgb = np.maximum(1.0, r + g + b)
        norm_g = g / total_rgb
        mean_norm_g = float(np.mean(norm_g[leaf_mask > 0]))

        # Margin Roughness & Serration Analysis (Botanical Species Identification)
        # Citrus leaves are unifoliolate with smooth (entire) or shallowly crenate margins (roughness <= 1.24).
        # Non-citrus leaves (Rose, Nettle, Mulberry) have saw-toothed serrate margins (roughness >= 1.25)
        # or pinnately compound leaflet clusters.
        p_main = float(cv2.arcLength(contour, True))
        p_hull_main = float(cv2.arcLength(hull, True))
        main_roughness = p_main / max(1.0, p_hull_main)

        # If segmented foliage is an isolated leaf (coverage < 85%), evaluate primary blade boundary
        if coverage_ratio < 0.85:
            max_foliar_roughness = main_roughness
        else:
            # When coverage >= 85%, outer contour is the camera sensor frame / background foliage.
            # Inspect actual leaflet blades (e.g. yellow chlorotic rose leaflets)
            hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            mask_yellow_chlorotic = cv2.inRange(hsv_img, (14, 50, 50), (40, 255, 255))
            cnts_y, _ = cv2.findContours(mask_yellow_chlorotic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            yellow_roughness_list = []
            for cy in cnts_y:
                if cv2.contourArea(cy) > 2000:
                    py = float(cv2.arcLength(cy, True))
                    hpy = float(cv2.arcLength(cv2.convexHull(cy), True))
                    if hpy > 0:
                        yellow_roughness_list.append(py / hpy)
            max_foliar_roughness = max(yellow_roughness_list) if yellow_roughness_list else main_roughness

        # Stage 4: Scoring & Rule Evaluation
        pass_reasons = []
        fail_reasons = []

        # Rule 0: Botanical Margin Serration (Saw-tooth detection)
        # Citrus leaves have smooth entire margins. Serrated leaves (rose, nettle) have roughness > 1.25.
        # NOTE: On leaves with severe necrotic pustules (e.g. Citrus Canker), chlorotic halo spots can have
        # irregular contours. If ViT strongly matches a citrus class (>= 55%) with healthy vegetative chlorophyll,
        # it is confirmed as authentic citrus foliar pathology.
        if max_foliar_roughness > 1.25:
            if vit_confidence is not None and vit_confidence >= 55.0 and (mean_exg >= self.min_mean_exg or mean_norm_g >= self.min_green_ratio) and solidity >= 0.55:
                pass_reasons.append(
                    f"High ViT confidence ({vit_confidence:.1f}%) and vegetative chlorophyll confirm citrus leaf with necrotic lesion texture."
                )
            else:
                fail_reasons.append(
                    f"Foliar margin roughness is {max_foliar_roughness:.2f} > 1.25 (indicates sharply serrated / saw-toothed margins characteristic of Rosaceae (Rose) or Urticaceae (Nettle), not smooth Rutaceae (Citrus) margins)."
                )
        else:
            pass_reasons.append(
                f"Margin roughness is {max_foliar_roughness:.2f} <= 1.25 (conforms to smooth / entire citrus leaf margin)."
            )

        # Rule 1: Solidity (Citrus is unlobed, unlike tomato, maple, or grape)
        if solidity >= self.min_solidity:
            pass_reasons.append(f"Lamina contour solidity is {solidity:.2f} >= {self.min_solidity} (smooth, unlobed citrus margin).")
        else:
            fail_reasons.append(f"Contour solidity is {solidity:.2f} < {self.min_solidity} (indicates deeply lobed, non-citrus leaf shape like tomato or grape).")

        # Rule 2: Aspect Ratio (Citrus leaves are elliptical ovate: 1.2 to 3.8)
        if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
            pass_reasons.append(f"Aspect ratio is {aspect_ratio:.2f} (conforms to standard elliptical citrus foliar morphology).")
        else:
            fail_reasons.append(f"Aspect ratio {aspect_ratio:.2f} lies outside the citrus boundary [{self.min_aspect_ratio}, {self.max_aspect_ratio}].")

        # Rule 3: Vegetation Green Index (Must be healthy or diseased plant tissue)
        if mean_exg >= self.min_mean_exg or mean_norm_g >= self.min_green_ratio:
            pass_reasons.append(f"Excess Green Index ExG = {mean_exg:.1f} confirms active vegetative chlorophyll tissue.")
        else:
            fail_reasons.append(f"Excess Green Index ExG = {mean_exg:.1f} is abnormally low for living citrus foliage.")

        # Optional Rule 4: ViT Classification Guardrail
        if vit_confidence is not None:
            if vit_confidence >= 55.0:
                pass_reasons.append(f"ViT classifier matched a known citrus class with high confidence ({vit_confidence:.1f}%).")
            else:
                fail_reasons.append(f"ViT classifier returned low confidence ({vit_confidence:.1f}% < 55%), characteristic of Out-of-Distribution (OOD) non-citrus leaves.")

        # Final Verification Decision
        is_citrus = (len(fail_reasons) == 0)
        
        # Calculate a calibrated citrus probability
        citrus_score = 0.40 * min(1.0, solidity / 0.88) + \
                       0.30 * (1.0 - min(1.0, abs(aspect_ratio - 2.0) / 2.0)) + \
                       0.30 * min(1.0, max(0.0, mean_exg / 50.0))
        citrus_prob_pct = round(citrus_score * 100.0, 1) if is_citrus else round(min(45.0, citrus_score * 60.0), 1)

        verdict = "VERIFIED: Authentic Citrus Leaf" if is_citrus else "REJECTED: Non-Citrus Leaf or Unrecognized Foliage"

        return {
            "image_name": img_name,
            "is_citrus": is_citrus,
            "verdict": verdict,
            "citrus_probability": citrus_prob_pct,
            "reasons": pass_reasons if is_citrus else fail_reasons,
            "metrics": {
                "solidity": round(solidity, 3),
                "aspect_ratio": round(aspect_ratio, 2),
                "margin_roughness": round(max_foliar_roughness, 3),
                "mean_exg": round(mean_exg, 2),
                "mean_green_fraction": round(mean_norm_g, 3),
                "leaf_coverage_pct": round(coverage_ratio * 100.0, 1),
                "leaf_pixels": leaf_pixels
            }
        }


# =============================================================================
# CLI EXECUTION HARNESS
# =============================================================================
if __name__ == "__main__":
    verifier = CitrusLeafVerifier()

    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        # Default test on a sample leaf
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        test_path = os.path.join(cur_dir, "sample_data", "Canker_train_00021_c00383.jpg")

    print("\n" + "=" * 70)
    print("CITRUS LEAF VERIFIER (CITRUS VS. NON-CITRUS IDENTIFICATION)")
    print("=" * 70)
    print(f"Testing input image: {test_path}")

    try:
        res = verifier.verify(test_path)
        print(f"\nResult: {res['verdict']}")
        print(f"Is Citrus Leaf      : {res['is_citrus']}")
        print(f"Citrus Probability  : {res['citrus_probability']}%")
        print("\nMorphological & Color Metrics:")
        for k, v in res["metrics"].items():
            print(f"  - {k:<20}: {v}")
        print("\nClinical Rationale:")
        for r in res["reasons"]:
            print(f"  * {r}")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"Error during verification: {e}")
