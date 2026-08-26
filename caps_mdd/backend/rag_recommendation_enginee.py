"""
RAG Treatment Recommendation Engine
=====================================
Full 3-step RAG pipeline:
  Step 1 — Case Interpretation  (same as before, builds structured case)
  Step 2 — Semantic Retrieval   (ChromaDB vector search, replaces dict lookup)
  Step 3 — LLM Synthesis        (Groq/Llama generates grounded recommendation)

Usage:
    from rag_recommendation_engine import get_recommendation
    result = get_recommendation(
        disease="Canker",
        confidence=94.1,
        severity_score=42.5,
        severity_label="Moderate"
    )
    print(result)

Environment variable required:
    GROQ_API_KEY=your_key_here
"""

import os
import json
from groq import Groq

from rag_retriever import retrieve_evidence

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — CASE INTERPRETATION LAYER  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

CONFIDENCE_THRESHOLD = 60.0

DISEASE_ALIASES = {
    "Greening":        "Greening",
    "HLB":             "Greening",
    "BlackSpot":       "BlackSpot",
    "Black_spot":      "BlackSpot",
    "BacterialBlight": "BacterialBlight",
}

# Minimal KB for Step 1 validation only (full content lives in vector store)
KNOWN_DISEASES = {
    "Canker", "Greening", "Anthracnose", "Melanose", "BlackSpot",
    "BacterialBlight", "CurlVirus", "DeficiencyLeaf", "DryLeaf",
    "SootyMould", "SpiderMites", "LeafMiner", "Healthy",
}

DISEASE_TYPES = {
    "Canker": "Bacterial", "Greening": "Bacterial (phloem-limited)",
    "Anthracnose": "Fungal", "Melanose": "Fungal", "BlackSpot": "Fungal",
    "BacterialBlight": "Bacterial", "CurlVirus": "Viral (aphid-transmitted)",
    "DeficiencyLeaf": "Abiotic", "DryLeaf": "Abiotic",
    "SootyMould": "Fungal (secondary)", "SpiderMites": "Pest (Arachnid)",
    "LeafMiner": "Pest (Lepidoptera)", "Healthy": "None",
}

DISEASE_FULL_NAMES = {
    "Canker":          "Citrus Canker (Xanthomonas axonopodis)",
    "Greening":        "Citrus Greening / Huanglongbing (HLB)",
    "Anthracnose":     "Anthracnose (Colletotrichum gloeosporioides)",
    "Melanose":        "Melanose (Diaporthe citri)",
    "BlackSpot":       "Citrus Black Spot (Phyllosticta citricarpa)",
    "BacterialBlight": "Bacterial Blight (Pseudomonas syringae)",
    "CurlVirus":       "Citrus Leaf Curl Virus",
    "DeficiencyLeaf":  "Nutrient Deficiency (micronutrient)",
    "DryLeaf":         "Dry Leaf / Drought Stress",
    "SootyMould":      "Sooty Mould (Capnodium citri)",
    "SpiderMites":     "Spider Mites (Panonychus citri / Tetranychus urticae)",
    "LeafMiner":       "Citrus Leaf Miner (Phyllocnistis citrella)",
    "Healthy":         "Healthy leaf — no disease detected",
}


def interpret_case(disease: str,
                   confidence: float,
                   severity_score: float,
                   severity_label: str) -> dict:
    """Step 1: Validate and structure the classifier output."""

    disease_norm = DISEASE_ALIASES.get(disease, disease)

    if disease_norm not in KNOWN_DISEASES:
        return {
            "valid":          False,
            "error":          f"Unknown disease: {disease}",
            "disease":        disease_norm,
            "confidence":     confidence,
            "severity_score": severity_score,
            "severity_label": severity_label,
            "treatment_need": "Unknown",
        }

    low_confidence = confidence < CONFIDENCE_THRESHOLD

    if disease_norm == "Healthy":
        treatment_need = "None"
    elif severity_label == "Mild":
        treatment_need = "Monitor"
    elif severity_label == "Moderate":
        treatment_need = "Treat"
    elif severity_label == "Severe":
        treatment_need = "Treat urgently"
    else:
        treatment_need = "Emergency — consult expert"

    return {
        "valid":          True,
        "disease":        disease_norm,
        "full_name":      DISEASE_FULL_NAMES.get(disease_norm, disease_norm),
        "disease_type":   DISEASE_TYPES.get(disease_norm, "Unknown"),
        "confidence":     round(confidence, 1),
        "low_confidence": low_confidence,
        "severity_score": round(severity_score, 1),
        "severity_label": severity_label,
        "treatment_need": treatment_need,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — LLM SYNTHESIS LAYER
# Replaces the hard-coded decision logic with a grounded LLM call.
# The LLM receives:
#   - case context (disease, severity, treatment_need)
#   - retrieved chunks as evidence
# And must return a structured JSON recommendation.
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert agricultural plant pathologist specialising in citrus diseases.
Your task is to generate a precise, evidence-based treatment recommendation for a detected citrus disease.

You will be given:
1. A structured case summary (disease, severity, confidence, treatment urgency)
2. Retrieved knowledge chunks from a citrus treatment database

Rules:
- Base your recommendation ONLY on the retrieved chunks. Do not invent products or doses.
- Select ONE primary treatment product. For Mild/Moderate severity, prefer organic options. For Severe/Critical, prefer conventional/stronger options.
- Adjust dose wording to reflect severity (lower range for Mild, upper range for Severe/Critical).
- Be concise and direct. Farmers need actionable advice.
- Always include the cultural advice and monitoring guidance from the retrieved context.
- If no treatment exists (Healthy), say so clearly.

You MUST respond with ONLY a valid JSON object — no markdown, no explanation, no preamble.
The JSON must exactly match this schema:
{
  "recommended_treatment": "string — product name",
  "product": "string",
  "dose": "string — adjusted for severity",
  "timing": "string",
  "interval": "string",
  "method": "string",
  "warning": "string or null",
  "organic": true/false,
  "cultural_advice": "string",
  "monitoring": "string",
  "urgency_note": "string",
  "reason": "string — 1-2 sentence clinical justification",
  "low_confidence_warning": "string or null"
}"""


def build_llm_prompt(case: dict, evidence: dict) -> str:
    """Build the user-turn prompt for the LLM synthesis step."""

    urgency_map = {
        "None":                       "No treatment needed — healthy plant.",
        "Monitor":                    "Low urgency — monitor, treat only if worsening.",
        "Treat":                      "Moderate urgency — apply treatment within 2–3 days.",
        "Treat urgently":             "High urgency — apply treatment today.",
        "Emergency — consult expert": "CRITICAL — apply emergency treatment immediately and contact an agronomist.",
    }

    prompt = f"""CASE SUMMARY
============
Disease       : {case['full_name']} ({case['disease']})
Disease type  : {case['disease_type']}
Confidence    : {case['confidence']}%
Severity      : {case['severity_label']} (score: {case['severity_score']}/100)
Treatment need: {case['treatment_need']}
Urgency       : {urgency_map.get(case['treatment_need'], '')}
Low confidence: {case.get('low_confidence', False)}

RETRIEVED EVIDENCE
==================
{evidence['raw_context'] if evidence['raw_context'] else 'No evidence retrieved.'}

INSTRUCTION
===========
Based on the case summary and retrieved evidence above, generate the treatment recommendation JSON.
Remember: severity is {case['severity_label']}, so {'prefer organic / lower dose range' if case['severity_label'] in ('Mild', 'Moderate') else 'prefer conventional / upper dose range'}.
"""
    return prompt


def synthesize_recommendation(case: dict, evidence: dict) -> dict:
    """
    Step 3: Call Groq (Llama 3.3 70B) to synthesize a grounded recommendation.

    Args:
        case     : output from interpret_case()
        evidence : output from retrieve_evidence()

    Returns:
        final recommendation dict
    """

    # ── Healthy shortcut — no LLM call needed ────────────────────────────────
    if case["disease"] == "Healthy" or case["treatment_need"] == "None":
        cultural = ""
        monitoring = ""
        if evidence["cultural_chunks"]:
            cultural   = evidence["cultural_chunks"][0]["text"]
        return {
            "disease":                case["disease"],
            "full_name":              case["full_name"],
            "confidence":             case["confidence"],
            "severity_label":         "Healthy",
            "severity_score":         case["severity_score"],
            "treatment_need":         "None",
            "recommended_treatment":  "No treatment required",
            "product":                None,
            "dose":                   None,
            "timing":                 None,
            "interval":               None,
            "method":                 None,
            "warning":                None,
            "organic":                None,
            "cultural_advice":        "Continue regular fertilization, irrigation, and pest monitoring.",
            "monitoring":             "Inspect weekly as a preventive measure.",
            "urgency_note":           None,
            "low_confidence_warning": None,
            "reason":                 "Leaf appears healthy. No disease detected.",
        }

    # ── Invalid case shortcut ────────────────────────────────────────────────
    if not case.get("valid"):
        return {
            "disease":                case.get("disease", "Unknown"),
            "full_name":              "Unknown",
            "confidence":             case.get("confidence", 0),
            "severity_label":         case.get("severity_label", "Unknown"),
            "treatment_need":         "Unknown",
            "recommended_treatment":  "Consult local agricultural extension officer",
            "product":                None,
            "dose":                   None,
            "timing":                 "As soon as possible",
            "interval":               None,
            "method":                 None,
            "warning":                f"Unrecognised disease: {case.get('error', '')}",
            "organic":                None,
            "cultural_advice":        "",
            "monitoring":             "",
            "urgency_note":           "Seek expert advice immediately.",
            "low_confidence_warning": None,
            "reason":                 f"Disease not found in knowledge base: {case.get('disease', '')}",
        }

    # ── LLM synthesis ─────────────────────────────────────────────────────────
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    user_prompt = build_llm_prompt(case, evidence)

    try:
        response = client.chat.completions.create(
            model      = "llama-3.3-70b-versatile",
            max_tokens = 1024,
            messages   = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        )

        raw_text = response.choices[0].message.content.strip()

        # Strip any accidental markdown fences
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        llm_output = json.loads(raw_text)

    except json.JSONDecodeError as e:
        # Fallback: return partial result with error note
        llm_output = {
            "recommended_treatment": "LLM parse error — consult agronomist",
            "warning":               f"JSON parse error: {e}",
            "reason":                "LLM output could not be parsed.",
        }
    except Exception as e:
        llm_output = {
            "recommended_treatment": "LLM error — consult agronomist",
            "warning":               f"API error: {e}",
            "reason":                "LLM call failed.",
        }

    # ── Merge LLM output with case metadata ──────────────────────────────────
    result = {
        "disease":        case["disease"],
        "full_name":      case["full_name"],
        "disease_type":   case["disease_type"],
        "confidence":     case["confidence"],
        "severity_label": case["severity_label"],
        "severity_score": case["severity_score"],
        "treatment_need": case["treatment_need"],
        **llm_output,
    }

    # Ensure low_confidence_warning is set if LLM omitted it
    if case.get("low_confidence") and not result.get("low_confidence_warning"):
        result["low_confidence_warning"] = (
            f"Model confidence is {case['confidence']}% — below threshold. "
            "Visually verify the disease identification before applying treatment."
        )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def get_recommendation(disease: str,
                       confidence: float,
                       severity_score: float,
                       severity_label: str) -> dict:
    """
    Full RAG pipeline.

    Args:
        disease        : disease name from classifier (e.g. "Canker")
        confidence     : model confidence % (e.g. 94.1)
        severity_score : hybrid severity score 0–100 (e.g. 42.5)
        severity_label : "Mild" / "Moderate" / "Severe" / "Critical"

    Returns:
        complete recommendation dict
    """
    case     = interpret_case(disease, confidence, severity_score, severity_label)
    evidence = retrieve_evidence(case)
    result   = synthesize_recommendation(case, evidence)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_cases = [
        ("Canker",      94.1, 42.5, "Moderate"),
        ("Anthracnose", 61.2, 68.0, "Severe"),
        ("Healthy",     98.5,  0.0, "Mild"),
        ("Greening",    88.0, 80.0, "Critical"),
        ("SpiderMites", 55.0, 18.0, "Mild"),
    ]

    for disease, conf, score, label in test_cases:
        print(f"\n{'═'*60}")
        print(f"  Disease: {disease}  |  Confidence: {conf}%  |  Severity: {label}")
        print(f"{'═'*60}")
        rec = get_recommendation(disease, conf, score, label)
        print(f"  Treatment : {rec.get('recommended_treatment')}")
        print(f"  Dose      : {rec.get('dose')}")
        print(f"  Timing    : {rec.get('timing')}")
        print(f"  Urgency   : {rec.get('urgency_note')}")
        print(f"  Reason    : {rec.get('reason')}")
        if rec.get("warning"):
            print(f"  Warning   : {rec['warning']}")
        if rec.get("low_confidence_warning"):
            print(f"  LOW CONF  : {rec['low_confidence_warning']}")