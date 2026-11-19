"""
RAG Knowledge Base — Document Chunking & Ingestion
====================================================
Converts the flat KNOWLEDGE_BASE dict into semantically rich text chunks,
embeds them, and stores them in ChromaDB for retrieval.

Install deps:
    pip install chromadb sentence-transformers anthropic

Run once to build the vector store:
    python rag_knowledge_base.py
"""

import os
import json
import chromadb
from chromadb.utils import embedding_functions

# ── Same KB as before (source of truth) ──────────────────────────────────────

KNOWLEDGE_BASE = {
    "Canker": {
        "full_name":    "Citrus Canker (Xanthomonas axonopodis)",
        "type":         "Bacterial",
        "treatments": [
            {
                "product":  "Copper-based bactericide",
                "dose":     "2–3 g/L water",
                "timing":   "Apply at first symptom appearance",
                "interval": "Repeat every 10–14 days",
                "method":   "Foliar spray",
                "warning":  "Do not spray during flowering. Avoid copper buildup in soil.",
                "organic":  True,
            },
            {
                "product":  "Streptomycin sulfate",
                "dose":     "200 ppm solution",
                "timing":   "Apply preventively before rainy season",
                "interval": "Every 7 days during high-risk period",
                "method":   "Foliar spray",
                "warning":  "Check local regulations — restricted in some regions.",
                "organic":  False,
            },
        ],
        "cultural":   "Remove and destroy infected plant parts. Disinfect pruning tools with 10% bleach.",
        "monitoring": "Inspect weekly. Isolate infected trees to prevent spread.",
    },
    "Greening": {
        "full_name":    "Citrus Greening / Huanglongbing (HLB)",
        "type":         "Bacterial (phloem-limited)",
        "treatments": [
            {
                "product":  "Imidacloprid (systemic insecticide)",
                "dose":     "As per label — soil drench or foliar",
                "timing":   "Apply to control Asian citrus psyllid vector",
                "interval": "Every 3 months",
                "method":   "Soil drench or foliar spray",
                "warning":  "No cure exists. Treatment controls the insect vector only.",
                "organic":  False,
            },
            {
                "product":  "Kaolin clay spray",
                "dose":     "25 g/L water",
                "timing":   "Apply before psyllid season",
                "interval": "Every 2 weeks",
                "method":   "Foliar spray",
                "warning":  "Organic option — reduces psyllid settling.",
                "organic":  True,
            },
        ],
        "cultural":   "No cure. Remove and destroy severely infected trees immediately to prevent spread to healthy trees.",
        "monitoring": "Scout for Asian citrus psyllid weekly. Use yellow sticky traps.",
    },
    "Anthracnose": {
        "full_name":    "Anthracnose (Colletotrichum gloeosporioides)",
        "type":         "Fungal",
        "treatments": [
            {
                "product":  "Azoxystrobin",
                "dose":     "As per label — typically 0.8–1.0 mL/L",
                "timing":   "Apply at early symptom stage",
                "interval": "Repeat after 7–10 days if required",
                "method":   "Foliar spray",
                "warning":  "Do not apply more than 3 times per season to avoid resistance.",
                "organic":  False,
            },
            {
                "product":  "Copper fungicide",
                "dose":     "2–3 g/L water",
                "timing":   "Apply during wet season preventively",
                "interval": "Every 10–14 days",
                "method":   "Foliar spray",
                "warning":  "Avoid spraying before forecast rainfall.",
                "organic":  True,
            },
        ],
        "cultural":   "Improve air circulation by pruning. Remove fallen leaves and debris.",
        "monitoring": "Check after rainfall events. High humidity increases risk.",
    },
    "Melanose": {
        "full_name":    "Melanose (Diaporthe citri)",
        "type":         "Fungal",
        "treatments": [
            {
                "product":  "Copper oxychloride",
                "dose":     "3 g/L water",
                "timing":   "Apply before rainy season, after petal fall",
                "interval": "Every 2–3 weeks during wet season",
                "method":   "Foliar spray",
                "warning":  "Do not apply in hot dry conditions.",
                "organic":  True,
            },
            {
                "product":  "Mancozeb",
                "dose":     "2.5 g/L water",
                "timing":   "Apply at fruit set stage",
                "interval": "Every 14 days",
                "method":   "Foliar spray",
                "warning":  "Pre-harvest interval: 7 days.",
                "organic":  False,
            },
        ],
        "cultural":   "Remove dead wood and twigs. Prune to reduce canopy density.",
        "monitoring": "Inspect during wet weather. Spores spread via rain splash.",
    },
    "BlackSpot": {
        "full_name":    "Citrus Black Spot (Phyllosticta citricarpa)",
        "type":         "Fungal",
        "treatments": [
            {
                "product":  "Trifloxystrobin + tebuconazole",
                "dose":     "As per label",
                "timing":   "Apply from fruit set through summer",
                "interval": "Every 6–8 weeks",
                "method":   "Foliar spray",
                "warning":  "Quarantine pest in some countries — check regulations.",
                "organic":  False,
            },
            {
                "product":  "Copper fungicide",
                "dose":     "2 g/L water",
                "timing":   "Preventive application before wet season",
                "interval": "Every 3 weeks",
                "method":   "Foliar spray",
                "warning":  "Organic option. Less effective at high disease pressure.",
                "organic":  True,
            },
        ],
        "cultural":   "Remove fallen leaves (primary inoculum source). Avoid over-irrigation.",
        "monitoring": "Inspect fruit surface fortnightly from fruit set.",
    },
    "BacterialBlight": {
        "full_name":    "Bacterial Blight (Pseudomonas syringae)",
        "type":         "Bacterial",
        "treatments": [
            {
                "product":  "Copper hydroxide spray",
                "dose":     "2–3 g/L water",
                "timing":   "Apply before rainy season and after pruning",
                "interval": "Every 10 days during wet conditions",
                "method":   "Foliar spray",
                "warning":  "Avoid overhead irrigation — promotes spread.",
                "organic":  True,
            },
        ],
        "cultural":   "Use copper-treated pruning cuts. Disinfect all tools. Avoid wounding plants.",
        "monitoring": "Monitor after storms or hail damage which creates entry wounds.",
    },
    "CurlVirus": {
        "full_name":    "Citrus Leaf Curl Virus",
        "type":         "Viral (aphid-transmitted)",
        "treatments": [
            {
                "product":  "Imidacloprid",
                "dose":     "As per label",
                "timing":   "Apply to control aphid vector population",
                "interval": "Every 3–4 weeks during aphid season",
                "method":   "Foliar spray or soil drench",
                "warning":  "Controls vector, not the virus itself.",
                "organic":  False,
            },
            {
                "product":  "Neem oil",
                "dose":     "5 mL/L water + few drops dish soap",
                "timing":   "Apply when aphids are detected",
                "interval": "Every 7 days",
                "method":   "Foliar spray (coat undersides of leaves)",
                "warning":  "Organic option. Do not spray in direct sun.",
                "organic":  True,
            },
        ],
        "cultural":   "Remove and destroy severely infected shoots. Control ant populations that protect aphids.",
        "monitoring": "Check new growth weekly for aphid colonies.",
    },
    "DeficiencyLeaf": {
        "full_name":    "Nutrient Deficiency (micronutrient)",
        "type":         "Abiotic",
        "treatments": [
            {
                "product":  "Chelated micronutrient fertilizer",
                "dose":     "As per soil test recommendation",
                "timing":   "Apply at start of growing season",
                "interval": "Every 6–8 weeks",
                "method":   "Soil application or foliar spray",
                "warning":  "Conduct soil test first to identify specific deficiency.",
                "organic":  True,
            },
            {
                "product":  "Zinc sulfate (if zinc deficient)",
                "dose":     "2 g/L water for foliar",
                "timing":   "Apply when deficiency symptoms appear",
                "interval": "Every 4 weeks until recovery",
                "method":   "Foliar spray",
                "warning":  "Excess zinc is toxic. Do not exceed recommended rate.",
                "organic":  False,
            },
        ],
        "cultural":   "Check soil pH — high pH locks out micronutrients. Adjust pH to 6.0–7.0.",
        "monitoring": "Compare leaf color against healthy reference. Test soil pH.",
    },
    "DryLeaf": {
        "full_name":    "Dry Leaf / Drought Stress",
        "type":         "Abiotic",
        "treatments": [
            {
                "product":  "Water / irrigation",
                "dose":     "Restore soil moisture to field capacity",
                "timing":   "Immediately upon detection",
                "interval": "Daily until recovery, then regular schedule",
                "method":   "Drip or basin irrigation",
                "warning":  "Sudden overwatering after drought can cause root shock.",
                "organic":  True,
            },
            {
                "product":  "Seaweed-based biostimulant",
                "dose":     "2–3 mL/L water",
                "timing":   "Apply after rehydration",
                "interval": "Every 2 weeks",
                "method":   "Foliar spray",
                "warning":  "Supports recovery — not a substitute for adequate irrigation.",
                "organic":  True,
            },
        ],
        "cultural":   "Mulch around base to retain soil moisture. Check irrigation system for blockages.",
        "monitoring": "Monitor soil moisture daily. Install drip irrigation if recurring.",
    },
    "SootyMould": {
        "full_name":    "Sooty Mould (Capnodium citri)",
        "type":         "Fungal (secondary — grows on insect honeydew)",
        "treatments": [
            {
                "product":  "Soap solution wash",
                "dose":     "5 mL mild dish soap per litre of water",
                "timing":   "Wash affected leaves immediately",
                "interval": "Weekly until mould disappears",
                "method":   "Spray and wipe leaves",
                "warning":  "Treat underlying insect pest first or mould will return.",
                "organic":  True,
            },
            {
                "product":  "Imidacloprid (for scale/mealybug control)",
                "dose":     "As per label",
                "timing":   "Apply to control honeydew-producing insects",
                "interval": "Every 4 weeks",
                "method":   "Foliar or soil drench",
                "warning":  "Target the insect pest — sooty mould is a symptom, not cause.",
                "organic":  False,
            },
        ],
        "cultural":   "Prune to improve airflow. Control scale insects and mealybugs which produce honeydew.",
        "monitoring": "Inspect undersides of leaves for scale, mealybug, whitefly colonies.",
    },
    "SpiderMites": {
        "full_name":    "Spider Mites (Panonychus citri / Tetranychus urticae)",
        "type":         "Pest (Arachnid)",
        "treatments": [
            {
                "product":  "Abamectin miticide",
                "dose":     "0.5–1.0 mL/L water",
                "timing":   "Apply when mite population exceeds threshold",
                "interval": "Repeat after 7–10 days if needed",
                "method":   "Foliar spray — ensure full coverage of leaf undersides",
                "warning":  "Rotate with different chemistry to prevent resistance.",
                "organic":  False,
            },
            {
                "product":  "Neem oil",
                "dose":     "5 mL/L water",
                "timing":   "Apply at first mite detection",
                "interval": "Every 5–7 days",
                "method":   "Foliar spray (coat leaf undersides thoroughly)",
                "warning":  "Organic option. Spray in cooler hours to avoid phytotoxicity.",
                "organic":  True,
            },
        ],
        "cultural":   "Increase humidity around plants. Avoid dusty conditions. Introduce predatory mites.",
        "monitoring": "Use hand lens to inspect leaf undersides. Look for fine webbing and stippling.",
    },
    "LeafMiner": {
        "full_name":    "Citrus Leaf Miner (Phyllocnistis citrella)",
        "type":         "Pest (Lepidoptera)",
        "treatments": [
            {
                "product":  "Spinosad",
                "dose":     "0.5 mL/L water",
                "timing":   "Apply when new leaf flushes are detected",
                "interval": "Every 7–10 days during flush period",
                "method":   "Foliar spray on new growth",
                "warning":  "Effective against young larvae only — not adults.",
                "organic":  True,
            },
            {
                "product":  "Imidacloprid (systemic)",
                "dose":     "As per label",
                "timing":   "Soil drench before new flush emergence",
                "interval": "Once per flush cycle",
                "method":   "Soil drench",
                "warning":  "Systemic — absorbed by plant and kills mining larvae.",
                "organic":  False,
            },
        ],
        "cultural":   "Remove severely mined leaves. Time pruning to avoid stimulating excessive new flushes.",
        "monitoring": "Monitor new growth flushes. Check for characteristic winding mines on young leaves.",
    },
    "Healthy": {
        "full_name":    "Healthy leaf — no disease detected",
        "type":         "None",
        "treatments":   [],
        "cultural":     "Continue regular fertilization, irrigation, and pest monitoring.",
        "monitoring":   "Inspect weekly as a preventive measure.",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# CHUNKING STRATEGY
# Each disease gets chunked into 3 document types for fine-grained retrieval:
#   1. disease_overview   — identity, type, full name
#   2. treatment_<n>      — one chunk per treatment option (preserves structure)
#   3. cultural_monitoring — cultural practices + monitoring guidance
# ══════════════════════════════════════════════════════════════════════════════

def build_chunks() -> list[dict]:
    """
    Convert KB entries into semantically chunked documents.
    Returns list of {id, text, metadata} dicts.
    """
    chunks = []

    for disease_key, entry in KNOWLEDGE_BASE.items():
        full_name    = entry["full_name"]
        disease_type = entry["type"]

        # ── Chunk 1: Disease overview ─────────────────────────────────────────
        overview_text = (
            f"Disease: {full_name} (key: {disease_key})\n"
            f"Type: {disease_type}\n"
            f"Common name: {disease_key}\n"
            f"This disease affects citrus leaves and is classified as {disease_type.lower()}."
        )
        chunks.append({
            "id":   f"{disease_key}_overview",
            "text": overview_text,
            "metadata": {
                "disease":      disease_key,
                "full_name":    full_name,
                "disease_type": disease_type,
                "chunk_type":   "overview",
                "organic":      "N/A",
            }
        })

        # ── Chunk 2+: One chunk per treatment option ─────────────────────────
        for i, t in enumerate(entry.get("treatments", [])):
            organic_label = "organic" if t.get("organic") else "conventional"
            treatment_text = (
                f"Treatment for {full_name} ({disease_key}) — Option {i+1} [{organic_label}]\n"
                f"Product: {t['product']}\n"
                f"Dose: {t['dose']}\n"
                f"Timing: {t['timing']}\n"
                f"Application interval: {t['interval']}\n"
                f"Method: {t['method']}\n"
                f"Warning: {t['warning']}\n"
                f"Organic: {'Yes' if t.get('organic') else 'No'}"
            )
            chunks.append({
                "id":   f"{disease_key}_treatment_{i}",
                "text": treatment_text,
                "metadata": {
                    "disease":        disease_key,
                    "full_name":      full_name,
                    "disease_type":   disease_type,
                    "chunk_type":     "treatment",
                    "treatment_index": i,
                    "product":        t["product"],
                    "organic":        str(t.get("organic", False)),
                }
            })

        # ── Chunk 3: Cultural practices + monitoring ──────────────────────────
        cultural_text = (
            f"Cultural practices and monitoring for {full_name} ({disease_key})\n"
            f"Cultural management: {entry.get('cultural', 'N/A')}\n"
            f"Monitoring guidance: {entry.get('monitoring', 'N/A')}"
        )
        chunks.append({
            "id":   f"{disease_key}_cultural",
            "text": cultural_text,
            "metadata": {
                "disease":      disease_key,
                "full_name":    full_name,
                "disease_type": disease_type,
                "chunk_type":   "cultural",
                "organic":      "N/A",
            }
        })

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# VECTOR STORE — ChromaDB with sentence-transformers embedding
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHROMA_PATH = os.path.join(BASE_DIR, "citrus_rag_db")
CHROMA_PATH       = os.environ.get("CHROMA_PATH", DEFAULT_CHROMA_PATH)
COLLECTION_NAME   = "citrus_treatments"
EMBED_MODEL_NAME  = "all-MiniLM-L6-v2"   # fast, good quality, runs locally


def build_vector_store(force_rebuild: bool = False) -> chromadb.Collection:
    """
    Build (or load) the ChromaDB collection from KB chunks.
    Set force_rebuild=True to wipe and re-ingest.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Embedding function — sentence-transformers, runs locally (no API key)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL_NAME
    )

    # Delete existing collection if rebuilding
    if force_rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"[KB] Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass

    # Get or create collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Only ingest if collection is empty
    if collection.count() == 0:
        chunks = build_chunks()
        print(f"[KB] Ingesting {len(chunks)} chunks into ChromaDB...")

        collection.add(
            ids       = [c["id"]       for c in chunks],
            documents = [c["text"]     for c in chunks],
            metadatas = [c["metadata"] for c in chunks],
        )
        print(f"[KB] Done. Collection has {collection.count()} documents.")
    else:
        print(f"[KB] Loaded existing collection ({collection.count()} documents).")

    return collection


def get_collection() -> chromadb.Collection:
    """Convenience loader — use in retrieval pipeline."""
    return build_vector_store(force_rebuild=False)


# ── Run directly to build the store ──────────────────────────────────────────
if __name__ == "__main__":
    col = build_vector_store(force_rebuild=True)
    print("\nSample chunks ingested:")
    results = col.query(
        query_texts=["treatment for citrus canker bacterial disease"],
        n_results=3
    )
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"\n  [{meta['chunk_type'].upper()}] {meta['disease']}")
        print(f"  {doc[:120]}...")
