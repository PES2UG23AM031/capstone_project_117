"""
RAG Retriever — Step 2: Evidence Retrieval Layer
=================================================
Replaces the flat dict lookup with semantic vector search.
Builds a structured query from the case context and retrieves
the top-k most relevant treatment chunks from ChromaDB.
"""

from rag_knowledge_base import get_collection

# ── Retrieval config ──────────────────────────────────────────────────────────

TOP_K_TREATMENTS  = 3   # treatment chunks to retrieve
TOP_K_CULTURAL    = 1   # cultural/monitoring chunks to retrieve
TOP_K_OVERVIEW    = 1   # overview chunks to retrieve


# ══════════════════════════════════════════════════════════════════════════════
# QUERY FORMULATION
# Turns structured case dict → natural language semantic query
# This is the key difference from the old lookup: the query encodes
# severity + urgency context so the vector search can rank accordingly
# ══════════════════════════════════════════════════════════════════════════════

def formulate_query(case: dict) -> dict:
    """
    Build targeted retrieval queries for each chunk type.

    Returns:
        {
            "treatment_query": str,   # used to retrieve treatment chunks
            "cultural_query":  str,   # used to retrieve cultural/monitoring chunks
        }
    """
    disease       = case.get("disease", "")
    full_name     = case.get("full_name", disease)
    severity      = case.get("severity_label", "Moderate")
    disease_type  = case.get("disease_type", "")
    treatment_need = case.get("treatment_need", "Treat")

    # Urgency language for semantic alignment with chunk content
    urgency_hints = {
        "Monitor":                    "mild symptoms monitor and wait",
        "Treat":                      "standard treatment apply spray",
        "Treat urgently":             "severe infection urgent strong treatment chemical",
        "Emergency — consult expert": "critical emergency immediate treatment expert",
    }
    urgency_phrase = urgency_hints.get(treatment_need, "treatment")

    # Organic preference hint — mild cases lean organic
    organic_hint = "organic natural" if severity in ("Mild", "Moderate") else "conventional chemical"

    treatment_query = (
        f"{severity} {disease} {full_name} {disease_type} "
        f"citrus leaf disease treatment product dose spray "
        f"{urgency_phrase} {organic_hint}"
    )

    cultural_query = (
        f"{disease} {full_name} cultural practices pruning "
        f"monitoring inspection management"
    )

    return {
        "treatment_query": treatment_query,
        "cultural_query":  cultural_query,
    }


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL
# Runs two separate queries (treatment + cultural) with metadata filtering
# where possible to reduce noise from cross-disease results
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_evidence(case: dict) -> dict:
    """
    Step 2 (RAG version): Retrieve relevant chunks from ChromaDB.

    Args:
        case : output from interpret_case()

    Returns:
        {
            "treatment_chunks": list[dict],  # ranked treatment options
            "cultural_chunks":  list[dict],  # cultural/monitoring advice
            "raw_context":      str,          # full text for LLM prompt
        }
    """
    if not case.get("valid", False):
        return {
            "treatment_chunks": [],
            "cultural_chunks":  [],
            "raw_context":      "",
        }

    collection = get_collection()
    disease    = case["disease"]
    queries    = formulate_query(case)

    # ── Retrieve treatment chunks ─────────────────────────────────────────────
    # Filter by disease + chunk_type for precision
    # Falls back to broader search if disease-filtered results are thin
    treatment_results = collection.query(
        query_texts = [queries["treatment_query"]],
        n_results   = TOP_K_TREATMENTS,
        where       = {
            "$and": [
                {"disease":    {"$eq": disease}},
                {"chunk_type": {"$eq": "treatment"}},
            ]
        }
    )

    treatment_chunks = _parse_results(treatment_results)

    # If no exact match (unknown disease path), broaden the search
    if not treatment_chunks:
        treatment_results = collection.query(
            query_texts = [queries["treatment_query"]],
            n_results   = TOP_K_TREATMENTS,
        )
        treatment_chunks = _parse_results(treatment_results)

    # ── Retrieve cultural + monitoring chunks ─────────────────────────────────
    cultural_results = collection.query(
        query_texts = [queries["cultural_query"]],
        n_results   = TOP_K_CULTURAL,
        where       = {
            "$and": [
                {"disease":    {"$eq": disease}},
                {"chunk_type": {"$eq": "cultural"}},
            ]
        }
    )

    cultural_chunks = _parse_results(cultural_results)

    # ── Build raw context string for LLM ────────────────────────────────────
    all_chunks = treatment_chunks + cultural_chunks
    raw_context = "\n\n---\n\n".join(
        f"[CHUNK: {c['metadata'].get('chunk_type', '').upper()} | "
        f"Disease: {c['metadata'].get('disease', '')} | "
        f"Similarity: {c['distance']:.3f}]\n{c['text']}"
        for c in all_chunks
    )

    return {
        "treatment_chunks": treatment_chunks,
        "cultural_chunks":  cultural_chunks,
        "raw_context":      raw_context,
        "queries":          queries,   # for logging/debugging
    }


# ── Helper ────────────────────────────────────────────────────────────────────

def _parse_results(results: dict) -> list[dict]:
    """Flatten ChromaDB query results into a list of chunk dicts."""
    chunks = []
    if not results or not results.get("documents"):
        return chunks

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metadatas, distances):
        chunks.append({
            "text":     doc,
            "metadata": meta,
            "distance": dist,   # cosine distance — lower = more similar
        })

    return chunks


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate a case from Step 1
    mock_case = {
        "valid":          True,
        "disease":        "Canker",
        "full_name":      "Citrus Canker (Xanthomonas axonopodis)",
        "disease_type":   "Bacterial",
        "confidence":     94.1,
        "severity_label": "Moderate",
        "severity_score": 42.5,
        "treatment_need": "Treat",
        "low_confidence": False,
    }

    evidence = retrieve_evidence(mock_case)

    print(f"\nQuery used: {evidence['queries']['treatment_query']}")
    print(f"\nRetrieved {len(evidence['treatment_chunks'])} treatment chunks:")
    for c in evidence["treatment_chunks"]:
        print(f"  [{c['distance']:.3f}] {c['text'][:100]}...")

    print(f"\nRetrieved {len(evidence['cultural_chunks'])} cultural chunks:")
    for c in evidence["cultural_chunks"]:
        print(f"  [{c['distance']:.3f}] {c['text'][:100]}...")
