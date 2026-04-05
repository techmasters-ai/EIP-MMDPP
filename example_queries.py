
"""
EIP-MMDPP — Example Queries
============================

Examples for each of the 3 retrieval query strategies.
All examples use the synchronous POST /v1/retrieval/query endpoint.

Prerequisites:
    pip install requests

Usage:
    python example_queries.py [1] [2] [3]

    Omit arguments to run all examples.
"""

import base64
import sys
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000/v1"
QUERY_URL = f"{BASE_URL}/retrieval/query"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_header(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_results(resp: dict) -> None:
    """Pretty-print a UnifiedQueryResponse."""
    print(f"  Strategy:        {resp.get('strategy', 'N/A')}")
    print(f"  Modality Filter: {resp.get('modality_filter', 'N/A')}")
    print(f"  Total Results:   {resp.get('total', 0)}\n")

    for i, item in enumerate(resp.get("results", []), 1):
        print(f"  --- Result {i} ---")
        score = item.get("score")
        if score is not None:
            print(f"  Score:          {score:.4f}")
        print(f"  Modality:       {item.get('modality', 'N/A')}")
        print(f"  Classification: {item.get('classification', 'N/A')}")
        if item.get("chunk_id"):
            print(f"  Chunk ID:       {item['chunk_id']}")
        if item.get("document_id"):
            print(f"  Document ID:    {item['document_id']}")
        if item.get("page_number"):
            print(f"  Page:           {item['page_number']}")
        if item.get("image_url"):
            print(f"  Image URL:      {item['image_url']}")

        text = item.get("content_text") or ""
        if text:
            preview = text[:200] + ("..." if len(text) > 200 else "")
            print(f"  Content:        {preview}")

        # Global query results include community context
        ctx = item.get("context")
        if ctx and isinstance(ctx, dict):
            source = ctx.get("source", "")
            if source:
                print(f"  Context Source:  {source}")
            community_ctx = ctx.get("community_context")
            if community_ctx and isinstance(community_ctx, dict):
                if community_ctx.get("community_id"):
                    print(f"  Community:      {community_ctx['community_id']}")
                if community_ctx.get("summary"):
                    summary_preview = community_ctx["summary"][:150]
                    print(f"  Summary:        {summary_preview}...")
        print()


# ---------------------------------------------------------------------------
# 1. Basic Text Query
# ---------------------------------------------------------------------------

def example_basic_text():
    """Basic vector search over text chunks stored in ArcadeDB.

    - Input:  text query only (no image support)
    - Output: ranked text/table chunks with scores
    - Speed:  fast (1-3s, no LLM calls)
    """
    print_header("1. Basic Text Query (strategy=basic)")

    payload = {
        "query_text": "radar signal processing specifications",
        "strategy": "basic",
        "top_k": 5,
        "include_context": True,
        # Optional filters:
        # "min_confidence": 0.3,
        # "reranker_top_n": 20,
        # "filters": {
        #     "classification": "UNCLASSIFIED",
        #     "modalities": ["text", "table"],
        #     "document_ids": ["<uuid>"],
        # },
    }

    resp = requests.post(QUERY_URL, json=payload)
    resp.raise_for_status()
    print_results(resp.json())


# ---------------------------------------------------------------------------
# 2. Multi-Modal Query (Hybrid)
# ---------------------------------------------------------------------------

def example_multi_modal():
    """Full multi-modal pipeline: text + CLIP image search, graph expansion,
    ontology traversal, weighted fusion scoring, cross-encoder reranking.

    - Input:  text and/or base64-encoded image
    - Output: mixed text, image, table, schematic, and image_description chunks
    - Speed:  medium (5-15s, includes graph expansion + reranking)
    """
    print_header("2. Multi-Modal Query (strategy=hybrid)")

    # --- 2a. Text-only hybrid query ---
    print("  [2a] Text-only hybrid search (all modalities):\n")
    payload = {
        "query_text": "VHF radar internal components",
        "strategy": "hybrid",
        "modality_filter": "all",  # "all", "text", or "image"
        "top_k": 5,
        "include_context": True,
    }

    resp = requests.post(QUERY_URL, json=payload)
    resp.raise_for_status()
    print_results(resp.json())

    # --- 2b. Image query (base64-encoded) ---
    print("  [2b] Image-based hybrid search:\n")
    image_path = Path("test_image.png")
    if image_path.exists():
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        payload_img = {
            "query_image": image_b64,
            "strategy": "hybrid",
            "modality_filter": "image",
            "top_k": 5,
        }
        resp = requests.post(QUERY_URL, json=payload_img)
        resp.raise_for_status()
        print_results(resp.json())
    else:
        print("  (Skipped — no test_image.png found. Place an image file in the")
        print("   project root to test image-based queries.)\n")

    # --- 2c. Combined text + image query ---
    print("  [2c] Combined text + image hybrid search:\n")
    if image_path.exists():
        payload_both = {
            "query_text": "missile launcher diagram",
            "query_image": image_b64,
            "strategy": "hybrid",
            "modality_filter": "all",
            "top_k": 5,
        }
        resp = requests.post(QUERY_URL, json=payload_both)
        resp.raise_for_status()
        print_results(resp.json())
    else:
        print("  (Skipped — no test_image.png found.)\n")


# ---------------------------------------------------------------------------
# 3. Global Query (Community Detection)
# ---------------------------------------------------------------------------

def example_global_query():
    """Community-aware global query using ArcadeDB community detection.

    Runs Louvain community detection over the ArcadeDB knowledge graph and
    uses LLM-generated community summaries to answer broad, holistic questions
    that span multiple entity groups and document sources.

    - Input:    text query only
    - Output:   LLM-generated response grounded in community summaries
    - Speed:    medium-slow (15-60s, depends on number of communities)
    - Requires: at least one successful community detection run
    """
    print_header("3. Global Query (strategy=global)")

    payload = {
        "query_text": "What are the major categories of air defense systems and how do they compare?",
        "strategy": "global",
        "top_k": 10,
        "include_context": True,
    }

    try:
        resp = requests.post(QUERY_URL, json=payload, timeout=120)
        resp.raise_for_status()
        print_results(resp.json())
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 409:
            print("  Community detection has not completed yet.")
            print("  Run: POST /v1/community/detect to trigger community detection.\n")
        else:
            raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXAMPLES = {
    "1": ("Basic Text",    example_basic_text),
    "2": ("Multi-Modal",   example_multi_modal),
    "3": ("Global Query",  example_global_query),
}


def main():
    if len(sys.argv) > 1:
        choices = sys.argv[1:]
    else:
        choices = list(EXAMPLES.keys())

    print("EIP-MMDPP Query Examples")
    print(f"API Base URL: {BASE_URL}")

    for choice in choices:
        if choice not in EXAMPLES:
            print(f"\nUnknown example '{choice}'. Valid: {', '.join(EXAMPLES.keys())}")
            continue
        name, func = EXAMPLES[choice]
        try:
            func()
        except requests.exceptions.ConnectionError:
            print(f"\n  ERROR: Cannot connect to {BASE_URL}")
            print("  Make sure the API is running: ./manage.sh --start\n")
            break
        except Exception as e:
            print(f"\n  ERROR running {name}: {e}\n")


if __name__ == "__main__":
    main()
