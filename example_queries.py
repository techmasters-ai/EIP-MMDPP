
"""
EIP-MMDPP — Example Queries
============================

MVP examples for each of the 6 retrieval query strategies.
All examples use the synchronous POST /v1/retrieval/query endpoint.
GraphRAG queries also include the async submit/poll/fetch pattern
since they can take 1-3+ minutes due to LLM calls.

Prerequisites:
    pip install requests

Usage:
    python example_queries.py
"""

import base64
import sys
import time
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
    print(f"  Strategy:        {resp['strategy']}")
    print(f"  Modality Filter: {resp['modality_filter']}")
    print(f"  Total Results:   {resp['total']}\n")

    for i, item in enumerate(resp["results"], 1):
        print(f"  --- Result {i} ---")
        print(f"  Score:          {item['score']:.4f}")
        print(f"  Modality:       {item['modality']}")
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

        # GraphRAG results include context with source info
        ctx = item.get("context")
        if ctx and isinstance(ctx, dict):
            source = ctx.get("source", "")
            if source:
                print(f"  Context Source:  {source}")
            gctx = ctx.get("graphrag_context")
            if gctx and isinstance(gctx, dict):
                if gctx.get("entities"):
                    print(f"  Entities:       {len(gctx['entities'])} found")
                if gctx.get("community_reports"):
                    print(f"  Communities:    {len(gctx['community_reports'])} reports")
        print()


def submit_and_poll(payload: dict, timeout: int = 300) -> dict:
    """Submit a GraphRAG query asynchronously and poll until complete.

    Use this pattern for GraphRAG queries in production to avoid HTTP
    timeouts on long-running LLM calls.
    """
    # Step 1: Submit
    resp = requests.post(f"{BASE_URL}/retrieval/graphrag/submit", json=payload)
    resp.raise_for_status()
    job = resp.json()
    job_id = job["job_id"]
    print(f"  Submitted async job: {job_id}")

    # Step 2: Poll with exponential backoff (1s → 10s cap)
    delay = 1.0
    elapsed = 0.0
    while elapsed < timeout:
        time.sleep(delay)
        elapsed += delay

        status_resp = requests.get(
            f"{BASE_URL}/retrieval/graphrag/status/{job_id}"
        )
        status_resp.raise_for_status()
        status = status_resp.json()
        print(f"  Status: {status['status']} ({elapsed:.0f}s elapsed)")

        if status["status"] == "completed":
            # Step 3: Fetch result
            result_resp = requests.get(
                f"{BASE_URL}/retrieval/graphrag/result/{job_id}"
            )
            result_resp.raise_for_status()
            return result_resp.json()

        if status["status"] == "failed":
            print(f"  ERROR: {status.get('error', 'Unknown error')}")
            return {}

        delay = min(delay * 1.5, 10.0)

    print(f"  TIMEOUT after {timeout}s")
    return {}


# ---------------------------------------------------------------------------
# 1. Basic Text Query
# ---------------------------------------------------------------------------

def example_basic_text():
    """Basic BGE vector search over text chunks in Qdrant.

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
    """Full multi-modal pipeline: BGE text + CLIP image search, Neo4j graph
    expansion, ontology traversal, weighted fusion scoring, cross-encoder
    reranking.

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
# 3. GraphRAG Local Query
# ---------------------------------------------------------------------------

def example_graphrag_local():
    """Entity-centric search with community context reports.

    Finds entities matching the query and returns a detailed LLM-generated
    response enriched with community report context.

    - Input:    text query only
    - Output:   single LLM-generated response with entity/community context
    - Speed:    slow (30-90s, involves LLM calls)
    - Requires: at least one successful GraphRAG indexing cycle
    """
    print_header("3. GraphRAG Local Query (strategy=graphrag_local)")

    payload = {
        "query_text": "What are the key components of the S-300 air defense system?",
        "strategy": "graphrag_local",
        "top_k": 10,
    }

    # Synchronous (may timeout for long queries):
    print("  [Sync] Direct query:\n")
    try:
        resp = requests.post(QUERY_URL, json=payload, timeout=120)
        resp.raise_for_status()
        print_results(resp.json())
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 409:
            print("  GraphRAG indexing has not completed yet.")
            print("  Run: POST /v1/graphrag/index to trigger indexing.\n")
        else:
            raise

    # Async (recommended for production):
    print("  [Async] Submit/poll/fetch pattern:\n")
    result = submit_and_poll(payload)
    if result:
        print_results(result)


# ---------------------------------------------------------------------------
# 4. GraphRAG Global Query
# ---------------------------------------------------------------------------

def example_graphrag_global():
    """Cross-community summarization for broad, holistic questions.

    Analyzes all communities to synthesize a comprehensive answer that
    spans multiple topics/entity groups.

    - Input:    text query only
    - Output:   single LLM-generated multi-paragraph summary
    - Speed:    slow (30-90s)
    - Requires: at least one successful GraphRAG indexing cycle
    """
    print_header("4. GraphRAG Global Query (strategy=graphrag_global)")

    payload = {
        "query_text": "What are the major categories of air defense systems and how do they compare?",
        "strategy": "graphrag_global",
        "top_k": 10,
    }

    # Synchronous:
    print("  [Sync] Direct query:\n")
    try:
        resp = requests.post(QUERY_URL, json=payload, timeout=120)
        resp.raise_for_status()
        print_results(resp.json())
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 409:
            print("  GraphRAG indexing has not completed yet.\n")
        else:
            raise

    # Async (recommended):
    print("  [Async] Submit/poll/fetch pattern:\n")
    result = submit_and_poll(payload)
    if result:
        print_results(result)


# ---------------------------------------------------------------------------
# 5. GraphRAG Drift Query
# ---------------------------------------------------------------------------

def example_graphrag_drift():
    """Community-informed expansion search (Microsoft DRIFT algorithm).

    Iteratively expands the search across communities, balancing precision
    (entity-focused) with coverage (community-aware). Best for nuanced
    queries that benefit from progressive context gathering.

    - Input:    text query only
    - Output:   single LLM-generated in-depth analysis
    - Speed:    slowest (30-120s, iterative expansion)
    - Requires: at least one successful GraphRAG indexing cycle
    """
    print_header("5. GraphRAG Drift Query (strategy=graphrag_drift)")

    payload = {
        "query_text": "How do radar warning receivers interact with electronic countermeasure systems?",
        "strategy": "graphrag_drift",
        "top_k": 10,
    }

    # Synchronous:
    print("  [Sync] Direct query:\n")
    try:
        resp = requests.post(QUERY_URL, json=payload, timeout=180)
        resp.raise_for_status()
        print_results(resp.json())
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 409:
            print("  GraphRAG indexing has not completed yet.\n")
        else:
            raise

    # Async (recommended):
    print("  [Async] Submit/poll/fetch pattern:\n")
    result = submit_and_poll(payload)
    if result:
        print_results(result)


# ---------------------------------------------------------------------------
# 6. GraphRAG Basic Query
# ---------------------------------------------------------------------------

def example_graphrag_basic():
    """Vector search over GraphRAG-extracted text units.

    Simpler than Local/Global/Drift — performs semantic search over text
    units extracted during GraphRAG indexing without community context.
    Fastest GraphRAG method, works with partial indexing.

    - Input:    text query only
    - Output:   single concise LLM-generated answer from text units
    - Speed:    medium (5-15s)
    - Requires: minimal — works with partial GraphRAG indexing
    """
    print_header("6. GraphRAG Basic Query (strategy=graphrag_basic)")

    payload = {
        "query_text": "SA-2 Guideline missile specifications",
        "strategy": "graphrag_basic",
        "top_k": 10,
    }

    # Synchronous:
    print("  [Sync] Direct query:\n")
    try:
        resp = requests.post(QUERY_URL, json=payload, timeout=60)
        resp.raise_for_status()
        print_results(resp.json())
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 409:
            print("  GraphRAG indexing has not completed yet.\n")
        else:
            raise

    # Async:
    print("  [Async] Submit/poll/fetch pattern:\n")
    result = submit_and_poll(payload)
    if result:
        print_results(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXAMPLES = {
    "1": ("Basic Text",      example_basic_text),
    "2": ("Multi-Modal",     example_multi_modal),
    "3": ("GraphRAG Local",  example_graphrag_local),
    "4": ("GraphRAG Global", example_graphrag_global),
    "5": ("GraphRAG Drift",  example_graphrag_drift),
    "6": ("GraphRAG Basic",  example_graphrag_basic),
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
