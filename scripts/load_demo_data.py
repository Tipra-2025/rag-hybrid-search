"""Demo data loader for PRECISION RAG.

Creates a "Acme HR Policies" knowledge base and ingests all 5 demo documents.

Usage:
    # Make sure the FastAPI server is running first:
    #   rag serve   (in a separate terminal)
    #
    python scripts/load_demo_data.py
    # or with a custom API URL:
    python scripts/load_demo_data.py --api-url http://localhost:8100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests


def main(api_url: str = "http://localhost:8100") -> None:
    demo_dir = Path(__file__).parent.parent / "demo_docs"
    if not demo_dir.exists():
        print(f"ERROR: demo_docs directory not found at {demo_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  PRECISION RAG — Demo Data Loader")
    print(f"{'='*60}")
    print(f"  API: {api_url}")
    print(f"  Demo docs: {demo_dir}")
    print()

    # Check API health
    try:
        r = requests.get(f"{api_url}/health", timeout=5)
        health = r.json()
        print(f"  API status: {health.get('status', 'unknown')}")
    except Exception as e:
        print(f"ERROR: Cannot reach API at {api_url}: {e}")
        print("  Make sure the backend is running: rag serve")
        sys.exit(1)

    # Create knowledge base
    print("\n[1] Creating knowledge base: 'Acme HR Policies'...")
    r = requests.post(
        f"{api_url}/knowledge-bases",
        json={
            "name": "Acme HR Policies",
            "description": (
                "Employee handbook, leave policy, IT security policy, "
                "expense policy, and travel policy for Acme Corporation."
            ),
        },
        timeout=10,
    )
    if r.status_code not in (200, 201):
        print(f"  ERROR creating KB: {r.text}")
        sys.exit(1)

    kb = r.json()
    kb_id = kb["id"]
    print(f"  Created KB: {kb['name']} (id={kb_id})")

    # Ingest documents
    docs = sorted(demo_dir.glob("*.md"))
    print(f"\n[2] Ingesting {len(docs)} documents...")

    success = 0
    failures = 0
    for doc_path in docs:
        print(f"  → {doc_path.name} ...", end=" ", flush=True)
        with doc_path.open("rb") as f:
            file_bytes = f.read()

        r = requests.post(
            f"{api_url}/knowledge-bases/{kb_id}/documents",
            files={"file": (doc_path.name, file_bytes, "text/markdown")},
            timeout=120,
        )
        if r.status_code in (200, 201):
            result = r.json()
            if result.get("success"):
                chunks = result.get("chunks_added", result.get("chunk_count", "?"))
                print(f"OK ({chunks} chunks indexed)")
                success += 1
            else:
                print(f"SKIPPED — {result.get('message', result.get('error', 'unknown'))}")
                success += 1  # skips are not failures
        else:
            print(f"FAILED — {r.text[:100]}")
            failures += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"  Done: {success} documents loaded, {failures} failed")
    if failures == 0:
        print(f"  KB '{kb['name']}' is ready to query!")
        print(f"\n  Open the UI: streamlit run src/rag/ui.py")
        print(f"  Or use the API: POST {api_url}/v1/ask")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load PRECISION RAG demo data")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8100",
        help="URL of the running PRECISION RAG API (default: http://localhost:8100)",
    )
    args = parser.parse_args()
    main(api_url=args.api_url)
