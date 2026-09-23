"""
Interactive CLI for spot-checking the medical coding RAG pipeline.

Enter keywords/a clinical description, see the CPT and ICD-10 codes
the API returns. Useful for eyeballing retrieval quality with a given
embedding model (e.g. voyage-4-lite) before running the full evaluation.

Usage:
    python evaluation/query_cli.py
    python evaluation/query_cli.py --mode standard --max-results 10
"""

import argparse
import time

import requests


def query_api(api_url: str, clinical_description: str, mode: str, max_results: int) -> dict:
    """Query the code-suggestions API"""
    try:
        response = requests.post(
            f"{api_url}/api/code-suggestions",
            json={
                "clinical_description": clinical_description,
                "search_mode": mode,
                "max_results": max_results,
            },
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"API returned {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}


def print_codes(label: str, codes: list):
    if not codes:
        print(f"  {label}: (none)")
        return
    print(f"  {label}:")
    for c in codes:
        code = c.get("code", "?")
        description = c.get("description", "")
        score = c.get("confidence_score")
        score_str = f" (score: {score:.3f})" if isinstance(score, (int, float)) else ""
        print(f"    {code:<10} {description[:70]}{score_str}")


def main():
    parser = argparse.ArgumentParser(description="Interactive keyword -> medical codes CLI")
    parser.add_argument("--mode", default="quick", choices=["quick", "standard", "expert"], help="Search mode")
    parser.add_argument("--max-results", type=int, default=5, help="Max results per code type")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Base URL of the running API")
    args = parser.parse_args()

    print("Medical Coding RAG - Interactive Query CLI")
    print(f"Mode: {args.mode} | Max results: {args.max_results} | API: {args.api_url}")
    print("Type a clinical description or keywords, or 'quit' to exit.\n")

    try:
        response = requests.get(f"{args.api_url}/health", timeout=5)
        health = response.json() if response.status_code == 200 else {}
        print(f"API health: {health.get('status', 'unknown')} | embedding model: {health.get('embedding_model', 'unknown')}\n")
    except Exception as e:
        print(f"WARNING: could not reach API health check ({e}). Continuing anyway.\n")

    while True:
        try:
            query = input("keywords> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break

        start = time.time()
        result = query_api(args.api_url, query, args.mode, args.max_results)
        elapsed_ms = (time.time() - start) * 1000

        if "error" in result:
            print(f"  ERROR: {result['error']}\n")
            continue

        print(f"  ({elapsed_ms:.0f}ms)")
        print_codes("CPT", result.get("cpt_codes", []))
        print_codes("ICD-10", result.get("icd10_codes", []))
        print()


if __name__ == "__main__":
    main()
