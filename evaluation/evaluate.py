"""
Evaluation Script for Medical Coding RAG System

Measures:
- Precision@K: What % of returned codes are correct
- Recall@K: What % of correct codes are returned
- MRR: Mean Reciprocal Rank
- Response Time: Average latency by mode
"""

import argparse
import json
import requests
import time
from typing import List, Dict, Tuple
from collections import defaultdict
import statistics


API_BASE_URL = "http://localhost:8000"


def load_test_cases(file_path: str = "test_cases.json") -> List[Dict]:
    """Load test cases from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def query_api(clinical_description: str, mode: str = "quick", max_results: int = 5) -> Dict:
    """Query the API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/code-suggestions",
            json={
                "clinical_description": clinical_description,
                "search_mode": mode,
                "max_results": max_results
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API returned {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def _normalize_code(code: str) -> str:
    """Strip formatting (dots) so 'E11.9' and 'E119' compare equal.

    ICD-10 source data is stored undotted (raw NCHS format), but test
    cases/clinicians often write codes in dotted clinical notation.
    """
    return code.replace(".", "").upper()


def calculate_precision_at_k(predicted: List[str], expected: List[str], k: int) -> float:
    """
    Precision@K: What percentage of top K predictions are correct?

    Example:
    predicted = ["E11.9", "E11.65", "I10", "R05"]
    expected = ["E11.9", "I10"]
    Precision@3 = 2/3 = 0.667 (2 correct out of 3 returned)
    """
    predicted_k = [_normalize_code(c) for c in predicted[:k]]
    expected_n = [_normalize_code(e) for e in expected]
    correct = sum(1 for code in predicted_k if any(exp in code or code in exp for exp in expected_n))
    return correct / k if k > 0 else 0


def calculate_recall_at_k(predicted: List[str], expected: List[str], k: int) -> float:
    """
    Recall@K: What percentage of expected codes are in top K?

    Example:
    predicted = ["E11.9", "E11.65", "I10"]
    expected = ["E11.9", "I10", "R05", "R06.02"]
    Recall@3 = 2/4 = 0.5 (found 2 out of 4 expected codes)
    """
    predicted_k = [_normalize_code(c) for c in predicted[:k]]
    expected_n = [_normalize_code(e) for e in expected]
    found = sum(1 for exp in expected_n if any(exp in code or code in exp for code in predicted_k))
    return found / len(expected) if expected else 0


def calculate_mrr(predicted: List[str], expected: List[str]) -> float:
    """
    Mean Reciprocal Rank: 1 / rank of first correct result

    Example:
    predicted = ["wrong1", "wrong2", "E11.9", "wrong3"]
    expected = ["E11.9"]
    MRR = 1/3 = 0.333 (first correct at position 3)
    """
    expected_n = [_normalize_code(e) for e in expected]
    for i, code in enumerate(predicted, 1):
        code_n = _normalize_code(code)
        if any(exp in code_n or code_n in exp for exp in expected_n):
            return 1 / i
    return 0


def evaluate_test_case(test_case: Dict, mode: str = "quick", k: int = 5) -> Dict:
    """Evaluate a single test case"""
    query = test_case["query"]
    expected_cpt = test_case.get("expected_cpt", [])
    expected_icd10 = test_case.get("expected_icd10", [])

    # Query API
    start_time = time.time()
    result = query_api(query, mode=mode, max_results=k)
    response_time = time.time() - start_time

    if "error" in result:
        return {
            "error": result["error"],
            "response_time": response_time
        }

    # Extract predicted codes
    predicted_cpt = [code["code"] for code in result.get("cpt_codes", [])]
    predicted_icd10 = [code["code"] for code in result.get("icd10_codes", [])]

    # Calculate metrics
    return {
        "query": query,
        "difficulty": test_case.get("difficulty", "unknown"),
        "response_time": response_time,
        "cpt_metrics": {
            "precision_at_k": calculate_precision_at_k(predicted_cpt, expected_cpt, k),
            "recall_at_k": calculate_recall_at_k(predicted_cpt, expected_cpt, k),
            "mrr": calculate_mrr(predicted_cpt, expected_cpt),
            "predicted": predicted_cpt[:3],  # Show top 3
            "expected": expected_cpt
        },
        "icd10_metrics": {
            "precision_at_k": calculate_precision_at_k(predicted_icd10, expected_icd10, k),
            "recall_at_k": calculate_recall_at_k(predicted_icd10, expected_icd10, k),
            "mrr": calculate_mrr(predicted_icd10, expected_icd10),
            "predicted": predicted_icd10[:3],
            "expected": expected_icd10
        }
    }


def print_results(results: List[Dict], mode: str, verbose: bool = False):
    """Print evaluation results"""
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS - {mode.upper()} MODE")
    print(f"{'='*80}\n")

    # Aggregate metrics
    cpt_precision = []
    cpt_recall = []
    cpt_mrr = []
    icd10_precision = []
    icd10_recall = []
    icd10_mrr = []
    response_times = []

    for result in results:
        if "error" not in result:
            cpt_precision.append(result["cpt_metrics"]["precision_at_k"])
            cpt_recall.append(result["cpt_metrics"]["recall_at_k"])
            cpt_mrr.append(result["cpt_metrics"]["mrr"])
            icd10_precision.append(result["icd10_metrics"]["precision_at_k"])
            icd10_recall.append(result["icd10_metrics"]["recall_at_k"])
            icd10_mrr.append(result["icd10_metrics"]["mrr"])
            response_times.append(result["response_time"])

    # Print summary metrics
    print("OVERALL METRICS\n")
    print(f"Total test cases: {len(results)}")
    print(f"Successful: {len(response_times)}")
    print(f"Failed: {len(results) - len(response_times)}\n")

    print("CPT Codes Performance:")
    print(f"  Precision@5: {statistics.mean(cpt_precision):.1%}")
    print(f"  Recall@5:    {statistics.mean(cpt_recall):.1%}")
    print(f"  MRR:         {statistics.mean(cpt_mrr):.3f}\n")

    print("ICD-10 Codes Performance:")
    print(f"  Precision@5: {statistics.mean(icd10_precision):.1%}")
    print(f"  Recall@5:    {statistics.mean(icd10_recall):.1%}")
    print(f"  MRR:         {statistics.mean(icd10_mrr):.3f}\n")

    print("⚡ RESPONSE TIME:")
    print(f"  Average: {statistics.mean(response_times)*1000:.0f}ms")
    print(f"  Min:     {min(response_times)*1000:.0f}ms")
    print(f"  Max:     {max(response_times)*1000:.0f}ms\n")

    # Print per-difficulty breakdown
    by_difficulty = defaultdict(lambda: {"cpt_p": [], "cpt_r": [], "icd10_p": [], "icd10_r": []})
    for result in results:
        if "error" not in result:
            diff = result["difficulty"]
            by_difficulty[diff]["cpt_p"].append(result["cpt_metrics"]["precision_at_k"])
            by_difficulty[diff]["cpt_r"].append(result["cpt_metrics"]["recall_at_k"])
            by_difficulty[diff]["icd10_p"].append(result["icd10_metrics"]["precision_at_k"])
            by_difficulty[diff]["icd10_r"].append(result["icd10_metrics"]["recall_at_k"])

    print("BY DIFFICULTY:\n")
    for difficulty in ["easy", "medium", "hard"]:
        if difficulty in by_difficulty:
            data = by_difficulty[difficulty]
            print(f"{difficulty.upper()}:")
            print(f"  CPT Precision: {statistics.mean(data['cpt_p']):.1%}")
            print(f"  ICD Precision: {statistics.mean(data['icd10_p']):.1%}")

    # Print some examples
    shown = results if verbose else results[:3]
    print(f"\n{'='*80}")
    label = "ALL TEST CASES" if verbose else "SAMPLE RESULTS (First 3 test cases)"
    print(f"{label}\n")

    for i, result in enumerate(shown, 1):
        if "error" in result:
            print(f"{i}. ERROR: {result['error']}\n")
            continue

        print(f"{i}. Query: {result['query'][:60]}...")
        print(f"   Difficulty: {result['difficulty']}")
        print(f"   Response time: {result['response_time']*1000:.0f}ms")
        print(f"   CPT - Precision: {result['cpt_metrics']['precision_at_k']:.1%}, Recall: {result['cpt_metrics']['recall_at_k']:.1%}")
        print(f"         Predicted: {result['cpt_metrics']['predicted']}")
        print(f"         Expected:  {result['cpt_metrics']['expected']}")
        print(f"   ICD - Precision: {result['icd10_metrics']['precision_at_k']:.1%}, Recall: {result['icd10_metrics']['recall_at_k']:.1%}")
        print(f"         Predicted: {result['icd10_metrics']['predicted']}")
        print(f"         Expected:  {result['icd10_metrics']['expected']}")
        print()


def main():
    """Run evaluation"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="test_cases.json", help="Path to a test cases JSON file")
    parser.add_argument("--verbose", action="store_true", help="Print per-query results for all test cases, not just the first 3")
    args = parser.parse_args()

    print("Medical Coding RAG System - Evaluation")
    print("=" * 80)

    # Check API health
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("ERROR: API is not responding. Please start the backend first.")
            return
        print("API is healthy\n")
    except:
        print("ERROR: Cannot connect to API. Please start the backend:")
        print("   cd backend && uvicorn app.main:app --reload")
        return

    # Load test cases
    test_cases = load_test_cases(args.file)
    print(f"Loaded {len(test_cases)} test cases from {args.file}\n")

    # Evaluate in quick mode
    print("Running evaluation in QUICK mode...")
    results_quick = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"  Testing {i}/{len(test_cases)}: {test_case['query'][:50]}...", end="\r")
        result = evaluate_test_case(test_case, mode="quick", k=5)
        results_quick.append(result)

    print_results(results_quick, "quick", verbose=args.verbose)

    # Optionally evaluate expert mode
    print(f"\n{'='*80}")
    response = input("\nRun evaluation in EXPERT mode? (y/n): ")
    if response.strip().lower() == 'y':
        print("\nRunning evaluation in EXPERT mode (will take longer)...")
        results_expert = []
        for i, test_case in enumerate(test_cases, 1):
            print(f"  Testing {i}/{len(test_cases)}: {test_case['query'][:50]}...", end="\r")
            result = evaluate_test_case(test_case, mode="expert", k=5)
            results_expert.append(result)

        print_results(results_expert, "expert", verbose=args.verbose)

    print(f"\n{'='*80}")
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
