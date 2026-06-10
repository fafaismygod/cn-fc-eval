"""Function calling evaluation logic."""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from .models import FunctionCallingClient
from .data import CATEGORY_NAMES


def _normalize_value(v: Any) -> Any:
    """Normalize a value for comparison (lowercase strings, etc.)."""
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, list):
        return [_normalize_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _normalize_value(val) for k, val in v.items()}
    return v


def _calculate_argument_score(expected: Dict, actual: Dict) -> float:
    """Calculate argument match score (0.0 to 1.0).

    Uses key-level precision: for each key in expected,
    check if it exists in actual and if its value matches.
    """
    if not expected:
        return 1.0 if not actual else 0.5  # no args expected, but model added some

    expected_norm = _normalize_value(expected)
    actual_norm = _normalize_value(actual)

    total_keys = len(expected_norm)
    matched_keys = 0

    for key, exp_val in expected_norm.items():
        if key in actual_norm:
            act_val = actual_norm[key]
            if isinstance(exp_val, dict) and isinstance(act_val, dict):
                # Recursive for nested objects
                sub_score = _calculate_argument_score(exp_val, act_val)
                matched_keys += sub_score
            elif isinstance(exp_val, list) and isinstance(act_val, list):
                # For arrays, check length and element overlap
                if len(exp_val) == len(act_val):
                    list_match = sum(
                        1 for i in range(len(exp_val))
                        if i < len(act_val) and exp_val[i] == act_val[i]
                    )
                    matched_keys += list_match / len(exp_val) if len(exp_val) > 0 else 1.0
                else:
                    # Partial credit for partial array match
                    overlap = sum(1 for e in exp_val if e in act_val)
                    matched_keys += (overlap / max(len(exp_val), len(act_val)))
            elif exp_val == act_val:
                matched_keys += 1.0
            # else: value mismatch, 0 credit for this key

    return matched_keys / total_keys if total_keys > 0 else 1.0


def _compare_single_call(expected: Dict, actual: Dict) -> Dict[str, Any]:
    """Compare a single expected function call with actual.

    Returns dict with 'tool_match', 'arg_score', 'details'.
    """
    if actual is None:
        return {
            "tool_match": False,
            "arg_score": 0.0,
            "expected_name": expected.get("name", ""),
            "actual_name": None,
            "details": "No function call made",
        }

    if isinstance(actual, dict) and "error" in actual:
        return {
            "tool_match": False,
            "arg_score": 0.0,
            "expected_name": expected.get("name", ""),
            "actual_name": None,
            "details": f"API error: {actual['error']}",
        }

    exp_name = expected.get("name", "")
    act_name = actual.get("name", "")
    tool_match = exp_name.lower() == act_name.lower() if act_name else False

    arg_score = 0.0
    if tool_match and "arguments" in expected:
        arg_score = _calculate_argument_score(
            expected["arguments"],
            actual.get("arguments", {}),
        )

    return {
        "tool_match": tool_match,
        "arg_score": arg_score,
        "expected_name": exp_name,
        "actual_name": act_name,
        "expected_args": expected.get("arguments", {}),
        "actual_args": actual.get("arguments", {}),
        "details": None,
    }


def _compare_parallel(expected_list: List[Dict], actual: Any) -> Dict[str, Any]:
    """Compare parallel function calls.

    Uses best-effort matching: for each expected call, find the best-matching
    actual call (by function name then arguments).
    """
    if not isinstance(actual, list):
        # Model returned single call instead of multiple
        return {
            "tool_match": False,
            "arg_score": 0.0,
            "expected_name": f"[{len(expected_list)} calls]",
            "actual_name": "single call" if actual else "none",
            "details": f"Expected {len(expected_list)} parallel calls, got {'1' if actual else '0'}",
        }

    total_tool_match = 0
    total_arg_score = 0.0
    individual_results = []

    # Greedy matching: for each expected, find best actual
    remaining_actuals = list(actual)
    for exp in expected_list:
        best_score = -1.0
        best_idx = -1
        best_result = None

        for i, act in enumerate(remaining_actuals):
            result = _compare_single_call(exp, act)
            score = (1.0 if result["tool_match"] else 0.0) + result["arg_score"]
            if score > best_score:
                best_score = score
                best_idx = i
                best_result = result

        if best_result:
            total_tool_match += 1 if best_result["tool_match"] else 0
            total_arg_score += best_result["arg_score"]
            individual_results.append(best_result)
            remaining_actuals.pop(best_idx)

    n = len(expected_list)
    return {
        "tool_match": total_tool_match / n,
        "arg_score": total_arg_score / n if n > 0 else 0.0,
        "expected_name": f"[{n} parallel calls]",
        "actual_name": f"[{len(actual)} calls]",
        "details": f"Matched {total_tool_match}/{n} tools",
        "individual_results": individual_results,
        "extra_calls": len(remaining_actuals),
    }


class Evaluator:
    """Evaluate function calling accuracy for a given LLM client."""

    def __init__(self, client: FunctionCallingClient, verbose: bool = True):
        self.client = client
        self.verbose = verbose

    def run(
        self,
        cases: List[Dict[str, Any]],
        delay: float = 0.5,
        max_cases: int = None,
    ) -> Dict[str, Any]:
        """Run evaluation on test cases.

        Args:
            cases: List of test case dicts from load_test_cases()
            delay: Delay between API calls in seconds
            max_cases: Limit to N cases (for quick tests)

        Returns:
            Results dict with per-category scores and detailed per-case results
        """
        if max_cases:
            cases = cases[:max_cases]

        per_case = []
        category_scores = defaultdict(lambda: {
            "total": 0,
            "tool_correct": 0,
            "arg_score_sum": 0.0,
            "cases": [],
        })

        for i, case in enumerate(cases):
            if self.verbose:
                print(f"[{i+1}/{len(cases)}] {case['id']}...", end=" ", flush=True)

            result = self._evaluate_one(case)
            per_case.append(result)

            cat = case["category"]
            category_scores[cat]["total"] += 1
            category_scores[cat]["tool_correct"] += result["tool_score"]
            category_scores[cat]["arg_score_sum"] += result["arg_score"]
            category_scores[cat]["cases"].append(result)

            if self.verbose:
                status = "✓" if result["tool_score"] >= 0.8 else "✗" if result["tool_score"] == 0 else "△"
                print(f"{status} (tool={result['tool_score']:.0%}, arg={result['arg_score']:.0%})")

            if delay and i < len(cases) - 1:
                time.sleep(delay)

        # Aggregate results
        total_tool = sum(r["tool_score"] for r in per_case)
        total_arg = sum(r["arg_score"] for r in per_case)
        n = len(per_case)

        summary = {
            "model": self.client.name,
            "total_cases": n,
            "tool_name_accuracy": total_tool / n if n > 0 else 0.0,
            "argument_accuracy": total_arg / n if n > 0 else 0.0,
            "by_category": {},
        }

        for cat in ["simple", "multi_step", "parallel", "nested", "tool_selection"]:
            if cat in category_scores:
                cs = category_scores[cat]
                t = cs["total"]
                summary["by_category"][cat] = {
                    "name": CATEGORY_NAMES.get(cat, cat),
                    "total": t,
                    "tool_name_accuracy": cs["tool_correct"] / t if t > 0 else 0.0,
                    "argument_accuracy": cs["arg_score_sum"] / t if t > 0 else 0.0,
                }

        return {
            "summary": summary,
            "per_case": per_case,
        }

    def _evaluate_one(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single test case."""
        expected = case["expected"]
        expected_type = case.get("expected_type", "single")
        tools = case["tools"]

        messages = [
            {"role": "system", "content": "你是一个 AI 助手，可以调用工具来完成任务。当用户提出请求时，请选择合适的工具并传入正确的参数。如果只需要调用一个工具，只调用那一个即可。如果需要多个工具，请同时调用。"},
            {"role": "user", "content": case["query"]},
        ]

        actual = self.client.call(messages, tools)

        # Compare based on expected type
        if expected_type == "parallel" or (isinstance(expected, list) and len(expected) > 1):
            comp = _compare_parallel(expected, actual)
        elif expected_type in ("multi_step_sequential", "multi_step_conditional"):
            # For multi-step, only check the first call
            if isinstance(expected, list):
                first_expected = expected[0]
            else:
                first_expected = expected
            first_actual = actual[0] if isinstance(actual, list) else actual
            comp = _compare_single_call(first_expected, first_actual)
        else:
            # Handle case where model returns multiple calls for single expected
            if isinstance(actual, list) and len(actual) > 0:
                actual = actual[0]
            comp = _compare_single_call(expected, actual)

        return {
            "id": case["id"],
            "category": case["category"],
            "query": case["query"],
            "tool_score": 1.0 if comp["tool_match"] else 0.0,
            "arg_score": comp["arg_score"],
            "expected_name": comp["expected_name"],
            "actual_name": comp["actual_name"],
            "expected_args": comp.get("expected_args", {}),
            "actual_args": comp.get("actual_args", {}),
            "details": comp.get("details"),
        }

    def print_report(self, results: Dict[str, Any]):
        """Print a formatted evaluation report."""
        s = results["summary"]

        print(f"\n{'='*70}")
        print(f"CN-FC-Eval 评估报告")
        print(f"{'='*70}")
        print(f"模型: {s['model']}")
        print(f"测试用例: {s['total_cases']}")
        print(f"\n总体成绩:")
        print(f"  工具名称准确率: {s['tool_name_accuracy']:.1%}")
        print(f"  参数匹配准确率: {s['argument_accuracy']:.1%}")

        print(f"\n{'分类':<16} {'用例数':>6} {'工具名':>8} {'参数':>8}")
        print(f"{'-'*42}")
        for cat, cs in s["by_category"].items():
            print(f"{cs['name']:<16} {cs['total']:>6} {cs['tool_name_accuracy']:>7.1%} {cs['argument_accuracy']:>7.1%}")

        # Failure mode analysis
        print(f"\n{'='*70}")
        print(f"错误分析 (工具名称不匹配的用例)")
        print(f"{'='*70}")
        failures = [r for r in results["per_case"] if r["tool_score"] == 0]
        if failures:
            for f in failures[:10]:  # Show top 10
                print(f"  [{f['id']}] {f['query'][:60]}...")
                print(f"    预期: {f['expected_name']} → 实际: {f['actual_name']}")
                if f['details']:
                    print(f"    备注: {f['details']}")
            if len(failures) > 10:
                print(f"  ... 以及另外 {len(failures) - 10} 个错误用例")
        else:
            print("  无错误用例！")

        # Category-specific insights
        print(f"\n{'='*70}")
        print(f"分类洞察")
        print(f"{'='*70}")
        for cat, cs in s["by_category"].items():
            cat_failures = [r for r in results["per_case"] if r["category"] == cat and r["tool_score"] < 1.0]
            if cat_failures:
                print(f"  {cs['name']}: {len(cat_failures)}/{cs['total']} 有问题")
            else:
                print(f"  {cs['name']}: 全部正确 ✓")


def compare_models(
    cases: List[Dict[str, Any]],
    clients: List[FunctionCallingClient],
    delay: float = 0.5,
) -> Dict[str, Any]:
    """Run evaluation across multiple models and compare results.

    Returns comparison dict suitable for table display.
    """
    all_results = {}
    for client in clients:
        print(f"\n{'#'*70}")
        print(f"# Evaluating: {client.name}")
        print(f"{'#'*70}")
        evaluator = Evaluator(client, verbose=False)
        all_results[client.name] = evaluator.run(cases, delay=delay)

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"模型对比: CN-FC-Eval")
    print(f"{'='*80}")
    header = f"{'模型':<24} {'工具名':>8} {'参数':>8} {'简单':>8} {'多步':>8} {'并行':>8} {'嵌套':>8} {'选择':>8}"
    print(header)
    print(f"{'-'*80}")

    for name, results in all_results.items():
        s = results["summary"]
        cats = s["by_category"]
        print(
            f"{name:<24} {s['tool_name_accuracy']:>7.1%} {s['argument_accuracy']:>7.1%} "
            f"{cats.get('simple', {}).get('tool_name_accuracy', 0):>7.1%} "
            f"{cats.get('multi_step', {}).get('tool_name_accuracy', 0):>7.1%} "
            f"{cats.get('parallel', {}).get('tool_name_accuracy', 0):>7.1%} "
            f"{cats.get('nested', {}).get('tool_name_accuracy', 0):>7.1%} "
            f"{cats.get('tool_selection', {}).get('tool_name_accuracy', 0):>7.1%}"
        )

    return all_results
