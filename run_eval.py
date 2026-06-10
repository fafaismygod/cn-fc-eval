#!/usr/bin/env python3
"""
CN-FC-Eval: Chinese Function Calling Evaluation
Run evaluation on DeepSeek (and optionally other models).

Usage:
    python3 run_eval.py                          # Run on DeepSeek
    python3 run_eval.py --model gpt4o           # Run on GPT-4o
    python3 run_eval.py --quick                  # Quick test (10 cases)
    python3 run_eval.py --compare                # Compare all available models
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cn_fc_eval import (
    load_test_cases,
    get_categories,
    Evaluator,
    DeepSeekClient,
    OpenAIClient,
)
from cn_fc_eval.evaluator import compare_models

DATA_DIR = Path(__file__).parent
# Reuse same key pattern as Phase 1 and Phase 2
DEEPSEEK_KEY = os.environ.get("ANTHROPIC_AUTH_TOKEN", "sk-49276940eeaf4b6ba985efdde1488cc1")


def main():
    parser = argparse.ArgumentParser(description="CN-FC-Eval: Chinese Function Calling Evaluation")
    parser.add_argument("--model", type=str, default="deepseek",
                        choices=["deepseek", "gpt4o", "qwen", "glm", "all"],
                        help="Model to evaluate (default: deepseek)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with only 10 cases")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all available models")
    parser.add_argument("--category", type=str, default=None,
                        help="Only run specific category (simple/multi_step/parallel/nested/tool_selection)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    args = parser.parse_args()

    # Load test cases
    categories = [args.category] if args.category else None
    cases = load_test_cases(categories=categories)
    case_count = 10 if args.quick else (len(cases) if not args.compare else len(cases))
    cases = cases[:case_count]

    print(f"Loaded {len(cases)} test cases")
    if categories:
        print(f"Category filter: {args.category}")
    print()

    # Build clients
    clients = {}
    clients["deepseek"] = DeepSeekClient(api_key=DEEPSEEK_KEY)

    if os.environ.get("OPENAI_API_KEY"):
        clients["gpt4o"] = OpenAIClient(
            api_key=os.environ["OPENAI_API_KEY"],
            model="gpt-4o",
        )

    if args.compare or args.model == "all":
        # Compare all available
        available = [c for name, c in clients.items()]
        if not available:
            print("No API keys found. Set ANTHROPIC_AUTH_TOKEN or OPENAI_API_KEY.")
            sys.exit(1)
        results = compare_models(cases, available)
    else:
        client = clients.get(args.model)
        if client is None:
            print(f"Model '{args.model}' not available. Check API keys.")
            sys.exit(1)

        evaluator = Evaluator(client, verbose=True)
        results = evaluator.run(cases)
        evaluator.print_report(results)

    # Save results
    output_path = args.output or (DATA_DIR / "results.json")
    # Extract serializable summary
    if args.compare:
        serializable = {
            name: r["summary"]
            for name, r in results.items()
        }
    else:
        serializable = results["summary"]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
