"""
CN-FC-Eval: Chinese Function Calling Evaluation

A benchmark for evaluating Chinese function calling capabilities
across LLM providers. Contains 100 test cases in 5 categories.

Usage:
    from cn_fc_eval import load_test_cases, Evaluator, DeepSeekClient

    cases = load_test_cases()
    client = DeepSeekClient(api_key="...")
    evaluator = Evaluator(client)
    results = evaluator.run(cases)
    evaluator.print_report(results)
"""

from .data import load_test_cases, get_categories
from .evaluator import Evaluator
from .models import DeepSeekClient, OpenAIClient

__version__ = "0.1.0"
__all__ = [
    "load_test_cases",
    "get_categories",
    "Evaluator",
    "DeepSeekClient",
    "OpenAIClient",
]
