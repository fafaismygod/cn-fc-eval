"""Test case data loader."""

import json
from pathlib import Path
from typing import List, Dict, Any

_DATA_DIR = Path(__file__).parent

CATEGORY_NAMES = {
    "simple": "简单调用",
    "multi_step": "多步调用",
    "parallel": "并行调用",
    "nested": "嵌套参数",
    "tool_selection": "工具选择推理",
}

CATEGORY_DESCRIPTIONS = {
    "simple": "单一工具调用，参数明确，查询意图直接",
    "multi_step": "需要先调用一个工具获取信息，再根据结果调用下一个工具",
    "parallel": "多个独立的工具调用可以同时进行，互不依赖",
    "nested": "参数包含复杂 JSON 嵌套结构（对象中嵌套对象/数组）",
    "tool_selection": "多个相似工具可用，需要推理选择最合适的一个",
}


def load_test_cases(categories: List[str] = None) -> List[Dict[str, Any]]:
    """Load test cases from the bundled JSON file.

    Args:
        categories: Optional list of categories to filter.
                   Any of: simple, multi_step, parallel, nested, tool_selection

    Returns:
        List of test case dicts, each with keys:
        - id, category, query, tools, expected
    """
    path = _DATA_DIR / "test_cases.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data["test_cases"]

    if categories:
        cases = [c for c in cases if c["category"] in categories]

    return cases


def get_categories() -> Dict[str, Dict[str, str]]:
    """Return category metadata."""
    return {
        key: {
            "name": CATEGORY_NAMES.get(key, key),
            "description": CATEGORY_DESCRIPTIONS.get(key, ""),
        }
        for key in CATEGORY_NAMES
    }
