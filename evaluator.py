"""Evaluator module — scores benchmark results against expected keywords and test cases."""

from __future__ import annotations

import re
from typing import Any


def _score_code_structural(response: str) -> tuple[int, int]:
    """Score structural elements of code response.

    Returns (matches, total_checks).
    """
    checks = [
        ("def ", r"\bdef\b"),
        ("type_hint", r":[ \t]+[A-Z\w\[\]\{\},\s]+"),
        ("docstring", '"""[^"]+"""\s*'),
        ("return", r"\breturn\b"),
        ("error_handling", r"(try|except|raise|ValueError|TypeError|KeyError)"),
        ("typing_import", r"(from typing import|import typing)"),
    ]
    matches = sum(1 for _label, pattern in checks if re.search(pattern, response))
    return matches, len(checks)


def _score_code_keywords(response: str, keywords: list[str]) -> tuple[int, list[str]]:
    """Check for expected keywords in response.

    Returns (matched_count, matched_list).
    """
    matched = []
    response_lower = response.lower()
    for kw in keywords:
        if kw.lower() in response_lower:
            matched.append(kw)
    return len(matched), matched


def score_result(result: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    """Score a single result entry against its task definition.

    Scoring rules by task type:
    - **code_execute**: structural checks + keyword matching.
      Score = (structural_matches + keyword_matches) / total_checks.
      Also checks for test_case_coverage if test_cases are provided.
    - **code_refactoring**: structural checks + keyword matching +
      checks for explanation quality.
    - **code_debug**: structural checks + keyword matching +
      checks for bug identification.
    - **reasoning_math**: keyword matching + numeric verification.
      If answer_range or answer is provided, numeric comparison is used.
    - **reasoning_science**: keyword matching + text quality check.
    - **reasoning_logic**: keyword matching + reasoning quality.
    - **instruction_following**: keyword matching + constraint verification.
    - **qa_knowledge**: keyword matching.

    Args:
        result: A result dict with at least ``response`` and ``task_type``.
        task: A task dict with at least ``type`` and ``expected_keywords``.

    Returns:
        The result dict with ``score`` (float 0.0-1.0) and
        ``matched_keywords`` (list[str]) populated.
    """
    response = result.get("response", "") or ""
    task_type = task.get("type", result.get("task_type", "qa_knowledge"))
    expected_keywords: list[str] = task.get("expected_keywords") or []

    matched_keywords: list[str] = []

    # --- code_execute ---
    if task_type == "code_execute":
        structural_matches, structural_total = _score_code_structural(response)
        keyword_matches, matched_keywords = _score_code_keywords(response, expected_keywords)

        # Bonus for test case coverage if test_cases are provided
        test_score = 0.0
        test_cases = task.get("test_cases", [])
        if test_cases:
            # Check if response mentions key algorithmic patterns
            response_lower = response.lower()
            algorithmic_terms = [
                "dict", "set", "list", "array", "hash", "map",
                "queue", "stack", "tree", "heap", "binary",
                "sort", "search", "traverse", "recurs", "iter",
                "dynamic", "memo", "cache", "dp",
                "bfs", "dfs", "greedy", "divide", "conquer",
            ]
            algo_matches = sum(1 for term in algorithmic_terms if term in response_lower)
            test_score = min(algo_matches / 5.0, 1.0)  # Up to 1.0 for 5+ algorithmic terms

        total_denominator = structural_total + len(expected_keywords)
        # Combine structural (50%), keywords (30%), and algorithmic patterns (20%)
        result["score"] = round(
            (structural_matches * 0.5 / structural_total +
             keyword_matches * 0.3 / max(len(expected_keywords), 1) +
             test_score * 0.2),
            4
        )
        result["matched_keywords"] = matched_keywords
        result["test_score"] = round(test_score, 4)
        return result

    # --- code_refactoring ---
    if task_type == "code_refactoring":
        structural_matches, structural_total = _score_code_structural(response)
        keyword_matches, matched_keywords = _score_code_keywords(response, expected_keywords)

        # Check for explanation quality
        explanation_indicators = ["because", "therefore", "improves", "better", "since", "thus"]
        explanation_score = sum(1 for word in explanation_indicators if word in response.lower()) / len(explanation_indicators)

        total_denominator = structural_total + len(expected_keywords)
        result["score"] = round(
            (structural_matches * 0.4 / structural_total +
             keyword_matches * 0.4 / max(len(expected_keywords), 1) +
             explanation_score * 0.2),
            4
        )
        result["matched_keywords"] = matched_keywords
        return result

    # --- code_debug ---
    if task_type == "code_debug":
        structural_matches, structural_total = _score_code_structural(response)
        keyword_matches, matched_keywords = _score_code_keywords(response, expected_keywords)

        # Check for bug identification language
        bug_indicators = ["bug", "error", "fix", "correct", "issue", "problem", "wrong", "off-by-one"]
        bug_score = sum(1 for word in bug_indicators if word in response.lower()) / len(bug_indicators)

        total_denominator = structural_total + len(expected_keywords)
        result["score"] = round(
            (structural_matches * 0.3 / structural_total +
             keyword_matches * 0.4 / max(len(expected_keywords), 1) +
             bug_score * 0.3),
            4
        )
        result["matched_keywords"] = matched_keywords
        return result

    # --- instruction_following ---
    if task_type == "instruction_following":
        structural_matches, structural_total = _score_code_structural(response)
        keyword_matches, matched_keywords = _score_code_keywords(response, expected_keywords)

        # Check constraint compliance
        constraints = task.get("constraints", [])
        constraint_score = 0.0
        if constraints:
            response_lower = response.lower()
            constraint_checks = {
                "no_comments": lambda r: "#" not in r.split('"""')[0] if '"""' in r else True,
                "no_imports": lambda r: "import " not in r,
                "no_loops": lambda r: "for " not in r and "while " not in r,
                "no_numpy": lambda r: "numpy" not in r and "np." not in r,
                "no_comprehensions": lambda r: "[" not in r.split("def")[1].split(":")[1] if "def" in r else True,
                "uses_recursion": lambda r: re.search(r"\b\w+\([^)]*\+\s*\w+|[^)]*-\s*\w+", r) is not None,
            }
            satisfied = sum(
                1 for c in constraints
                if c.get("type") in constraint_checks and constraint_checks[c["type"]](response)
            )
            constraint_score = satisfied / len(constraints)

        total_denominator = structural_total + len(expected_keywords)
        result["score"] = round(
            (structural_matches * 0.3 / structural_total +
             keyword_matches * 0.3 / max(len(expected_keywords), 1) +
             constraint_score * 0.4),
            4
        )
        result["matched_keywords"] = matched_keywords
        result["constraint_score"] = round(constraint_score, 4)
        return result

    # --- reasoning_math ---
    if task_type == "reasoning_math":
        response_lower = response.lower()
        for kw in expected_keywords:
            if kw.lower() in response_lower:
                matched_keywords.append(kw)

        keyword_score = len(matched_keywords) / max(len(expected_keywords), 1)

        # Numeric verification if available
        numeric_score = 0.0
        verification = task.get("verification")
        if verification == "numeric_exact":
            expected_answer = task.get("answer", 0)
            # Try to extract a number from the response
            numbers = re.findall(r'\b(\d+)\b', response)
            if numbers:
                # Use the last significant number found
                last_num = int(numbers[-1])
                if last_num == expected_answer:
                    numeric_score = 1.0
                elif abs(last_num - expected_answer) <= expected_answer * 0.1:
                    numeric_score = 0.5
        elif verification == "numeric_range":
            answer_range = task.get("answer_range", [0, float('inf')])
            numbers = re.findall(r'\b(\d+)\b', response)
            if numbers:
                last_num = int(numbers[-1])
                if answer_range[0] <= last_num <= answer_range[1]:
                    numeric_score = 1.0

        # Combine keyword and numeric scores
        result["score"] = round(keyword_score * 0.5 + numeric_score * 0.5, 4)
        result["matched_keywords"] = matched_keywords
        result["numeric_score"] = round(numeric_score, 4)
        return result

    # --- reasoning_science ---
    if task_type == "reasoning_science":
        response_lower = response.lower()
        for kw in expected_keywords:
            if kw.lower() in response_lower:
                matched_keywords.append(kw)

        keyword_score = len(matched_keywords) / max(len(expected_keywords), 1)

        # Check for scientific reasoning indicators
        reasoning_indicators = [
            "therefore", "because", "thus", "implies", "consequently",
            "derive", "calculate", "formula", "equation", "principle",
            "mechanism", "process", "explain", "demonstrate"
        ]
        reasoning_score = sum(1 for word in reasoning_indicators if word in response_lower) / len(reasoning_indicators)

        # Check response length (deeper explanations score higher)
        word_count = len(response.split())
        length_score = min(word_count / 100, 1.0)  # 100+ words = max score

        result["score"] = round(
            keyword_score * 0.4 + reasoning_score * 0.3 + length_score * 0.3,
            4
        )
        result["matched_keywords"] = matched_keywords
        return result

    # --- reasoning_logic ---
    if task_type == "reasoning_logic":
        response_lower = response.lower()
        for kw in expected_keywords:
            if kw.lower() in response_lower:
                matched_keywords.append(kw)

        keyword_score = len(matched_keywords) / max(len(expected_keywords), 1)

        # Check for step-by-step reasoning indicators
        step_indicators = [
            "step", "first", "second", "third", "next", "then",
            "therefore", "conclusion", "thus", "implies", "if and only if",
            "suppose", "assume", "contradiction", "wlog"
        ]
        step_score = sum(1 for word in step_indicators if word in response_lower) / len(step_indicators)

        result["score"] = round(
            keyword_score * 0.5 + step_score * 0.5,
            4
        )
        result["matched_keywords"] = matched_keywords
        return result

    # --- qa_knowledge (default) ---
    response_lower = response.lower()
    for kw in expected_keywords:
        if kw.lower() in response_lower:
            matched_keywords.append(kw)

    result["score"] = round(len(matched_keywords) / max(len(expected_keywords), 1), 4)
    result["matched_keywords"] = matched_keywords
    return result


def score_results(
    results: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score all results by matching them to their tasks.

    Args:
        results: List of result dicts (each with a ``task_id`` field).
        tasks: List of task dicts (each with an ``id`` field).

    Returns:
        The same results list, mutated in-place with ``score`` and
        ``matched_keywords`` populated.
    """
    task_lookup: dict[int, dict[str, Any]] = {t["id"]: t for t in tasks}

    for result in results:
        task_id = result.get("task_id")
        task = task_lookup.get(task_id)
        if task is not None:
            score_result(result, task)
        else:
            # Unknown task — default neutral score
            result["score"] = 0.5
            result["matched_keywords"] = []

    return results


def get_category_scores(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute average scores by task type and difficulty.

    Returns:
        Dict mapping task_type -> {
            "average": float,
            "count": int,
            "by_difficulty": {"easy": float, "medium": float, "hard": float}
        }
    """
    categories: dict[str, dict] = {}

    for result in results:
        task_type = result.get("task_type", "unknown")
        score = result.get("score", 0.0)
        difficulty = result.get("difficulty", "medium")

        if task_type not in categories:
            categories[task_type] = {
                "scores": [],
                "by_difficulty": {"easy": [], "medium": [], "hard": []},
            }

        categories[task_type]["scores"].append(score)
        if difficulty in categories[task_type]["by_difficulty"]:
            categories[task_type]["by_difficulty"][difficulty].append(score)

    output = {}
    for ttype, data in categories.items():
        avg = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0.0
        by_diff = {}
        for diff, scores in data["by_difficulty"].items():
            by_diff[diff] = sum(scores) / len(scores) if scores else 0.0
        output[ttype] = {
            "average": round(avg, 4),
            "count": len(data["scores"]),
            "by_difficulty": by_diff,
        }

    return output
