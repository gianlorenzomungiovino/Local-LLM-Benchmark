"""Evaluator module — scores benchmark results against expected keywords."""

from __future__ import annotations

import re
from typing import Any


def score_result(result: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    """Score a single result entry against its task definition.

    Scoring rules:
    - **code** tasks: check for structural elements (``def ``, type hints
      ``:`` with space, docstrings using triple-quotes, and any
      expected_keywords).  Score = structural_matches / (len(expected_keywords)
      + 3) if keywords exist, else 0.5.
    - **qa** / **reasoning** tasks: case-insensitive substring match of each
      expected keyword against the response text.  Score = matched / total.
    - **empty** expected_keywords: neutral default of 0.5.

    Args:
        result: A result dict with at least ``response`` and ``task_type``.
        task: A task dict with at least ``type`` and ``expected_keywords``.

    Returns:
        The result dict with ``score`` (float 0.0-1.0) and
        ``matched_keywords`` (list[str]) populated.
    """
    response = result.get("response", "") or ""
    task_type = task.get("type", result.get("task_type", "qa"))
    expected_keywords: list[str] = task.get("expected_keywords") or []

    matched_keywords: list[str] = []

    if not expected_keywords:
        # Neutral default when no keywords are specified
        result["score"] = 0.5
        result["matched_keywords"] = []
        return result

    if task_type == "code":
        # Structural elements to check
        structural_checks = [
            ("def ", "def "),
            ("type_hint", r":\s"),
            ("docstring", '"""'),
        ]
        structural_matches = 0
        for _label, pattern in structural_checks:
            if re.search(pattern, response):
                structural_matches += 1

        # Also check expected keywords as structural hints
        keyword_matches = 0
        for kw in expected_keywords:
            if kw.lower() in response.lower():
                keyword_matches += 1
                matched_keywords.append(kw)

        total_denominator = len(expected_keywords) + 3
        result["score"] = round((structural_matches + keyword_matches) / total_denominator, 4)
        result["matched_keywords"] = matched_keywords
        return result

    # qa or reasoning: case-insensitive substring match
    response_lower = response.lower()
    for kw in expected_keywords:
        if kw.lower() in response_lower:
            matched_keywords.append(kw)

    result["score"] = round(len(matched_keywords) / len(expected_keywords), 4)
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
