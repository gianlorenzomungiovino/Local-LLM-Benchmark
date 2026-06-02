"""Deterministic scoring for benchmark tasks."""

from __future__ import annotations

import ast
import re
import unicodedata
from typing import Any


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace, strip unicode normalization."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text


def evaluate_qa(output: str, expected: str) -> dict[str, Any]:
    """Evaluate a QA task using substring match + keyword overlap.

    Returns:
        dict with score (0.0-1.0) and notes.
    """
    output_norm = normalize_text(output)
    expected_norm = normalize_text(expected)

    # Empty expected — no valid answer possible
    if not expected_norm:
        return {"score": 0.0, "notes": "Empty expected text"}

    # Exact substring match
    if expected_norm in output_norm:
        return {"score": 1.0, "notes": "Exact substring match"}

    # Partial substring match
    expected_words = set(expected_norm.split())
    output_words = set(output_norm.split())

    if not expected_words:
        return {"score": 0.0, "notes": "Empty expected text"}

    # Keyword overlap
    matched = expected_words & output_words
    keyword_score = len(matched) / len(expected_words)

    # Check if expected appears as a substring of any output word or vice versa
    partial_match = any(
        exp in out or out in exp
        for exp in expected_words
        for out in output_words
        if len(exp) > 3 and len(out) > 3
    )

    if partial_match:
        final_score = max(keyword_score, 0.5)
        return {
            "score": round(final_score, 2),
            "notes": f"Partial match — keyword overlap: {keyword_score:.2f}",
        }

    return {
        "score": round(keyword_score, 2),
        "notes": f"Keyword overlap only: {len(matched)}/{len(expected_words)}",
    }


def evaluate_code(output: str, expected: str) -> dict[str, Any]:
    """Evaluate a code task with AST validation + substring match + quality metrics.

    Returns:
        dict with score (0.0-1.0) and notes.
    """
    notes_parts = []
    ast_valid = False
    substring_score = 0.0
    quality_score = 0.0

    # 1. AST validation
    try:
        ast.parse(output)
        ast_valid = True
        notes_parts.append("AST valid")
    except SyntaxError:
        notes_parts.append("Syntax error")

    # 2. Substring / keyword match against expected
    output_norm = normalize_text(output)
    expected_norm = normalize_text(expected)

    if expected_norm in output_norm:
        substring_score = 1.0
        notes_parts.append("Expected keyword present")
    else:
        # Check for key identifiers
        expected_tokens = re.findall(r"[a-zA-Z_]\w*", expected_norm)
        output_tokens = set(re.findall(r"[a-zA-Z_]\w*", output_norm))
        if expected_tokens:
            matched = sum(1 for t in expected_tokens if t in output_tokens)
            substring_score = matched / len(expected_tokens)
            notes_parts.append(f"Keyword match: {substring_score:.2f}")
        else:
            substring_score = 0.0

    # 3. Quality metrics
    quality_checks = 0
    total_checks = 4

    # Has function/class definition
    if re.search(r"\b(def|class)\b", output):
        quality_checks += 1

    # Has reasonable length (not a one-liner for complex tasks)
    lines = [l.strip() for l in output.split("\n") if l.strip()]
    if len(lines) >= 3:
        quality_checks += 1

    # Has error handling or validation
    if re.search(r"\b(raise|try|except|if.*is None|if.*== None)\b", output):
        quality_checks += 1

    # Has type hints or docstring (bonus)
    if re.search(r":\s*(str|int|float|bool|list|dict|Optional|->)", output) or '"""' in output:
        quality_checks += 1

    quality_score = quality_checks / total_checks

    # Final score: weighted combination
    # AST validity is required for a meaningful score
    if not ast_valid:
        final_score = 0.0
        notes_parts.append("AST invalid — score 0")
    else:
        final_score = (
            substring_score * 0.4
            + quality_score * 0.3
            + (1.0 if substring_score > 0.5 else 0.0) * 0.3
        )
        final_score = min(final_score, 1.0)

    return {
        "score": round(final_score, 2),
        "notes": "; ".join(notes_parts),
    }


def evaluate_reasoning(output: str, expected: str) -> dict[str, Any]:
    """Evaluate a reasoning task using keyword overlap + logical check.

    Returns:
        dict with score (0.0-1.0) and notes.
    """
    output_norm = normalize_text(output)
    expected_norm = normalize_text(expected)

    # Exact match
    if expected_norm == output_norm:
        return {"score": 1.0, "notes": "Exact match"}

    # Substring match
    if expected_norm in output_norm:
        return {"score": 0.9, "notes": "Expected text contained in output"}

    # Keyword overlap
    expected_words = set(expected_norm.split())
    output_words = set(output_norm.split())

    if not expected_words:
        return {"score": 0.0, "notes": "Empty expected text"}

    matched = expected_words & output_words
    keyword_score = len(matched) / len(expected_words)

    # Bonus for related concepts
    # For complexity questions, check for O() notation patterns
    if "o(" in expected_norm or "o(" in output_norm:
        if "o(" in expected_norm and "o(" in output_norm:
            keyword_score = max(keyword_score, 0.7)

    # For error questions, check for error type names
    error_types = {"indexerror", "keyerror", "typeerror", "valueerror",
                   "recursionerror", "attributeerror", "importerror"}
    if error_types & expected_words:
        if error_types & output_words:
            keyword_score = max(keyword_score, 0.6)

    # Check for explanatory content (reasoning tasks should have explanation)
    has_explanation = len(output_words) > len(expected_words) * 2
    if has_explanation and keyword_score > 0.3:
        keyword_score = min(keyword_score + 0.1, 0.95)

    return {
        "score": round(keyword_score, 2),
        "notes": f"Keyword overlap: {len(matched)}/{len(expected_words)}",
    }


def evaluate(task_type: str, output: str, expected: str) -> dict[str, Any]:
    """Evaluate a task response based on its type.

    Args:
        task_type: One of 'code', 'qa', 'reasoning'.
        output: The model's output.
        expected: The expected answer.

    Returns:
        dict with 'score' (0.0-1.0) and 'notes' (str).
    """
    evaluators = {
        "code": evaluate_code,
        "qa": evaluate_qa,
        "reasoning": evaluate_reasoning,
    }

    evaluator = evaluators.get(task_type)
    if not evaluator:
        return {"score": 0.0, "notes": f"Unknown task type: {task_type}"}

    return evaluator(output, expected)
