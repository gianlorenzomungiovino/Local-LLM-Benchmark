"""Tests for evaluator.py."""

import pytest
from evaluator import (
    evaluate,
    evaluate_qa,
    evaluate_code,
    evaluate_reasoning,
    normalize_text,
)


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("Hello WORLD") == "hello world"

    def test_collapse_whitespace(self):
        assert normalize_text("hello   world") == "hello world"

    def test_strip(self):
        assert normalize_text("  hello  ") == "hello"

    def test_unicode_normalization(self):
        # NFKC normalization: fullwidth chars to halfwidth
        assert normalize_text("ｈｅｌｌｏ") == "hello"


class TestEvaluateQA:
    def test_exact_match(self):
        result = evaluate_qa("1 2 1", "1 2 1")
        assert result["score"] == 1.0
        assert "Exact" in result["notes"]

    def test_case_insensitive(self):
        result = evaluate_qa("1 2 1", "1 2 1")
        assert result["score"] == 1.0

    def test_partial_match(self):
        result = evaluate_qa("The output is 1 2 1", "1 2 1")
        assert result["score"] == 1.0

    def test_keyword_overlap(self):
        result = evaluate_qa("O(n) time complexity", "O(n)")
        assert result["score"] > 0

    def test_no_match(self):
        result = evaluate_qa("completely different answer", "expected output")
        assert result["score"] < 0.5

    def test_empty_expected(self):
        result = evaluate_qa("some output", "")
        assert result["score"] == 0.0


class TestEvaluateCode:
    def test_valid_code_with_keyword(self):
        output = "def validate_email(email):\n    return True"
        result = evaluate_code(output, "def validate_email")
        assert result["score"] > 0
        assert "AST valid" in result["notes"]

    def test_syntax_error(self):
        output = "def invalid code here ("
        result = evaluate_code(output, "def validate_email")
        assert result["score"] == 0.0
        assert "Syntax error" in result["notes"]

    def test_class_definition(self):
        output = "class LRUCache:\n    def __init__(self, n):\n        pass"
        result = evaluate_code(output, "class LRUCache")
        assert result["score"] > 0
        assert "AST valid" in result["notes"]

    def test_async_function(self):
        output = "async def fetch_with_retry(url):\n    return 'ok'"
        result = evaluate_code(output, "async def fetch_with_retry")
        assert result["score"] > 0

    def test_no_keyword_match(self):
        output = "def something_completely_different():\n    return 42"
        result = evaluate_code(output, "def validate_email")
        assert result["score"] < 0.5

    def test_quality_checks(self):
        output = """def validate_email(email: str) -> bool:
    '''Validate email address.'''
    if not email:
        raise ValueError("Empty email")
    return "@" in email
"""
        result = evaluate_code(output, "def validate_email")
        assert result["score"] > 0.5


class TestEvaluateReasoning:
    def test_exact_match(self):
        result = evaluate_reasoning("O(n)", "O(n)")
        assert result["score"] == 1.0

    def test_substring_match(self):
        result = evaluate_reasoning("The complexity is O(n) because we iterate once", "O(n)")
        assert result["score"] >= 0.9

    def test_keyword_overlap(self):
        result = evaluate_reasoning("This is O(n) time complexity with a single pass", "O(n)")
        assert result["score"] > 0.5

    def test_error_type_match(self):
        result = evaluate_reasoning(
            "This will raise a RecursionError because there's no base case for negative numbers",
            "RecursionError",
        )
        assert result["score"] >= 0.6

    def test_no_match(self):
        result = evaluate_reasoning("I don't know the answer", "O(n)")
        assert result["score"] < 0.3


class TestEvaluate:
    def test_code_type(self):
        result = evaluate("code", "def foo(): pass", "def foo")
        assert "score" in result
        assert "notes" in result

    def test_qa_type(self):
        result = evaluate("qa", "hello world", "hello")
        assert "score" in result

    def test_reasoning_type(self):
        result = evaluate("reasoning", "O(n)", "O(n)")
        assert result["score"] == 1.0

    def test_unknown_type(self):
        result = evaluate("unknown", "output", "expected")
        assert result["score"] == 0.0
        assert "Unknown" in result["notes"]
