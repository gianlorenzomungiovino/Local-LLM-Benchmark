"""Comprehensive tests for the evaluator module."""

from evaluator import score_result, score_results


# ---------------------------------------------------------------------------
# score_result — keyword (qa / reasoning) tests
# ---------------------------------------------------------------------------

def test_keyword_match_exact():
    """Exact keyword match returns score=1.0."""
    result = {"response": "The time complexity is O(n log n) average case"}
    task = {"type": "qa", "expected_keywords": ["o(n log n)"]}
    score_result(result, task)
    assert result["score"] == 1.0
    assert result["matched_keywords"] == ["o(n log n)"]


def test_keyword_match_partial():
    """Partial / substring match counts."""
    result = {"response": "A generator is an iterator that yields values"}
    task = {"type": "qa", "expected_keywords": ["generator", "iterator"]}
    score_result(result, task)
    assert result["score"] == 1.0
    assert "generator" in result["matched_keywords"]
    assert "iterator" in result["matched_keywords"]


def test_keyword_match_case_insensitive():
    """Case differences don't affect matching."""
    result = {"response": "The Yield keyword creates a generator object"}
    task = {"type": "qa", "expected_keywords": ["yield", "GENERATOR"]}
    score_result(result, task)
    assert result["score"] == 1.0
    assert "yield" in result["matched_keywords"]
    assert "GENERATOR" in result["matched_keywords"]


def test_keyword_match_none():
    """Empty expected_keywords returns score=0.5 (neutral default)."""
    result = {"response": "anything at all"}
    task = {"type": "qa", "expected_keywords": []}
    score_result(result, task)
    assert result["score"] == 0.5
    assert result["matched_keywords"] == []


def test_keyword_match_all_miss():
    """No keywords match returns score=0.0."""
    result = {"response": "The answer is completely unrelated to the question"}
    task = {"type": "qa", "expected_keywords": ["python", "generator", "yield"]}
    score_result(result, task)
    assert result["score"] == 0.0
    assert result["matched_keywords"] == []


# ---------------------------------------------------------------------------
# score_result — code task tests
# ---------------------------------------------------------------------------

def test_code_scoring_with_keywords():
    """Code task with expected keywords scores based on structural + keyword matches."""
    result = {"response": 'def reverse_linked_list(head):\n    """Reverse a linked list."""\n    pass'}
    task = {"type": "code", "expected_keywords": ["reverse", "linked", "list"]}
    score_result(result, task)
    # Structural: def (1), type_hint (1), docstring (1) = 3
    # Keywords: "reverse" found, "linked" found, "list" found = 3
    # score = (2 + 3) / (3 + 3) = 0.8333
    assert result["score"] == round(5 / 6, 4)
    assert "reverse" in result["matched_keywords"]
    assert "linked" in result["matched_keywords"]
    assert "list" in result["matched_keywords"]


def test_code_scoring_no_keywords():
    """Code task without keywords scores on structural elements only."""
    result = {"response": "def foo():\n    pass"}
    task = {"type": "code", "expected_keywords": []}
    score_result(result, task)
    assert result["score"] == round(1 / 3, 4)
    assert result["matched_keywords"] == []


def test_code_scoring_structural_only():
    """Code task with keywords but no keyword matches still gets structural score."""
    result = {"response": 'def binary_search(arr, target):\n    """Binary search implementation."""\n    return -1'}
    task = {"type": "code", "expected_keywords": ["nonexistent_kw"]}
    score_result(result, task)
    # Structural: def (1), type_hint (1), docstring (1) = 3
    # Keywords: 0 matches
    # score = 2 / (1 + 3) = 0.5
    assert result["score"] == 0.5


def test_code_scoring_no_structure_no_keywords():
    """Code task with no structure and no keyword matches scores 0."""
    result = {"response": "just a plain string with no code"}
    task = {"type": "code", "expected_keywords": ["no_match"]}
    score_result(result, task)
    # Structural: 0, Keywords: 0
    # score = 0 / (1 + 3) = 0.0
    assert result["score"] == 0.0


# ---------------------------------------------------------------------------
# score_result — reasoning task tests
# ---------------------------------------------------------------------------

def test_reasoning_keyword_match():
    """Reasoning task scores via keyword matching."""
    result = {"response": "No, we cannot conclude that. It is not necessarily true."}
    task = {"type": "reasoning", "expected_keywords": ["no", "not necessarily"]}
    score_result(result, task)
    assert result["score"] == 1.0
    assert "no" in result["matched_keywords"]
    assert "not necessarily" in result["matched_keywords"]


def test_reasoning_partial_match():
    """Reasoning task with partial keyword match."""
    result = {"response": "The answer is 2 hours after the second train leaves."}
    task = {"type": "reasoning", "expected_keywords": ["2", "hours", "catch", "distance"]}
    score_result(result, task)
    # 2 matched out of 4 = 0.5
    assert result["score"] == 0.5
    assert "2" in result["matched_keywords"]
    assert "hours" in result["matched_keywords"]
    assert "catch" not in result["matched_keywords"]
    assert "distance" not in result["matched_keywords"]


# ---------------------------------------------------------------------------
# score_results — batch tests
# ---------------------------------------------------------------------------

def test_score_results_batch():
    """score_results processes multiple results correctly."""
    results = [
        {"task_id": 1, "response": "O(n log n) is the average case"},
        {"task_id": 2, "response": "completely wrong answer"},
    ]
    tasks = [
        {"id": 1, "type": "qa", "expected_keywords": ["o(n log n)"]},
        {"id": 2, "type": "qa", "expected_keywords": ["binary", "search"]},
    ]
    score_results(results, tasks)
    assert results[0]["score"] == 1.0
    assert results[1]["score"] == 0.0


def test_score_results_preserves_fields():
    """Original fields (run_id, task_id, prompt, response) are preserved."""
    results = [
        {
            "run_id": "abc-123",
            "task_id": 1,
            "task_type": "qa",
            "prompt": "What is quicksort?",
            "response": "O(n log n) average",
            "score": None,
            "timestamp": "2026-01-01T00:00:00Z",
        }
    ]
    tasks = [
        {"id": 1, "type": "qa", "expected_keywords": ["o(n log n)"]},
    ]
    score_results(results, tasks)
    assert results[0]["run_id"] == "abc-123"
    assert results[0]["task_id"] == 1
    assert results[0]["prompt"] == "What is quicksort?"
    assert results[0]["response"] == "O(n log n) average"
    assert results[0]["timestamp"] == "2026-01-01T00:00:00Z"
    assert results[0]["score"] == 1.0
    assert "matched_keywords" in results[0]


def test_score_results_task_lookup():
    """Results are matched to tasks by task_id."""
    results = [
        {"task_id": 10, "response": "yield creates a generator"},
        {"task_id": 9, "response": "quicksort is O(n log n)"},
    ]
    tasks = [
        {"id": 9, "type": "qa", "expected_keywords": ["o(n log n)"]},
        {"id": 10, "type": "qa", "expected_keywords": ["yield", "generator"]},
    ]
    score_results(results, tasks)
    # Task 10: both keywords matched → 1.0
    assert results[0]["score"] == 1.0
    # Task 9: keyword matched → 1.0
    assert results[1]["score"] == 1.0


def test_score_results_unknown_task():
    """Results with unknown task_id get neutral score 0.5."""
    results = [
        {"task_id": 999, "response": "orphan result"},
    ]
    tasks = [
        {"id": 1, "type": "qa", "expected_keywords": ["foo"]},
    ]
    score_results(results, tasks)
    assert results[0]["score"] == 0.5
    assert results[0]["matched_keywords"] == []


def test_score_results_empty_list():
    """Empty results list returns empty list."""
    result = score_results([], [])
    assert result == []


def test_score_results_mixed_types():
    """score_results handles mixed task types (code, qa, reasoning)."""
    results = [
        {"task_id": 1, "response": "def foo():\n    pass"},
        {"task_id": 2, "response": "O(n log n) average"},
        {"task_id": 3, "response": "No, not necessarily true"},
    ]
    tasks = [
        {"id": 1, "type": "code", "expected_keywords": []},
        {"id": 2, "type": "qa", "expected_keywords": ["o(n log n)"]},
        {"id": 3, "type": "reasoning", "expected_keywords": ["no", "not necessarily"]},
    ]
    score_results(results, tasks)
    # Code with no keywords → structural (def only = 1/3)
    assert results[0]["score"] == round(1 / 3, 4)
    # QA with keyword match → 1.0
    assert results[1]["score"] == 1.0
    # Reasoning with both keywords → 1.0
    assert results[2]["score"] == 1.0
