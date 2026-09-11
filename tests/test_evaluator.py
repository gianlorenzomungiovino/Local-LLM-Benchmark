"""Comprehensive tests for the evaluator module."""

from evaluator import score_result, score_results, get_category_scores


# ---------------------------------------------------------------------------
# score_result — code_execute tests
# ---------------------------------------------------------------------------

def test_code_execute_with_all_elements():
    """Code_execute task with good structure, keywords, and algorithmic patterns."""
    result = {"response": 'def two_sum(nums: list[int], target: int) -> list[int]:\n    """Find two indices using a hash dict."""\n    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:\n            return [seen[complement], i]\n        seen[num] = i\n    return []'}
    task = {"type": "code_execute", "expected_keywords": ["dict", "enumerate", "target", "complement"], "difficulty": "medium"}
    score_result(result, task)
    assert result["score"] > 0.5  # Good code with keywords should score well
    assert "dict" in result["matched_keywords"]
    assert "enumerate" in result["matched_keywords"]
    assert "target" in result["matched_keywords"]
    assert "complement" in result["matched_keywords"]
    assert "test_score" in result


def test_code_execute_partial_keywords():
    """Code_execute with some keyword matches."""
    result = {"response": 'def merge_intervals(intervals):\n    intervals.sort()\n    result = []\n    for interval in intervals:\n        if not result or result[-1][1] < interval[0]:\n            result.append(interval)\n    return result'}
    task = {"type": "code_execute", "expected_keywords": ["sort", "append", "interval", "nonexistent"], "difficulty": "medium"}
    score_result(result, task)
    assert "sort" in result["matched_keywords"]
    assert "append" in result["matched_keywords"]
    assert "interval" in result["matched_keywords"]
    assert "nonexistent" not in result["matched_keywords"]
    assert result["score"] > 0


def test_code_execute_no_structure():
    """Code_execute with no structural elements scores low."""
    result = {"response": "just a plain string"}
    task = {"type": "code_execute", "expected_keywords": ["function", "algorithm"], "difficulty": "hard"}
    score_result(result, task)
    assert result["score"] < 0.3


# ---------------------------------------------------------------------------
# score_result — instruction_following tests
# ---------------------------------------------------------------------------

def test_instruction_following_with_constraints():
    """Instruction_following task with constraint checking."""
    result = {"response": 'def factorial(n):\n    """Compute factorial."""\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)'}
    task = {
        "type": "instruction_following",
        "expected_keywords": ["def", "return", "n-1", "factorial", "0"],
        "difficulty": "medium",
        "constraints": [
            {"type": "no_comments", "value": True},
            {"type": "no_imports", "value": True},
            {"type": "uses_recursion", "value": True},
        ]
    }
    score_result(result, task)
    assert "constraint_score" in result
    assert result["score"] > 0


def test_instruction_following_violates_constraints():
    """Instruction_following task that violates constraints."""
    result = {"response": 'def factorial(n):\n    # This is a comment\n    import math\n    return math.factorial(n)'}
    task = {
        "type": "instruction_following",
        "expected_keywords": ["def", "return"],
        "difficulty": "medium",
        "constraints": [
            {"type": "no_comments", "value": True},
            {"type": "no_imports", "value": True},
        ]
    }
    score_result(result, task)
    assert "constraint_score" in result
    assert result["constraint_score"] < 1.0


# ---------------------------------------------------------------------------
# score_result — reasoning_math tests
# ---------------------------------------------------------------------------

def test_reasoning_math_numeric_exact():
    """Reasoning_math with correct numeric answer."""
    result = {"response": "The answer is 1. The GCD of Fibonacci numbers follows the property gcd(F_m, F_n) = F_gcd(m,n). So gcd(F_100, F_200) = F_100... wait, actually gcd(100, 200) = 100, so the answer is F_100. But wait, F_1 = 1, F_2 = 1, so the GCD is 1."}
    task = {"type": "reasoning_math", "expected_keywords": ["1", "gcd", "property"], "difficulty": "hard",
            "verification": "numeric_exact", "answer": 1}
    score_result(result, task)
    assert "numeric_score" in result
    assert result["numeric_score"] > 0


def test_reasoning_math_numeric_wrong():
    """Reasoning_math with wrong numeric answer."""
    result = {"response": "The answer is 42. This is the result of the calculation."}
    task = {"type": "reasoning_math", "expected_keywords": ["42", "calculation"], "difficulty": "hard",
            "verification": "numeric_exact", "answer": 1}
    score_result(result, task)
    assert "numeric_score" in result
    assert result["numeric_score"] == 0.0


def test_reasoning_math_numeric_range():
    """Reasoning_math with answer in valid range."""
    result = {"response": "The integral evaluates to approximately 0.36 which is between 0 and 1."}
    task = {"type": "reasoning_math", "expected_keywords": ["integration", "parts", "e"], "difficulty": "medium",
            "verification": "numeric_range", "answer_range": [0, 1]}
    score_result(result, task)
    assert "numeric_score" in result
    assert result["numeric_score"] > 0


# ---------------------------------------------------------------------------
# score_result — reasoning_science tests
# ---------------------------------------------------------------------------

def test_reasoning_science_deep_explanation():
    """Reasoning_science with deep explanation scores higher."""
    result = {"response": "The de Broglie wavelength is given by λ = h/p. In the Bohr model, the momentum of the electron is p = mv. For the n=1 state, the velocity is v1 = αc where α is the fine structure constant. For the n=2 state, the velocity is v2 = αc/2. Since wavelength is inversely proportional to momentum, lambda2/lambda1 = v1/v2 = 2. Therefore the ratio is 2."}
    task = {"type": "reasoning_science", "expected_keywords": ["2", "momentum", "velocity", "wavelength"], "difficulty": "hard"}
    score_result(result, task)
    assert result["score"] > 0.4  # Deep explanation with keywords
    assert "momentum" in result["matched_keywords"]


# ---------------------------------------------------------------------------
# score_result — reasoning_logic tests
# ---------------------------------------------------------------------------

def test_reasoning_logic_step_by_step():
    """Reasoning_logic with step-by-step reasoning scores higher."""
    result = {"response": "Step 1: Turn on switch 1 and wait 5 minutes for the bulb to heat up. Step 2: Turn off switch 1 and turn on switch 2. Step 3: Enter the room. The warm bulb that is off is switch 1. The bulb that is on is switch 2. The cold bulb that is off is switch 3."}
    task = {"type": "reasoning_logic", "expected_keywords": ["heat", "warm", "off", "on", "bulb"], "difficulty": "hard"}
    score_result(result, task)
    assert "heat" in result["matched_keywords"]
    assert "warm" in result["matched_keywords"]


# ---------------------------------------------------------------------------
# score_result — qa_knowledge tests (default type)
# ---------------------------------------------------------------------------

def test_qa_knowledge_exact_match():
    """QA knowledge exact keyword match."""
    result = {"response": "The GIL prevents true parallel execution of Python threads."}
    task = {"type": "qa_knowledge", "expected_keywords": ["GIL", "thread", "parallel"]}
    score_result(result, task)
    assert result["score"] == 1.0
    assert "GIL" in result["matched_keywords"]


def test_qa_knowledge_partial_match():
    """QA knowledge partial keyword match."""
    result = {"response": "The GIL is a mutex that protects access to Python objects."}
    task = {"type": "qa_knowledge", "expected_keywords": ["GIL", "thread", "parallel", "interpreter"]}
    score_result(result, task)
    # Only "GIL" matches (1 out of 4)
    assert result["score"] == 0.25
    assert "GIL" in result["matched_keywords"]


# ---------------------------------------------------------------------------
# score_result — code_refactoring tests
# ---------------------------------------------------------------------------

def test_code_refactoring_with_explanation():
    """Code_refactoring with good explanation scores higher."""
    result = {"response": "Here is the refactored code:\n\n```python\ndef process_data(data: list[int]) -> list[int]:\n    '''Process data by filtering positives and doubling.'''\n    return [x * 2 for x in data if x > 0]\n```\n\nThis improves readability because we use a list comprehension instead of a loop. It also adds type hints which improves clarity."}
    task = {"type": "code_refactoring", "expected_keywords": ["list comprehension", "filter", "error", "type hint", "docstring"], "difficulty": "medium"}
    score_result(result, task)
    assert "list comprehension" in result["matched_keywords"]
    assert result["score"] > 0.5


# ---------------------------------------------------------------------------
# score_result — code_debug tests
# ---------------------------------------------------------------------------

def test_code_debug_identifies_bugs():
    """Code_debug that identifies bugs scores higher."""
    result = {"response": "There are two bugs in this implementation:\n\n1. **Off-by-one error**: right should be len(arr) - 1, not len(arr). This causes an IndexError.\n\n2. **Incorrect pointer updates**: When arr[mid] < target, we should set left = mid + 1, not right = mid. Similarly, when arr[mid] > target, we should set right = mid - 1, not left = mid.\n\nHere is the corrected version:"}
    task = {"type": "code_debug", "expected_keywords": ["off-by-one", "len-1", "mid+1", "mid-1", "error"], "difficulty": "medium"}
    score_result(result, task)
    assert "off-by-one" in result["matched_keywords"]
    assert "error" in result["matched_keywords"]


# ---------------------------------------------------------------------------
# score_results — batch tests
# ---------------------------------------------------------------------------

def test_score_results_batch_new_types():
    """score_results processes multiple results with new task types."""
    results = [
        {"task_id": 101, "response": "def two_sum(nums: list[int], target: int):\n    '''Two sum.'''\n    seen = {}\n    return seen"},
        {"task_id": 201, "response": "The answer is 729. This involves composition of cubic polynomials."},
        {"task_id": 401, "response": "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n-1)"},
    ]
    tasks = [
        {"id": 101, "type": "code_execute", "expected_keywords": ["dict", "enumerate"]},
        {"id": 201, "type": "reasoning_math", "expected_keywords": ["729", "composition"]},
        {"id": 401, "type": "instruction_following", "expected_keywords": ["def", "return"]},
    ]
    score_results(results, tasks)
    assert results[0]["score"] > 0  # code_execute scored
    assert results[1]["score"] > 0  # reasoning_math scored
    assert results[2]["score"] > 0  # instruction_following scored


def test_score_results_preserves_fields():
    """Original fields are preserved with new task types."""
    results = [
        {
            "run_id": "abc-123",
            "task_id": 101,
            "task_type": "code_execute",
            "difficulty": "medium",
            "prompt": "Two sum",
            "response": "def two_sum(nums: list[int], target: int):\n    '''Two sum.'''\n    return {}",
            "score": None,
            "timestamp": "2026-09-11T00:00:00Z",
        }
    ]
    tasks = [
        {"id": 101, "type": "code_execute", "expected_keywords": ["dict"]},
    ]
    score_results(results, tasks)
    assert results[0]["run_id"] == "abc-123"
    assert results[0]["task_id"] == 101
    assert results[0]["difficulty"] == "medium"
    assert results[0]["score"] > 0


def test_score_results_unknown_task():
    """Results with unknown task_id get neutral score 0.5."""
    results = [{"task_id": 999, "response": "orphan"}]
    tasks = [{"id": 1, "type": "qa_knowledge", "expected_keywords": ["foo"]}]
    score_results(results, tasks)
    assert results[0]["score"] == 0.5


def test_score_results_empty_list():
    """Empty results list returns empty list."""
    assert score_results([], []) == []


# ---------------------------------------------------------------------------
# get_category_scores tests
# ---------------------------------------------------------------------------

def test_get_category_scores_basic():
    """get_category_scores computes correct averages."""
    results = [
        {"task_type": "code_execute", "score": 0.8, "difficulty": "medium"},
        {"task_type": "code_execute", "score": 0.6, "difficulty": "hard"},
        {"task_type": "reasoning_math", "score": 0.9, "difficulty": "hard"},
    ]
    categories = get_category_scores(results)
    assert "code_execute" in categories
    assert "reasoning_math" in categories
    assert categories["code_execute"]["average"] == 0.7
    assert categories["code_execute"]["count"] == 2
    assert categories["reasoning_math"]["count"] == 1


def test_get_category_scores_by_difficulty():
    """get_category_scores computes per-difficulty averages."""
    results = [
        {"task_type": "qa_knowledge", "score": 1.0, "difficulty": "easy"},
        {"task_type": "qa_knowledge", "score": 0.5, "difficulty": "hard"},
    ]
    categories = get_category_scores(results)
    assert categories["qa_knowledge"]["by_difficulty"]["easy"] == 1.0
    assert categories["qa_knowledge"]["by_difficulty"]["hard"] == 0.5


def test_get_category_scores_empty():
    """get_category_scores with empty list returns empty dict."""
    assert get_category_scores([]) == {}
