"""Comprehensive tests for the markdown report generator."""

import json
import tempfile
from pathlib import Path

from report import generate_report


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sample_results():
    """Return a realistic list of scored benchmark results."""
    return [
        {
            "run_id": "test-run-001",
            "task_id": 1,
            "task_type": "code",
            "prompt": "Write a function",
            "response": "def foo(): pass",
            "score": 0.5,
            "timestamp": "2026-06-01T12:00:00+00:00",
            "matched_keywords": ["def"],
        },
        {
            "run_id": "test-run-001",
            "task_id": 2,
            "task_type": "qa",
            "prompt": "What is time complexity?",
            "response": "O(n log n) average",
            "score": 1.0,
            "timestamp": "2026-06-01T12:00:01+00:00",
            "matched_keywords": ["o(n log n)"],
        },
        {
            "run_id": "test-run-001",
            "task_id": 3,
            "task_type": "reasoning",
            "prompt": "Explain sorting",
            "response": "comparison-based sort",
            "score": 0.75,
            "timestamp": "2026-06-01T12:00:02+00:00",
            "matched_keywords": ["comparison"],
        },
    ]


def _sample_config():
    """Return a minimal config dict."""
    return {
        "model": "llama3.1:8b",
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.95,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_report_header_includes_run_id():
    """Generated report contains run_id from results."""
    results = _sample_results()
    config = _sample_config()
    md = generate_report(results, config)
    assert "test-run-001" in md
    assert "# Benchmark Results" in md


def test_report_task_breakdown_table():
    """Table contains all tasks with scores."""
    results = _sample_results()
    config = _sample_config()
    md = generate_report(results, config)
    lines = md.split("\n")
    # Find the table rows — stop at the next section header
    table_started = False
    task_rows = 0
    for line in lines:
        if "| Task ID | Type | Score | Matched Keywords |" in line:
            table_started = True
            continue
        if table_started:
            if line.startswith("#"):
                break  # any heading
            if line.strip().startswith("|"):
                if "---" in line:
                    continue
                task_rows += 1
                cells = [c.strip() for c in line.split("|") if c.strip()]
                assert len(cells) == 4, f"Expected 4 cells, got {len(cells)} in row: {line}"
    assert task_rows == 3, f"Expected 3 task rows, got {task_rows}"


def test_report_configuration_section():
    """Config parameters appear in report."""
    results = _sample_results()
    config = _sample_config()
    md = generate_report(results, config)
    assert "## Configuration" in md
    for key in config:
        assert key in md, f"Config key '{key}' not found in report"


def test_report_parameter_averages():
    """Average scores per task type are calculated correctly."""
    results = _sample_results()
    config = _sample_config()
    md = generate_report(results, config)
    assert "## Parameter Averages" in md

    # code: only task 1 with score 0.5 → avg = 0.5
    # qa: only task 2 with score 1.0 → avg = 1.0
    # reasoning: only task 3 with score 0.75 → avg = 0.75
    lines = md.split("\n")
    in_avg_table = False
    for line in lines:
        if "| Task Type | Average Score |" in line:
            in_avg_table = True
            continue
        if in_avg_table and line.strip().startswith("|"):
            if "---" in line:
                continue
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if len(cells) == 2:
                tt, avg_str = cells
                avg_val = float(avg_str)
                if tt == "code":
                    assert abs(avg_val - 0.5) < 0.001, f"code avg should be 0.5, got {avg_val}"
                elif tt == "qa":
                    assert abs(avg_val - 1.0) < 0.001, f"qa avg should be 1.0, got {avg_val}"
                elif tt == "reasoning":
                    assert abs(avg_val - 0.75) < 0.001, f"reasoning avg should be 0.75, got {avg_val}"


def test_report_ranking_section():
    """Ranking section is present with average score."""
    results = _sample_results()
    config = _sample_config()
    md = generate_report(results, config)
    assert "## Ranking" in md
    # Overall average: (0.5 + 1.0 + 0.75) / 3 = 0.75
    # Ranking section present with proper structure
    assert "| Run |" in md
    assert "Rank" in md


def test_report_writes_to_file():
    """Report is written to the specified output path."""
    results = _sample_results()
    config = _sample_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = str(Path(tmpdir) / "output" / "results.md")
        md = generate_report(results, config, output_path=out_path)
        assert Path(out_path).exists(), f"Report file not written to {out_path}"
        content = Path(out_path).read_text(encoding="utf-8")
        assert content == md


def test_report_empty_results():
    """Handles empty results list gracefully (no crash)."""
    results: list = []
    config = _sample_config()
    md = generate_report(results, config)
    assert "# Benchmark Results" in md
    assert "No results available." in md
    # No task rows, averages, or ranking tables should appear
    assert "| Task ID |" not in md
    assert "| Task Type | Average Score |" not in md
    assert "| Run | Average Score |" not in md

def test_report_preserves_matched_keywords():
    """matched_keywords from evaluator appear in the table."""
    results = _sample_results()
    config = _sample_config()
    md = generate_report(results, config)
    # All matched keywords should appear in the report
    for r in results:
        for kw in r.get("matched_keywords") or []:
            assert kw in md, f"Matched keyword '{kw}' not found in report"
