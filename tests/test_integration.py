"""Integration tests for the full scoring → report pipeline.

These tests verify end-to-end correctness without a live server:
raw results → score_results() → generate_report() → results.md.
"""

import asyncio
import json
import math
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from evaluator import score_results
from report import generate_report, generate_report_from_file

import json
import math
import tempfile
from pathlib import Path

from evaluator import score_results
from report import generate_report, generate_report_from_file


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_results_unscored():
    """Return raw results with score=null (unscored)."""
    return [
        {
            "run_id": "integration-test-001",
            "task_id": 1,
            "task_type": "code",
            "prompt": "Write a linked-list reverse function",
            "response": "def reverse_linked_list(head: 'ListNode') -> 'ListNode':\n    \"\"\"Reverse a linked list.\"\"\"\n    prev: 'ListNode' = None\n    curr: 'ListNode' = head\n    while curr:\n        nxt = curr.next\n        curr.next = prev\n        prev = curr\n        curr = nxt\n    return prev",
            "score": None,
            "timestamp": "2026-06-01T12:00:00+00:00",
        },
        {
            "run_id": "integration-test-001",
            "task_id": 2,
            "task_type": "qa",
            "prompt": "What is the time complexity of quicksort?",
            "response": "Quicksort has O(n log n) average case time complexity and O(n^2) worst case.",
            "score": None,
            "timestamp": "2026-06-01T12:00:01+00:00",
        },
        {
            "run_id": "integration-test-001",
            "task_id": 3,
            "task_type": "reasoning",
            "prompt": "Is it necessarily true that all generators are iterators?",
            "response": "Yes, in Python every generator is an iterator because it implements __next__ and __iter__.",
            "score": None,
            "timestamp": "2026-06-01T12:00:02+00:00",
        },
    ]


def _mock_tasks():
    """Return task definitions matching the mock results."""
    return [
        {
            "id": 1,
            "type": "code",
            "expected_keywords": ["reverse", "linked", "list"],
        },
        {
            "id": 2,
            "type": "qa",
            "expected_keywords": ["o(n log n)", "quicksort"],
        },
        {
            "id": 3,
            "type": "reasoning",
            "expected_keywords": ["yes", "generator", "iterator"],
        },
    ]


def _mock_config():
    """Return a minimal config dict."""
    return {
        "model": "llama3.1:8b",
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.95,
    }


# ---------------------------------------------------------------------------
# Test 1: Full pipeline — scoring populates scores
# ---------------------------------------------------------------------------

def test_full_pipeline_scored_results():
    """score_results populates score (float) on every entry that was null."""
    results = _mock_results_unscored()
    tasks = _mock_tasks()

    # Verify pre-condition: all scores are None
    for r in results:
        assert r["score"] is None, "Pre-condition: score should be None before scoring"

    # Run the scoring step
    score_results(results, tasks)

    # Verify post-condition: all scores are now floats
    for r in results:
        assert isinstance(r["score"], float), (
            f"Expected float score for task {r['task_id']}, got {type(r['score'])}"
        )
        assert 0.0 <= r["score"] <= 1.0, (
            f"Score {r['score']} for task {r['task_id']} out of [0, 1]"
        )
        assert "matched_keywords" in r, f"matched_keywords missing for task {r['task_id']}"


# ---------------------------------------------------------------------------
# Test 2: Full pipeline — scoring + report generation
# ---------------------------------------------------------------------------

def test_full_pipeline_report_generation():
    """Score results then generate report; verify file is written with sections."""
    results = _mock_results_unscored()
    tasks = _mock_tasks()
    config = _mock_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = str(Path(tmpdir) / "results.md")

        # Step 1: Score
        score_results(results, tasks)

        # Step 2: Generate report
        md = generate_report(results, config, output_path=out_path)

        # Step 3: Verify file exists
        assert Path(out_path).exists(), f"Report file not written to {out_path}"

        # Step 4: Verify all expected sections are present
        for section in [
            "# Benchmark Results",
            "Task Breakdown",
            "Configuration",
            "Parameter Averages",
            "## Ranking",
        ]:
            assert section in md, f"Missing section '{section}' in report"

        # Step 5: Verify run_id appears
        assert "integration-test-001" in md


# ---------------------------------------------------------------------------
# Test 3: Report contains all task types
# ---------------------------------------------------------------------------

def test_report_contains_all_task_types():
    """Report has table entries for code, qa, and reasoning task types."""
    results = _mock_results_unscored()
    tasks = _mock_tasks()
    config = _mock_config()

    score_results(results, tasks)
    md = generate_report(results, config)

    # Each task type should appear in the Task Breakdown table
    for tt in ("code", "qa", "reasoning"):
        assert tt in md, f"Task type '{tt}' not found in report"

    # Count table rows — should have exactly 3 data rows
    lines = md.split("\n")
    in_breakdown = False
    data_rows = 0
    for line in lines:
        if "| Task ID | Type | Difficulty | Score | Matched Keywords |" in line:
            in_breakdown = True
            continue
        if in_breakdown:
            if line.startswith("#"):
                break
            if line.strip().startswith("|") and "---" not in line:
                cells = [c.strip() for c in line.split("|") if c.strip()]
                if len(cells) == 5:
                    data_rows += 1

    assert data_rows == 3, f"Expected 3 task rows in breakdown, got {data_rows}"


# ---------------------------------------------------------------------------
# Test 4: Average calculation is mathematically correct
# ---------------------------------------------------------------------------

def test_average_calculation_correct():
    """Known scores produce mathematically correct averages per task type."""
    results = [
        {
            "run_id": "avg-test",
            "task_id": 1,
            "task_type": "qa",
            "response": "hello world",
            "score": 0.5,
            "timestamp": "2026-06-01T12:00:00+00:00",
            "matched_keywords": ["hello"],
        },
        {
            "run_id": "avg-test",
            "task_id": 2,
            "task_type": "qa",
            "response": "foo bar baz",
            "score": 1.0,
            "timestamp": "2026-06-01T12:00:01+00:00",
            "matched_keywords": ["foo", "bar", "baz"],
        },
        {
            "run_id": "avg-test",
            "task_id": 3,
            "task_type": "reasoning",
            "response": "no it is not",
            "score": 0.25,
            "timestamp": "2026-06-01T12:00:02+00:00",
            "matched_keywords": ["no"],
        },
    ]
    config = _mock_config()

    md = generate_report(results, config)

    # QA average: (0.5 + 1.0) / 2 = 0.75
    # Reasoning average: 0.25 / 1 = 0.25
    # Overall average: (0.5 + 1.0 + 0.25) / 3 = 0.5833...

    lines = md.split("\n")
    in_avg_table = False
    found_qa = False
    found_reasoning = False

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
                if tt == "qa":
                    assert abs(avg_val - 0.75) < 0.001, (
                        f"QA avg should be 0.75, got {avg_val}"
                    )
                    found_qa = True
                elif tt == "reasoning":
                    assert abs(avg_val - 0.25) < 0.001, (
                        f"Reasoning avg should be 0.25, got {avg_val}"
                    )
                    found_reasoning = True

    assert found_qa, "QA average row not found"
    assert found_reasoning, "Reasoning average row not found"

    # Verify overall ranking average
    assert "0.5833" in md, (
        f"Overall average 0.5833 not found in report (md snippet: ...{md[-200:]})"
    )


# ---------------------------------------------------------------------------
# Test 5: Markdown table syntax is valid
# ---------------------------------------------------------------------------

def test_report_markdown_valid():
    """Generated markdown has proper table syntax (pipes, header separators)."""
    results = _mock_results_unscored()
    tasks = _mock_tasks()
    config = _mock_config()

    score_results(results, tasks)
    md = generate_report(results, config)

    lines = md.split("\n")

    # Find the Task Breakdown table and verify structure
    table_started = False
    header_found = False
    separator_found = False
    data_rows = 0

    for line in lines:
        if "## Task Breakdown" in line:
            table_started = True
            continue
        if table_started:
            if line.startswith("#"):
                break  # any heading
            if not line.strip():
                continue
            if "| Task ID | Type | Difficulty | Score | Matched Keywords |" in line:
                header_found = True
                continue
            if "|---" in line or "|---------" in line:
                separator_found = True
                continue
            if line.strip().startswith("|"):
                cells = [c.strip() for c in line.split("|") if c.strip()]
                assert len(cells) == 5, (
                    f"Table row should have 5 cells, got {len(cells)}: {line}"
                )
                data_rows += 1

    assert header_found, "Table header row not found"
    assert separator_found, "Table separator row (---) not found"
    assert data_rows == 3, f"Expected 3 data rows, got {data_rows}"

    # Verify Configuration table also has proper syntax
    config_started = False
    config_header = False
    config_separator = False

    for line in lines:
        if "## Configuration" in line:
            config_started = True
            continue
        if config_started:
            if line.startswith("#"):
                break
            if not line.strip():
                continue
            if "| Parameter | Value |" in line:
                config_header = True
                continue
            if "|---" in line or "|-----------" in line:
                config_separator = True
                continue
            if line.strip().startswith("|"):
                cells = [c.strip() for c in line.split("|") if c.strip()]
                assert len(cells) == 2, (
                    f"Config row should have 2 cells, got {len(cells)}: {line}"
                )

    assert config_header, "Configuration table header not found"
    assert config_separator, "Configuration table separator not found"


# ---------------------------------------------------------------------------
# Test 6: Score → write → read → report roundtrip
# ---------------------------------------------------------------------------

def test_score_then_report_roundtrip():
    """Write scored results to results.json, read back, generate report, verify consistency."""
    results = _mock_results_unscored()
    tasks = _mock_tasks()
    config = _mock_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        results_json = Path(tmpdir) / "results.json"
        report_md = Path(tmpdir) / "results.md"

        # Step 1: Score
        score_results(results, tasks)

        # Step 2: Write scored results to JSON
        results_json.write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )

        # Step 3: Write config to JSON
        config_path = Path(tmpdir) / "run.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        # Step 4: Generate report from file (reads from disk)
        md_from_file = generate_report_from_file(
            results_path=str(results_json),
            config_path=str(config_path),
            output_path=str(report_md),
        )

        # Step 5: Generate report directly (in-memory) for comparison
        md_direct = generate_report(results, config, output_path=str(report_md) + ".direct")

        # Step 6: Verify both reports are identical
        assert md_from_file == md_direct, (
            "Report from file should match report generated directly"
        )

        # Step 7: Verify the file was written
        assert report_md.exists(), f"Report file not written to {report_md}"

        # Step 8: Verify consistency — all scores from results appear in report
        for r in results:
            score_str = f"{r['score']:.4f}"
            assert score_str in md_from_file, (
                f"Score {score_str} for task {r['task_id']} not found in report"
            )

        # Step 9: Verify the written file content matches the returned string
        written_content = report_md.read_text(encoding="utf-8")
        assert written_content == md_from_file, (
            "Written file content should match returned markdown string"
        )


# ---------------------------------------------------------------------------
# Test 7: --report flag generates results.md
# ---------------------------------------------------------------------------

def test_report_flag_generates_file(monkeypatch, tmp_path):
    """When --report is passed, run.py calls generate_report_from_file() and results.md is created."""
    import sys
    from pathlib import Path as StdPath
    from unittest.mock import patch

    # Create a minimal config file
    config = {"baseUrl": "http://localhost:8080", "models": [{"id": "llama3.1:8b"}]}
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    # Create sample results.json
    results = [
        {
            "run_id": "cli-test-001",
            "task_id": 1,
            "task_type": "code",
            "prompt": "Write a function",
            "response": "def foo(): pass",
            "score": 0.5,
            "timestamp": "2026-06-01T12:00:00+00:00",
            "matched_keywords": ["def"],
        },
    ]
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results), encoding="utf-8")

    # Redirect results paths to tmp so the project's real results/ stays clean
    import run as run_mod
    import report as report_mod
    real_generate = report_mod.generate_report_from_file

    results_json_path = tmp_path / "results.json"
    results_json_path.write_text(json.dumps(results), encoding="utf-8")
    report_path = tmp_path / "results.md"

    monkeypatch.setattr(run_mod, "_RESULTS_PATH", results_json_path)
    monkeypatch.setattr(run_mod, "_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        report_mod,
        "generate_report_from_file",
        lambda: real_generate(results_path=str(results_json_path), output_path=str(report_path)),
    )

    with patch("run.run_benchmark", return_value=results):
        monkeypatch.setattr(sys, "argv", ["run.py", "--report", "--config", str(config_path)])
        monkeypatch.setattr(sys, "exit", lambda code=None: None)

        from run import main
        main()

    assert report_path.exists(), "results.md was not created by --report flag"
    content = report_path.read_text(encoding="utf-8")
    assert "# Benchmark Results" in content


# ---------------------------------------------------------------------------
# Test 8: --report flag content has all 4 sections
# ---------------------------------------------------------------------------

def test_report_flag_content_sections(monkeypatch, tmp_path):
    """The report generated by --report contains all 4 required sections."""
    import sys
    from pathlib import Path as StdPath
    from unittest.mock import patch

    # Create a minimal config file
    config = {
        "baseUrl": "http://localhost:8080",
        "models": [{"id": "llama3.1:8b"}],
    }
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    # Create sample results.json with multiple task types
    results = [
        {
            "run_id": "cli-test-002",
            "task_id": 1,
            "task_type": "code",
            "prompt": "Write a function",
            "response": "def foo(): pass",
            "score": 0.5,
            "timestamp": "2026-06-01T12:00:00+00:00",
            "matched_keywords": ["def"],
        },
        {
            "run_id": "cli-test-002",
            "task_id": 2,
            "task_type": "qa",
            "prompt": "What is time complexity?",
            "response": "O(n log n)",
            "score": 1.0,
            "timestamp": "2026-06-01T12:00:01+00:00",
            "matched_keywords": ["o(n log n)"],
        },
        {
            "run_id": "cli-test-002",
            "task_id": 3,
            "task_type": "reasoning",
            "prompt": "Explain sorting",
            "response": "comparison-based",
            "score": 0.75,
            "timestamp": "2026-06-01T12:00:02+00:00",
            "matched_keywords": ["comparison"],
        },
    ]
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results), encoding="utf-8")

    # Redirect results paths to tmp so the project's real results/ stays clean
    import run as run_mod
    import report as report_mod
    real_generate = report_mod.generate_report_from_file

    results_json_path = tmp_path / "results.json"
    results_json_path.write_text(json.dumps(results), encoding="utf-8")
    report_path = tmp_path / "results.md"

    monkeypatch.setattr(run_mod, "_RESULTS_PATH", results_json_path)
    monkeypatch.setattr(run_mod, "_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        report_mod,
        "generate_report_from_file",
        lambda: real_generate(results_path=str(results_json_path), output_path=str(report_path)),
    )

    with patch("run.run_benchmark", return_value=results):
        monkeypatch.setattr(sys, "argv", ["run.py", "--report", "--config", str(config_path)])
        monkeypatch.setattr(sys, "exit", lambda code=None: None)

        from run import main
        main()

    assert report_path.exists(), "results.md not created"
    content = report_path.read_text(encoding="utf-8")

    # Verify all 4 required sections are present
    required_sections = [
        "Task Breakdown",
        "Configuration",
        "Parameter Averages",
        "## Ranking",
    ]
    for section in required_sections:
        assert section in content, f"Missing section '{section}' in generated report"

    # Verify: config parameters appear in Configuration section
    for key in ("model", "baseUrl"):
        assert key in content, f"Config key '{key}' not found in report"

    # Verify: all task types appear in Task Breakdown
    for tt in ("code", "qa", "reasoning"):
        assert tt in content, f"Task type '{tt}' not found in report"


# ---------------------------------------------------------------------------
# Test 9: Without --report flag, no report is generated
# ---------------------------------------------------------------------------

def test_no_report_flag_no_file(monkeypatch, tmp_path):
    """When --report is NOT passed, run.py does NOT create results.md."""
    import sys
    from pathlib import Path as StdPath
    from unittest.mock import patch

    # Create a minimal config file
    config = {"baseUrl": "http://localhost:8080", "models": [{"id": "llama3.1:8b"}]}
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    # Create sample results.json
    results = [
        {
            "run_id": "cli-test-003",
            "task_id": 1,
            "task_type": "code",
            "prompt": "Write a function",
            "response": "def foo(): pass",
            "score": 0.5,
            "timestamp": "2026-06-01T12:00:00+00:00",
            "matched_keywords": ["def"],
        },
    ]
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results), encoding="utf-8")

    # Redirect results paths to tmp so the project's real results/ stays clean
    import run as run_mod

    results_json_path = tmp_path / "results.json"
    results_json_path.write_text(json.dumps(results), encoding="utf-8")
    report_path = tmp_path / "results.md"

    monkeypatch.setattr(run_mod, "_RESULTS_PATH", results_json_path)
    monkeypatch.setattr(run_mod, "_RESULTS_DIR", tmp_path)

    # Mock run_benchmark and run main() WITHOUT --report flag
    with patch("run.run_benchmark", return_value=results):
        monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(config_path)])
        monkeypatch.setattr(sys, "exit", lambda code=None: None)

        from run import main
        main()

    # Verify: no report was generated
    assert not report_path.exists(), "results.md was created even without --report flag"


# ---------------------------------------------------------------------------
# Test 13: banner shows detected model name
# ---------------------------------------------------------------------------

def test_banner_shows_detected_model(capsys):
    """When run_stamp carries a model id, the banner includes
    'Model: {model}' in its output."""
    import sys
    from io import StringIO
    from run import _print_banner

    # Capture stderr where the banner is printed
    old_stderr = sys.stderr
    sys.stderr = StringIO()

    try:
        _print_banner(
            task_count=10,
            run_stamp={"model": "llama-3-8b-instruct", "baseUrl": "http://localhost:8080"},
            config_path="configs/models.json",
            limit=None,
        )
    finally:
        output = sys.stderr.getvalue()
        sys.stderr = old_stderr

    # Verify: banner includes the detected model line
    assert "Model: llama-3-8b-instruct" in output


# ---------------------------------------------------------------------------
# Test 14: banner omits model line when no detected model
# ---------------------------------------------------------------------------

def test_banner_omits_model_when_none(capsys):
    """When run_stamp carries no model id, the banner does NOT include a 'Model:' line."""
    import sys
    from io import StringIO
    from run import _print_banner

    old_stderr = sys.stderr
    sys.stderr = StringIO()

    try:
        _print_banner(
            task_count=5,
            run_stamp={"baseUrl": "http://localhost:8080"},
            config_path="configs/models.json",
            limit=3,
        )
    finally:
        output = sys.stderr.getvalue()
        sys.stderr = old_stderr

    # Verify: no 'Model:' line in output
    assert "Model:" not in output
