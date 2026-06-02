"""Tests for runner.py."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from runner import (
    load_tasks,
    load_config,
    generate_run_id,
    run_benchmark,
    generate_report,
)


class TestLoadTasks:
    def test_load_tasks_success(self, tmp_path):
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps([
            {"id": "t1", "prompt": "hello", "type": "code", "expected": "def foo"}
        ]))
        tasks = load_tasks(tasks_file)
        assert len(tasks) == 1
        assert tasks[0]["id"] == "t1"

    def test_load_tasks_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_tasks("nonexistent.json")

    def test_load_tasks_not_array(self, tmp_path):
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text('{"id": "t1"}')
        with pytest.raises(ValueError, match="must contain a JSON array"):
            load_tasks(tasks_file)

    def test_load_tasks_missing_field(self, tmp_path):
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps([{"id": "t1", "prompt": "hello"}]))
        with pytest.raises(ValueError, match="missing field"):
            load_tasks(tasks_file)


class TestLoadConfig:
    def test_load_config_success(self, tmp_path):
        config_file = tmp_path / "run.json"
        config_file.write_text(json.dumps({
            "temperature": 0.3,
            "top_k": 30,
            "top_p": 0.95,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
        }))
        config = load_config(config_file)
        assert config["temperature"] == 0.3
        assert config["top_k"] == 30

    def test_load_config_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.json")

    def test_load_config_missing_field(self, tmp_path):
        config_file = tmp_path / "run.json"
        config_file.write_text(json.dumps({"temperature": 0.3}))
        with pytest.raises(ValueError, match="missing required field"):
            load_config(config_file)


class TestGenerateRunId:
    def test_run_id_format(self):
        run_id = generate_run_id()
        assert run_id.startswith("run-")
        assert len(run_id) > 10
        # Format: run-YYYYMMDD-HHMM-XXXX
        parts = run_id.split("-")
        assert len(parts) == 4
        assert parts[0] == "run"

    def test_run_id_uniqueness(self):
        ids = {generate_run_id() for _ in range(10)}
        assert len(ids) == 10


class TestRunBenchmark:
    def test_run_benchmark_basic(self, tmp_path):
        # Create tasks file
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps([
            {"id": "t1", "prompt": "hello", "type": "code", "expected": "def foo"},
            {"id": "t2", "prompt": "world", "type": "qa", "expected": "hello"},
        ]))

        # Create config file
        config_file = tmp_path / "run.json"
        config_file.write_text(json.dumps({
            "temperature": 0.3,
            "top_k": 30,
            "top_p": 0.95,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
        }))

        results_dir = tmp_path / "results"

        # Mock the client
        mock_result = {
            "content": "def foo(): pass",
            "model": "test-model",
            "raw": {},
        }

        with patch("runner.LLMClient") as MockClient:
            mock_client = MagicMock()
            mock_client.send_with_retry.return_value = mock_result
            MockClient.return_value = mock_client

            config = load_config(config_file)
            tasks = load_tasks(tasks_file)
            results = run_benchmark(
                server_url="http://127.0.0.1:8080",
                config=config,
                tasks=tasks,
                results_dir=results_dir,
            )

        assert len(results) == 2
        assert results[0]["task_id"] == "t1"
        assert results[0]["score"] > 0
        assert results[0]["run_id"].startswith("run-")

        # Check results.json was created
        results_file = results_dir / "results.json"
        assert results_file.exists()
        saved = json.loads(results_file.read_text())
        assert len(saved) == 2

    def test_run_benchmark_with_limit(self, tmp_path):
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps([
            {"id": f"t{i}", "prompt": f"prompt {i}", "type": "code", "expected": "def foo"}
            for i in range(5)
        ]))

        config_file = tmp_path / "run.json"
        config_file.write_text(json.dumps({
            "temperature": 0.3,
            "top_k": 30,
            "top_p": 0.95,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
        }))

        results_dir = tmp_path / "results"

        with patch("runner.LLMClient") as MockClient:
            mock_client = MagicMock()
            mock_client.send_with_retry.return_value = {
                "content": "def foo(): pass",
                "model": "test",
                "raw": {},
            }
            MockClient.return_value = mock_client

            config = load_config(config_file)
            tasks = load_tasks(tasks_file)
            results = run_benchmark(
                server_url="http://127.0.0.1:8080",
                config=config,
                tasks=tasks,
                results_dir=results_dir,
                limit=2,
            )

        assert len(results) == 2

    def test_run_benchmark_error_handling(self, tmp_path):
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps([
            {"id": "t1", "prompt": "hello", "type": "code", "expected": "def foo"},
        ]))

        config_file = tmp_path / "run.json"
        config_file.write_text(json.dumps({
            "temperature": 0.3,
            "top_k": 30,
            "top_p": 0.95,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
        }))

        results_dir = tmp_path / "results"

        with patch("runner.LLMClient") as MockClient:
            mock_client = MagicMock()
            mock_client.send_with_retry.side_effect = RuntimeError("server down")
            MockClient.return_value = mock_client

            config = load_config(config_file)
            tasks = load_tasks(tasks_file)
            results = run_benchmark(
                server_url="http://127.0.0.1:8080",
                config=config,
                tasks=tasks,
                results_dir=results_dir,
            )

        assert len(results) == 1
        assert results[0]["score"] == 0.0
        assert "Error" in results[0]["notes"]


class TestGenerateReport:
    def test_generate_report_basic(self, tmp_path):
        results = [
            {
                "task_id": "t1",
                "task_type": "code",
                "prompt": "hello",
                "params": {"temperature": 0.3, "top_k": 30, "top_p": 0.95},
                "run_id": "run-001",
                "timestamp": "2026-06-01T00:00:00Z",
                "model": "test",
                "output": "def foo(): pass",
                "score": 0.8,
                "notes": "AST valid",
            },
            {
                "task_id": "t2",
                "task_type": "qa",
                "prompt": "world",
                "params": {"temperature": 0.3, "top_k": 30, "top_p": 0.95},
                "run_id": "run-001",
                "timestamp": "2026-06-01T00:00:01Z",
                "model": "test",
                "output": "hello world",
                "score": 1.0,
                "notes": "Exact match",
            },
        ]

        report_path = tmp_path / "results.md"
        report = generate_report(results, report_path)

        assert report_path.exists()
        assert "# Benchmark Results" in report
        assert "run-001" in report
        assert "t1" in report
        assert "t2" in report
        assert "0.80" in report
        assert "1.00" in report

    def test_generate_report_multiple_runs(self, tmp_path):
        results = [
            {
                "task_id": "t1",
                "task_type": "code",
                "prompt": "hello",
                "params": {"temperature": 0.3, "top_k": 30, "top_p": 0.95},
                "run_id": "run-001",
                "timestamp": "2026-06-01T00:00:00Z",
                "model": "test",
                "output": "def foo(): pass",
                "score": 0.8,
                "notes": "ok",
            },
            {
                "task_id": "t1",
                "task_type": "code",
                "prompt": "hello",
                "params": {"temperature": 0.7, "top_k": 50, "top_p": 0.8},
                "run_id": "run-002",
                "timestamp": "2026-06-01T01:00:00Z",
                "model": "test",
                "output": "def foo(): pass",
                "score": 0.9,
                "notes": "ok",
            },
        ]

        report_path = tmp_path / "results.md"
        report = generate_report(results, report_path)

        assert "run-001" in report
        assert "run-002" in report
        assert "## Summary by Run" in report

    def test_generate_report_with_pipe_in_notes(self, tmp_path):
        results = [
            {
                "task_id": "t1",
                "task_type": "code",
                "prompt": "hello",
                "params": {"temperature": 0.3, "top_k": 30, "top_p": 0.95},
                "run_id": "run-001",
                "timestamp": "2026-06-01T00:00:00Z",
                "model": "test",
                "output": "def foo(): pass",
                "score": 0.8,
                "notes": "has | pipe char",
            },
        ]

        report_path = tmp_path / "results.md"
        report = generate_report(results, report_path)

        # Pipe chars should be escaped in markdown table
        assert "\\|" in report
