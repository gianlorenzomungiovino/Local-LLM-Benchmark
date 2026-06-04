"""Unit tests for runner — config loading, task iteration, results.json writing."""

import json
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import the module under test
import runner


# ── helpers ──────────────────────────────────────────────────────

def _make_mock_client():
    """Create a mock LLMClient that returns predictable responses."""
    mock = MagicMock()
    mock.complete = AsyncMock(return_value="mocked response")
    mock.close = AsyncMock()
    return mock


def _write_temp_config(tmp_path: Path, **overrides) -> Path:
    """Write a temporary config JSON and return its path."""
    config = {
        "model": None,
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.95,
        "min_p": 0.0,
        "repeat_penalty": 1.1,
        "presence_penalty": 0.0,
    }
    config.update(overrides)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return config_path


# ── tests: config loading ────────────────────────────────────────

class TestConfigLoading:
    """Verify _load_config reads and parses JSON correctly."""

    def test_load_config_returns_dict(self, tmp_path):
        config_path = _write_temp_config(tmp_path, temperature=0.9)
        config = runner._load_config(str(config_path))
        assert isinstance(config, dict)
        assert config["temperature"] == 0.9

    def test_load_config_relative_path_resolved(self, tmp_path, monkeypatch):
        """Relative config path is resolved against project root (runner.py's parent)."""
        config_path = _write_temp_config(tmp_path, top_k=10)
        # Copy config to project root so relative path resolves
        project_root = Path(runner.__file__).resolve().parent
        (project_root / "config.json").write_text(config_path.read_text())
        try:
            config = runner._load_config("config.json")
            assert config["top_k"] == 10
        finally:
            (project_root / "config.json").unlink(missing_ok=True)

    def test_load_config_missing_field_defaults(self, tmp_path):
        """Config with fewer fields still loads without error."""
        minimal = {"temperature": 0.5}
        config_path = tmp_path / "minimal.json"
        config_path.write_text(json.dumps(minimal))
        config = runner._load_config(str(config_path))
        assert config["temperature"] == 0.5


# ── tests: task iteration ────────────────────────────────────────

class TestTaskIteration:
    """Verify _load_tasks and limit behavior."""

    def test_load_tasks_returns_list_of_15(self):
        tasks = runner._load_tasks()
        assert isinstance(tasks, list)
        assert len(tasks) == 15

    def test_task_structure(self):
        tasks = runner._load_tasks()
        task = tasks[0]
        assert "id" in task
        assert "type" in task
        assert "system_prompt" in task
        assert "user_prompt" in task

    def test_task_type_distribution(self):
        tasks = runner._load_tasks()
        types = [t["type"] for t in tasks]
        assert types.count("code") == 8
        assert types.count("qa") == 4
        assert types.count("reasoning") == 3

    def test_limit_truncates_tasks(self, tmp_path):
        """When limit is set, only the first N tasks are processed."""
        config_path = _write_temp_config(tmp_path)
        results_dir = tmp_path / "results"
        results_path = results_dir / "results.json"

        with patch.object(runner, "_load_config", return_value={"temperature": 0.7}), \
             patch.object(runner, "_load_tasks", return_value=[
                 {"id": 1, "type": "code", "system_prompt": "s", "user_prompt": "u"},
                 {"id": 2, "type": "qa", "system_prompt": "s", "user_prompt": "u"},
                 {"id": 3, "type": "reasoning", "system_prompt": "s", "user_prompt": "u"},
             ]), \
             patch("runner.LLMClient") as MockClient, \
             patch.object(runner, "_RESULTS_DIR", results_dir), \
             patch.object(runner, "_RESULTS_PATH", results_path):

            mock_client = _make_mock_client()
            MockClient.return_value = mock_client

            results = runner.run_benchmark(
                config_path=str(config_path),
                server_url="http://localhost:8080",
                limit=2,
            )

            assert len(results) == 2
            assert mock_client.complete.call_count == 2

    def test_no_limit_runs_all(self, tmp_path):
        """Without limit, all tasks are processed."""
        config_path = _write_temp_config(tmp_path)
        results_dir = tmp_path / "results"
        results_path = results_dir / "results.json"

        with patch.object(runner, "_load_config", return_value={"temperature": 0.7}), \
             patch.object(runner, "_load_tasks", return_value=[
                 {"id": i, "type": "code", "system_prompt": "s", "user_prompt": f"q{i}"}
                 for i in range(5)
             ]), \
             patch("runner.LLMClient") as MockClient, \
             patch.object(runner, "_RESULTS_DIR", results_dir), \
             patch.object(runner, "_RESULTS_PATH", results_path):

            mock_client = _make_mock_client()
            MockClient.return_value = mock_client

            results = runner.run_benchmark(
                config_path=str(config_path),
                server_url="http://localhost:8080",
            )

            assert len(results) == 5
            assert mock_client.complete.call_count == 5


# ── tests: results.json writing ──────────────────────────────────

class TestResultsWriting:
    """Verify results are written to results.json with correct structure."""

    def test_results_json_written_with_run_id(self, tmp_path):
        """results.json is written with a valid run_id (UUID)."""
        config_path = _write_temp_config(tmp_path)
        results_dir = tmp_path / "results"
        results_path = results_dir / "results.json"

        with patch.object(runner, "_load_config", return_value={"temperature": 0.7}), \
             patch.object(runner, "_load_tasks", return_value=[
                 {"id": 1, "type": "code", "system_prompt": "s", "user_prompt": "u"},
             ]), \
             patch("runner.LLMClient") as MockClient, \
             patch.object(runner, "_RESULTS_DIR", results_dir), \
             patch.object(runner, "_RESULTS_PATH", results_path):

            mock_client = _make_mock_client()
            MockClient.return_value = mock_client

            results = runner.run_benchmark(
                config_path=str(config_path),
                server_url="http://localhost:8080",
            )

            assert results_path.exists()
            data = json.loads(results_path.read_text())
            assert isinstance(data, dict)
            assert 'results' in data
            assert len(data['results']) == 1
            entry = data['results'][0]
            # Verify run_id is a valid UUID
            uuid.UUID(entry["run_id"])

    def test_results_entry_structure(self, tmp_path):
        """Each result entry has all required fields."""
        config_path = _write_temp_config(tmp_path)
        results_dir = tmp_path / "results"
        results_path = results_dir / "results.json"

        with patch.object(runner, "_load_config", return_value={"temperature": 0.7}), \
             patch.object(runner, "_load_tasks", return_value=[
                 {"id": 42, "type": "qa", "system_prompt": "sys", "user_prompt": "usr"},
             ]), \
             patch("runner.LLMClient") as MockClient, \
             patch.object(runner, "_RESULTS_DIR", results_dir), \
             patch.object(runner, "_RESULTS_PATH", results_path):

            mock_client = _make_mock_client()
            MockClient.return_value = mock_client

            results = runner.run_benchmark(
                config_path=str(config_path),
                server_url="http://localhost:8080",
            )

            entry = results[0]
            assert "run_id" in entry
            assert entry["task_id"] == 42
            assert entry["task_type"] == "qa"
            assert entry["prompt"] == "usr"
            assert entry["response"] == "mocked response"
            assert entry["score"] is not None
            assert "timestamp" in entry

    def test_results_have_scores_populated(self, tmp_path):
        """All result entries have score populated (not None) and within [0.0, 1.0]."""
        config_path = _write_temp_config(tmp_path)
        results_dir = tmp_path / "results"
        results_path = results_dir / "results.json"

        with patch.object(runner, "_load_config", return_value={"temperature": 0.7}), \
             patch.object(runner, "_load_tasks", return_value=[
                 {"id": 1, "type": "qa", "system_prompt": "s", "user_prompt": "u", "expected_keywords": ["hello"]},
                 {"id": 2, "type": "code", "system_prompt": "s", "user_prompt": "u", "expected_keywords": ["def"]},
                 {"id": 3, "type": "reasoning", "system_prompt": "s", "user_prompt": "u", "expected_keywords": []},
             ]), \
             patch("runner.LLMClient") as MockClient, \
             patch.object(runner, "_RESULTS_DIR", results_dir), \
             patch.object(runner, "_RESULTS_PATH", results_path):

            mock_client = _make_mock_client()
            MockClient.return_value = mock_client

            results = runner.run_benchmark(
                config_path=str(config_path),
                server_url="http://localhost:8080",
            )

            # All entries must have score populated
            for entry in results:
                assert entry["score"] is not None, f"score is None for task {entry['task_id']}"
                assert isinstance(entry["score"], float), f"score is not a float for task {entry['task_id']}"
                assert 0.0 <= entry["score"] <= 1.0, f"score out of range for task {entry['task_id']}: {entry['score']}"

            # Verify written file also has scores (data is dict with 'results' key)
            data = json.loads(results_path.read_text())
            results_list = data.get("results", data) if isinstance(data, dict) else data
            for entry in results_list:
                assert entry["score"] is not None
                assert isinstance(entry["score"], float)
                assert 0.0 <= entry["score"] <= 1.0

    def test_results_json_preserves_historical_runs(self, tmp_path):
        """Results are written as a complete JSON array, not appended."""
        config_path = _write_temp_config(tmp_path)
        results_dir = tmp_path / "results"
        results_path = results_dir / "results.json"

        with patch.object(runner, "_load_config", return_value={"temperature": 0.7}), \
             patch.object(runner, "_load_tasks", return_value=[
                 {"id": 1, "type": "code", "system_prompt": "s", "user_prompt": "u"},
             ]), \
             patch("runner.LLMClient") as MockClient, \
             patch.object(runner, "_RESULTS_DIR", results_dir), \
             patch.object(runner, "_RESULTS_PATH", results_path):

            mock_client = _make_mock_client()
            MockClient.return_value = mock_client

            # First run
            runner.run_benchmark(
                config_path=str(config_path),
                server_url="http://localhost:8080",
            )
            first_data = json.loads(results_path.read_text())

            # Second run — should overwrite, not append
            runner.run_benchmark(
                config_path=str(config_path),
                server_url="http://localhost:8080",
            )
            second_data = json.loads(results_path.read_text())

            # First run: 1 result
            assert len(first_data.get("results", first_data)) == 1
            # Second run: 2 results (1 historical + 1 new)
            assert len(second_data.get("results", second_data)) == 2

    def test_run_benchmark_returns_results_list(self, tmp_path):
        """run_benchmark returns the list of result dicts."""
        config_path = _write_temp_config(tmp_path)
        results_dir = tmp_path / "results"
        results_path = results_dir / "results.json"

        with patch.object(runner, "_load_config", return_value={"temperature": 0.7}), \
             patch.object(runner, "_load_tasks", return_value=[
                 {"id": 1, "type": "code", "system_prompt": "s", "user_prompt": "u"},
             ]), \
             patch("runner.LLMClient") as MockClient, \
             patch.object(runner, "_RESULTS_DIR", results_dir), \
             patch.object(runner, "_RESULTS_PATH", results_path):

            mock_client = _make_mock_client()
            MockClient.return_value = mock_client

            result = runner.run_benchmark(
                config_path=str(config_path),
                server_url="http://localhost:8080",
            )

            assert isinstance(result, list)
            assert len(result) == 1

    def test_client_close_called_even_on_error(self, tmp_path):
        """LLMClient.close() is called in the finally block."""
        config_path = _write_temp_config(tmp_path)
        results_dir = tmp_path / "results"

        with patch.object(runner, "_load_config", return_value={"temperature": 0.7}), \
             patch.object(runner, "_load_tasks", return_value=[
                 {"id": 1, "type": "code", "system_prompt": "s", "user_prompt": "u"},
             ]), \
             patch("runner.LLMClient") as MockClient, \
             patch.object(runner, "_RESULTS_DIR", results_dir):

            mock_client = _make_mock_client()
            mock_client.complete.side_effect = Exception("server down")
            MockClient.return_value = mock_client

            with pytest.raises(Exception, match="server down"):
                runner.run_benchmark(
                    config_path=str(config_path),
                    server_url="http://localhost:8080",
                )

            # close() should still be called
            mock_client.close.assert_called_once()
