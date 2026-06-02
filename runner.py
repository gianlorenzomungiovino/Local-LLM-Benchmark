"""Task runner — orchestrates benchmark execution against a llama.cpp server."""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from llm_client import LLMClient

# Paths relative to this file's parent (project root)
_PROJECT_ROOT = Path(__file__).resolve().parent
_TASKS_PATH = _PROJECT_ROOT / "tasks" / "tasks.json"
_RESULTS_DIR = _PROJECT_ROOT / "results"
_RESULTS_PATH = _RESULTS_DIR / "results.json"


def run_benchmark(config_path: str = "configs/run.json", server_url: str = "http://localhost:8080", limit: int | None = None) -> list[dict]:
    """Run all (or *limit*) benchmark tasks against the server and return results.

    Args:
        config_path: Path to JSON config (temperature, top_k, etc.).
        server_url: llama.cpp server URL, e.g. 'http://localhost:8080'.
        limit: If set, run only the first N tasks (for debugging).

    Returns:
        List of result dicts written to results/results.json.
    """
    # Load config
    config = _load_config(config_path)

    # Load tasks
    tasks = _load_tasks()

    # Apply limit
    if limit is not None:
        tasks = tasks[:limit]

    run_id = str(uuid.uuid4())
    results: list[dict] = []

    async def _execute():
        client = LLMClient(server_url, **config)
        try:
            for task in tasks:
                response = await client.complete(
                    system_prompt=task["system_prompt"],
                    user_prompt=task["user_prompt"],
                )
                results.append({
                    "run_id": run_id,
                    "task_id": task["id"],
                    "task_type": task["type"],
                    "prompt": task["user_prompt"],
                    "response": response,
                    "score": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
        finally:
            await client.close()

    asyncio.run(_execute())

    # Write results
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def _load_config(path: str) -> dict:
    """Load server config from JSON file."""
    config_file = Path(path)
    if not config_file.is_absolute():
        config_file = _PROJECT_ROOT / config_file
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_tasks() -> list[dict]:
    """Load benchmark tasks from tasks/tasks.json."""
    with open(_TASKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
