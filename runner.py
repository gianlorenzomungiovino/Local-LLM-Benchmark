"""Task runner — orchestrates benchmark execution against a llama.cpp server."""

import asyncio
import json
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from llm_client import LLMClient
from evaluator import score_results

# Concurrency: sequential to avoid server overload
# Per-request timeout: 240s (fast failure) vs global 300s (safety net)
REQUEST_TIMEOUT = 240  # seconds per single request


class ProgressTracker:
    """Track and display benchmark progress on stderr."""

    def __init__(self, total: int):
        self.total = total
        self._type_times: dict[str, list[float]] = defaultdict(list)
        self._type_scores: dict[str, list[float]] = defaultdict(list)
        self._type_counts: dict[str, int] = defaultdict(int)

    def log_task(
        self,
        task_id: str | int,
        task_type: str,
        elapsed_seconds: float,
        response_length: int,
        score: float | None,
    ) -> None:
        """Log a single task result to stderr.

        Format: [N/total] type:id → Xs, Y chars, score:Z
        """
        n = self._type_counts[task_type] + 1
        self._type_counts[task_type] += 1
        self._type_times[task_type].append(elapsed_seconds)
        if score is not None:
            self._type_scores[task_type].append(score)

        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(
            f"[{n}/{self.total}] {task_type}:{task_id} → "
            f"{elapsed_seconds:.1f}s, {response_length} chars, score:{score_str}",
            file=sys.stderr,
        )

    def print_summary(self) -> None:
        """Print a final summary with averages per type and overall score."""
        print("\n── Benchmark Summary ──", file=sys.stderr)
        for ttype in sorted(self._type_times):
            times = self._type_times[ttype]
            scores = self._type_scores.get(ttype, [])
            avg_time = sum(times) / len(times)
            avg_score = sum(scores) / len(scores) if scores else 0.0
            print(
                f"  {ttype}: {len(times)} tasks, "
                f"avg time {avg_time:.1f}s, avg score {avg_score:.4f}",
                file=sys.stderr,
            )
        all_scores = []
        for scores in self._type_scores.values():
            all_scores.extend(scores)
        if all_scores:
            overall = sum(all_scores) / len(all_scores)
            print(f"  Overall: {len(all_scores)} tasks, avg score {overall:.4f}", file=sys.stderr)
        print("── End Summary ──\n", file=sys.stderr)

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
    elapsed_map: dict[str, float] = {}  # task_id → elapsed_seconds
    tracker = ProgressTracker(len(tasks))

    # Load previous results to preserve historical runs
    previous_results: list[dict] = []
    if _RESULTS_PATH.exists() and _RESULTS_PATH.stat().st_size > 0:
        with open(_RESULTS_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                previous_results = json.loads(content)

    async def _execute():
        client = LLMClient(server_url, **config)
        try:
            rome_now = datetime.now(ZoneInfo("Europe/Rome"))

            async def _run_task(task):
                t0 = asyncio.get_event_loop().time()
                response = await asyncio.wait_for(
                    client.complete(
                        system_prompt=task["system_prompt"],
                        user_prompt=task["user_prompt"],
                    ),
                    timeout=REQUEST_TIMEOUT,
                )
                elapsed = asyncio.get_event_loop().time() - t0
                elapsed_map[task["id"]] = elapsed
                return {
                    "run_id": run_id,
                    "task_id": task["id"],
                    "task_type": task["type"],
                    "prompt": task["user_prompt"],
                    "response": response,
                    "score": None,
                    "timestamp": rome_now.isoformat(),
                }

            # Execute tasks sequentially with per-request timeout
            # This prevents the last tasks from timing out due to server overload
            for task in tasks:
                try:
                    result = await _run_task(task)
                    results.append(result)
                except asyncio.TimeoutError:
                    print(
                        f"[ERROR] Task {task['id']} timed out after {REQUEST_TIMEOUT}s",
                        file=sys.stderr,
                    )
                    results.append({
                        "run_id": run_id,
                        "task_id": task["id"],
                        "task_type": task["type"],
                        "prompt": task["user_prompt"],
                        "response": "",
                        "score": 0.0,
                        "timestamp": rome_now.isoformat(),
                        "error": f"Timeout after {REQUEST_TIMEOUT}s",
                    })
        finally:
            await client.close()

    asyncio.run(_execute())

    # Score results before writing
    score_results(results, tasks)

    # Log each task with its score (now computed)
    for result in results:
        tracker.log_task(
            task_id=result["task_id"],
            task_type=result["task_type"],
            elapsed_seconds=elapsed_map.get(result["task_id"], 0.0),
            response_length=len(result.get("response", "")),
            score=result.get("score"),
        )

    tracker.print_summary()

    # Merge with previous results (preserve historical runs)
    if isinstance(previous_results, dict):
        # New format: dict with "results" and "configs"
        prev_results_list = previous_results.get("results", [])
        prev_configs = previous_results.get("configs", [])
    else:
        # Old format: flat list
        prev_results_list = previous_results if isinstance(previous_results, list) else []
        prev_configs = []

    all_results = prev_results_list + results
    all_configs = prev_configs + [config]

    # Write results + configs
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {"results": all_results, "configs": all_configs}
    with open(_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return all_results


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
