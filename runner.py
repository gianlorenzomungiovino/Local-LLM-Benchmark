"""Task runner — orchestrates benchmark execution against a local OpenAI-compatible server."""

import asyncio
import json
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from llm_client import (
    LLMClient,
    OutputOverflow,
    StallTimeout,
    TaskDeadlineExceeded,
    _resolve_reasoning_effort,
)
from evaluator import score_results

# Concurrency: sequential to avoid server overload
# Completion is stream-driven ([DONE]); time gates are activity-based
# (idle watchdog + adaptive per-task deadline, see LLMClient)


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


def _save_partial_results(
    all_results: list[dict],
    original_configs: list[dict],
    all_configs: list[dict],
) -> None:
    """Write current results state to results.json.

    Called after each task so that an interruption (Ctrl+C, crash)
    never loses already-completed work.
    """
    output = {
        "results": all_results,
        "original_configs": original_configs,
        "configs": all_configs,
    }
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def _load_launch_recipe(api_config: dict) -> dict:
    """Read the launch-recipe file (api_config['launch_config']) verbatim.

    The recipe documents how the user started the server (engine flags,
    any engine). It is never interpreted — only stamped for provenance.
    Returns an empty dict if absent or unreadable.
    """
    path = api_config.get("launch_config")
    if not path:
        return {}
    recipe_file = Path(path)
    if not recipe_file.is_absolute():
        recipe_file = _PROJECT_ROOT / recipe_file
    try:
        with open(recipe_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] Launch recipe unreadable: {exc}", file=sys.stderr)
        return {}


def _build_run_stamp(api_config: dict) -> dict:
    """Build the flat stamp recorded in results for this run.

    Combines the launch recipe verbatim (provenance of the flags the
    server was started with) plus the resolved API identity, reasoning
    level and guardrails actually used by the run.
    """
    model = api_config["models"][0]
    level = api_config.get("thinking_level")
    effort = _resolve_reasoning_effort(api_config)
    stamp = dict(_load_launch_recipe(api_config))
    stamp.update({
        "model": model.get("id"),
        "baseUrl": api_config["baseUrl"],
        "thinking_level": level,
        "reasoning_effort": effort,
        "n_ctx": model.get("contextWindow"),
        "max_tokens": model.get("maxTokens"),
    })
    return stamp


def run_benchmark(config_path: str = "configs/models.json", limit: int | None = None) -> list[dict]:
    """Run all (or *limit*) benchmark tasks against the server and return results.

    Args:
        config_path: Path to models.json (baseUrl + model + per-run choices).
        limit: If set, run only the first N tasks (for debugging).

    Returns:
        List of result dicts written to results/results.json.
    """
    # Load API configuration (never rewritten at runtime)
    config = _load_config(config_path)
    run_stamp = _build_run_stamp(config)

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
    # Keep ALL previous results regardless of task type — historical data
    # must never be silently dropped when tasks.json is restructured.
    previous_results: list[dict] = []
    previous_configs: list[dict] = []
    original_configs: list[dict] = []
    if _RESULTS_PATH.exists() and _RESULTS_PATH.stat().st_size > 0:
        with open(_RESULTS_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                raw_data = json.loads(content)
                if isinstance(raw_data, dict):
                    # New format: dict with "results", "original_configs", "configs"
                    raw_results = raw_data.get("results", [])
                    # original_configs: saved by run.py BEFORE detection
                    # configs: saved by runner.py AFTER detection (legacy, may be empty)
                    # We prefer original_configs for the report (has all params).
                    original_configs = raw_data.get("original_configs", [])
                    legacy_configs = raw_data.get("configs", [])
                    # Use original_configs if available; fall back to legacy
                    previous_configs = original_configs if original_configs else legacy_configs
                else:
                    # Old format: flat list (no configs saved)
                    raw_results = raw_data if isinstance(raw_data, list) else []
                # Preserve ALL historical results (no type filter)
                previous_results = list(raw_results)

    async def _execute():
        # Load timeout/guardrail params from separate file
        timeout_path = _PROJECT_ROOT / "configs" / "timeout.json"
        with open(timeout_path, "r", encoding="utf-8") as f:
            timeout_config = json.load(f)
        client = LLMClient(config, timeout_config=timeout_config)
        # Extract model name for embedding in results (single source: models.json)
        model_name = config["models"][0]["id"]
        try:
            rome_now = datetime.now(ZoneInfo("Europe/Rome"))

            async def _run_task(task):
                t0 = asyncio.get_running_loop().time()
                response = await client.complete(
                    system_prompt=task["system_prompt"],
                    user_prompt=task["user_prompt"],
                )
                elapsed = asyncio.get_running_loop().time() - t0
                elapsed_map[task["id"]] = elapsed
                return {
                    "run_id": run_id,
                    "task_id": task["id"],
                    "task_type": task["type"],
                    "difficulty": task.get("difficulty", "medium"),
                    "prompt": task["user_prompt"],
                    "response": response,
                    "score": None,
                    "timestamp": rome_now.isoformat(),
                    "model": model_name,
                }

            # Execute tasks sequentially; each task ends when its stream
            # completes, or when an activity gate trips (idle/output/deadline)
            for task in tasks:
                try:
                    result = await _run_task(task)
                    results.append(result)
                except (StallTimeout, OutputOverflow, TaskDeadlineExceeded, httpx.HTTPError) as exc:
                    print(f"[ERROR] Task {task['id']}: {exc}", file=sys.stderr)
                    results.append({
                        "run_id": run_id,
                        "task_id": task["id"],
                        "task_type": task["type"],
                        "difficulty": task.get("difficulty", "medium"),
                        "prompt": task["user_prompt"],
                        "response": "",
                        "score": 0.0,
                        "timestamp": rome_now.isoformat(),
                        "error": str(exc),
                        "model": model_name,
                    })
                # Incremental save: persist after each task so an
                # interruption (Ctrl+C, crash) never loses completed work.
                _save_partial_results(
                    previous_results + results,
                    original_configs,
                    previous_configs + [run_stamp],
                )
        finally:
            await client.close()

    try:
        asyncio.run(_execute())
    except (KeyboardInterrupt, asyncio.CancelledError) as exc:
        print(f"[WARN] Interrupted — saving {len(results)} partial results", file=sys.stderr)
        # Results already saved incrementally; just score what we have.

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
    all_results = previous_results + results
    all_configs = previous_configs + [run_stamp]

    # Final write (with scores) — overwrites the incremental saves
    _save_partial_results(all_results, original_configs, all_configs)

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
