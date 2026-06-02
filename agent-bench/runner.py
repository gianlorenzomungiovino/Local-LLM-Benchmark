"""Task execution runner — loads tasks, sends requests, saves results."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_client import LLMClient
from evaluator import evaluate


def load_tasks(tasks_path: str | Path) -> list[dict[str, Any]]:
    """Load tasks from JSON file."""
    path = Path(tasks_path)
    if not path.exists():
        raise FileNotFoundError(f"Tasks file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    if not isinstance(tasks, list):
        raise ValueError("tasks.json must contain a JSON array")

    for i, task in enumerate(tasks):
        for field in ("id", "prompt", "type", "expected"):
            if field not in task:
                raise ValueError(f"Task at index {i} missing field: {field}")

    return tasks


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load run configuration from JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Validate required fields
    required = ("temperature", "top_k", "top_p", "min_p", "repeat_penalty", "presence_penalty")
    for field in required:
        if field not in config:
            raise ValueError(f"Config missing required field: {field}")

    return config


def generate_run_id() -> str:
    """Generate a unique run ID based on timestamp."""
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d-%H%M")
    short_uuid = uuid.uuid4().hex[:4]
    return f"run-{timestamp}-{short_uuid}"


def run_benchmark(
    server_url: str,
    config: dict[str, Any],
    tasks: list[dict[str, Any]],
    results_dir: str | Path = "results",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Run the full benchmark.

    Args:
        server_url: Base URL of the LLM server.
        config: Run configuration with sampling parameters.
        tasks: List of task dicts.
        results_dir: Directory to save results.
        limit: Optional limit on number of tasks to run (for debugging).

    Returns:
        List of result dicts.
    """
    results_path = Path(results_dir) / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if any
    existing_results: list[dict[str, Any]] = []
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            existing_results = json.load(f)

    client = LLMClient(server_url)
    run_id = generate_run_id()

    # Store server params in config for the run
    run_config = {
        "run_id": run_id,
        "model": config.get("model", ""),
        "temperature": config["temperature"],
        "top_k": config["top_k"],
        "top_p": config["top_p"],
        "min_p": config["min_p"],
        "repeat_penalty": config["repeat_penalty"],
        "presence_penalty": config["presence_penalty"],
    }

    tasks_to_run = tasks[:limit] if limit else tasks
    results: list[dict[str, Any]] = []

    for task in tasks_to_run:
        task_result = {
            "task_id": task["id"],
            "task_type": task["type"],
            "prompt": task["prompt"],
            "params": run_config,
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": "",
            "output": "",
            "score": 0.0,
            "notes": "",
        }

        try:
            response = client.send_with_retry(task["prompt"], run_config)
            task_result["output"] = response["content"]
            task_result["model"] = response["model"]

            # Evaluate
            score_result = evaluate(task["type"], response["content"], task["expected"])
            task_result["score"] = score_result["score"]
            task_result["notes"] = score_result["notes"]

        except Exception as e:
            task_result["notes"] = f"Error: {e}"

        results.append(task_result)

    # Save accumulated results
    all_results = existing_results + results
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return results


def generate_report(results: list[dict[str, Any]], output_path: str | Path = "results/results.md") -> str:
    """Generate a markdown report from benchmark results.

    Groups results by run_id and parameter configuration,
    shows average score per configuration, and ranks configurations.

    Returns:
        The generated markdown string.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Group by run_id
    runs: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        run_id = result.get("run_id", "unknown")
        runs.setdefault(run_id, []).append(result)

    # Build report
    lines = [
        "# Benchmark Results",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Total runs:** {len(runs)}",
        f"**Total tasks:** {len(results)}",
        "",
        "---",
        "",
    ]

    # Summary table: run_id, avg score, best task score
    run_summaries = []
    for run_id, run_results in runs.items():
        scores = [r["score"] for r in run_results if r["score"] > 0]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        best_score = max((r["score"] for r in run_results), default=0.0)
        params = run_results[0].get("params", {}) if run_results else {}
        run_summaries.append({
            "run_id": run_id,
            "avg_score": avg_score,
            "best_score": best_score,
            "total": len(run_results),
            "params": params,
        })

    # Rank runs by average score
    run_summaries.sort(key=lambda x: x["avg_score"], reverse=True)

    lines.append("## Summary by Run")
    lines.append("")
    lines.append("| Rank | Run ID | Avg Score | Best Score | Tasks | Params |")
    lines.append("|------|--------|-----------|------------|-------|--------|")

    for rank, summary in enumerate(run_summaries, 1):
        params_str = (
            f"t={summary['params'].get('temperature', '?')}, "
            f"top_k={summary['params'].get('top_k', '?')}, "
            f"top_p={summary['params'].get('top_p', '?')}"
        )
        lines.append(
            f"| {rank} | {summary['run_id']} | "
            f"{summary['avg_score']:.2f} | "
            f"{summary['best_score']:.2f} | "
            f"{summary['total']} | "
            f"{params_str} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    # Detailed results per run
    for summary in run_summaries:
        run_id = summary["run_id"]
        run_results = runs[run_id]

        lines.append(f"## {run_id}")
        lines.append("")
        lines.append(f"**Avg Score:** {summary['avg_score']:.2f} | **Best:** {summary['best_score']:.2f}")
        lines.append("")
        lines.append("| Task ID | Type | Score | Notes |")
        lines.append("|---------|------|-------|-------|")

        for r in run_results:
            notes = r.get("notes", "").replace("\n", " ").replace("|", "\\|")[:80]
            lines.append(
                f"| {r['task_id']} | {r['task_type']} | "
                f"{r['score']:.2f} | {notes} |"
            )

        lines.append("")
        lines.append("---")
        lines.append("")

    report = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return report
