"""Markdown report generator for benchmark results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def generate_report(
    results: list[dict[str, Any]],
    config: dict[str, Any],
    output_path: str = "results/results.md",
) -> str:
    """Generate a human-readable markdown report from benchmark results.

    Sections:
    - **Header** with run_id and timestamp from the first result.
    - **Task Breakdown** table (Task ID, Type, Score, Matched Keywords).
    - **Configuration** table (parameter name, value).
    - **Parameter Averages** — mean score per task type.
    - **Ranking** — average score and rank for the single config.

    Args:
        results: List of result dicts (each with task_id, task_type, score,
            matched_keywords, run_id, timestamp).
        config: Configuration dict with parameter names and values.
        output_path: File path to write the markdown report.

    Returns:
        The generated markdown string.
    """
    lines: list[str] = []

    # --- Header ---
    lines.append("# Benchmark Results\n")

    if results:
        first = results[0]
        run_id = first.get("run_id", "unknown")
        ts = first.get("timestamp", "unknown")
        lines.append(f"- **Run ID:** `{run_id}`")
        lines.append(f"- **Timestamp:** {ts}")
    else:
        lines.append("- **Run ID:** N/A")
        lines.append("- **Timestamp:** N/A")
    lines.append("")

    # --- Task Breakdown ---
    lines.append("## Task Breakdown\n")
    lines.append("| Task ID | Type | Score | Matched Keywords |")
    lines.append("|---------|------|-------|------------------|")

    for r in results:
        task_id = r.get("task_id", "?")
        task_type = r.get("task_type", "unknown")
        score = r.get("score")
        score_str = f"{score:.4f}" if score is not None else "N/A"
        matched = r.get("matched_keywords") or []
        kw_str = ", ".join(matched) if matched else "—"
        lines.append(f"| {task_id} | {task_type} | {score_str} | {kw_str} |")

    lines.append("")

    # --- Configuration ---
    lines.append("## Configuration\n")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")

    for key, value in config.items():
        display_value = str(value)
        lines.append(f"| {key} | `{display_value}` |")

    lines.append("")

    # --- Parameter Averages ---
    lines.append("## Parameter Averages\n")

    type_scores: dict[str, list[float]] = {}
    for r in results:
        tt = r.get("task_type", "unknown")
        sc = r.get("score")
        if sc is not None:
            type_scores.setdefault(tt, []).append(sc)

    if type_scores:
        lines.append("| Task Type | Average Score |")
        lines.append("|-----------|---------------|")
        for tt in sorted(type_scores):
            scores = type_scores[tt]
            avg = sum(scores) / len(scores)
            lines.append(f"| {tt} | {avg:.4f} |")
    else:
        lines.append("No scored results available.")

    lines.append("")

    # --- Ranking ---
    lines.append("## Ranking\n")

    if type_scores:
        all_scores: list[float] = []
        for scores in type_scores.values():
            all_scores.extend(scores)
        overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
        lines.append(f"| Configuration | Average Score | Rank |")
        lines.append("|---------------|---------------|------|")
        lines.append(f"| current | {overall_avg:.4f} | 1 |")
    else:
        lines.append("No scored results available for ranking.")

    lines.append("")

    markdown = "\n".join(lines)

    # Write to file
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")

    return markdown


def generate_report_from_file(
    results_path: str = "results/results.json",
    config_path: str = "configs/run.json",
    output_path: str = "results/results.md",
) -> str:
    """Load results and config from JSON files and generate a report.

    Convenience wrapper around ``generate_report`` that reads from disk.

    Args:
        results_path: Path to the results JSON file.
        config_path: Path to the config JSON file.
        output_path: Path for the markdown output.

    Returns:
        The generated markdown string.
    """
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return generate_report(results, config, output_path)
