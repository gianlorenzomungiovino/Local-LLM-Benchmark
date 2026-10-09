"""Markdown report generator for benchmark results."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _format_config_value(value: Any, max_len: int = 160) -> str:
    """Render a config value for a markdown table cell.

    Nested structures (list/dict) are serialized as compact JSON; long values
    are truncated (full values remain in results.json). Pipes are escaped so
    cells cannot break the table layout.
    """
    if isinstance(value, (list, dict)):
        s = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    else:
        s = str(value)
    if len(s) > max_len:
        s = s[: max_len - 1] + "\u2026"
    return s.replace("|", "\\|")


def _format_timestamp(ts: str) -> str:
    """Convert an ISO timestamp to Europe/Rome display format."""
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        from zoneinfo import ZoneInfo
        dt_rome = dt.astimezone(ZoneInfo("Europe/Rome"))
        return dt_rome.strftime("%Y-%m-%d %H:%M:%S %Z (UTC%z)")
    except Exception:
        return ts


def _avg(scores: list[float]) -> float:
    """Compute average of a list of floats."""
    return sum(scores) / len(scores) if scores else 0.0


def generate_report(
    results: list[dict[str, Any]],
    configs: list[dict[str, Any]] | dict[str, Any] | None = None,
    original_configs: list[dict[str, Any]] | None = None,
    output_path: str | None = None,
) -> str:
    """Generate a human-readable markdown report from benchmark results.

    Sections:
    - **Run List** - all runs with run_id, timestamp, task count, and per-run config.
    - **Per-Run Task Breakdown** - table nested under each run.
    - **Per-Run Parameter Averages** - mean score per task type per run.
    - **Global Parameter Averages** - mean score per task type across all runs.
    - **Ranking** - average score and rank per run.

    Args:
        results: Flat list of all result dicts.
        configs: Either a single config dict (shared across all runs)
                 or a list of config dicts, one per run (in run order).
    """
    lines: list[str] = []

    if configs is None:
        configs = {}

    # --- Header ---
    lines.append("# Benchmark Results\n")

    if results:
        runs: dict[str, list[dict]] = defaultdict(list)
        for r in results:
            runs[r.get("run_id", "unknown")].append(r)

        run_list = sorted(runs.keys(), key=lambda rid: (
            min((r.get("timestamp", "") for r in runs[rid]), default="")
        ))

        lines.append(f"**{len(run_list)} run(s) recorded.**\n")

        # Ground-truth model per run, taken from the result entries.
        run_models_per_run: list[str | None] = [
            next(
                (r.get("model") for r in runs[rid] if r.get("model")),
                None,
            )
            for rid in run_list
        ]

        def _pick_configs(pool: list[dict]) -> list[dict]:
            """Pair config entries with runs, matching by model name.

            The configs arrays grow on every run.py invocation, including
            invocations that produced no results, so positional indexing is
            unreliable when len(pool) != len(runs). Prefer the earliest unused
            entry whose model matches the run's model.
            """
            picked: list[dict] = []
            used: set[int] = set()
            for _ in range(len(run_models_per_run)):
                idx: int | None = None
                model = run_models_per_run[len(picked)]
                if model:
                    idx = next(
                        (
                            j
                            for j, cfg in enumerate(pool)
                            if j not in used and cfg.get("model") == model
                        ),
                        None,
                    )
                if idx is None and len(pool) == len(run_models_per_run):
                    # Same cardinality: the arrays are aligned by construction.
                    idx = len(picked)
                if idx is None:
                    idx = next(
                        (j for j in range(len(pool)) if j not in used), None
                    )
                if idx is None:
                    picked.append({})
                    continue
                used.add(idx)
                picked.append(dict(pool[idx]))
            return picked

        # --- Per-Run: header + config + breakdown + averages (all nested) ---
        for i, run_id in enumerate(run_list, 1):
            run_results = runs[run_id]
            ts = run_results[0].get("timestamp", "unknown")
            if i > 1:
                lines.append("---")
                lines.append("")

            lines.append(f"### Run {i}: `{run_id}`\n")
            lines.append(f"- **Timestamp:** {_format_timestamp(ts)}")
            lines.append(f"- **Tasks:** {len(run_results)}")
            lines.append("")

            # Per-run configuration
            # Use original_configs (saved before detection) for the full param
            # set. Fall back to configs (post-detection), then shared dict.
            run_config = {}
            if isinstance(original_configs, list) and original_configs:
                run_config = _pick_configs(original_configs)[i - 1]
            elif isinstance(configs, list) and configs:
                run_config = _pick_configs(configs)[i - 1]
            elif isinstance(configs, dict) and configs:
                run_config = dict(configs) if i == len(run_list) else {}

            # Override model with the one from result entries (ground truth)
            models = [r.get("model") for r in run_results if r.get("model")]
            if models and models[0] != "unknown":
                run_config["model"] = models[0]

            if run_config:
                lines.append("#### Configuration\n")
                lines.append("| Parameter | Value |")
                lines.append("|-----------|-------|")
                nested: dict[str, Any] = {}
                for key, value in run_config.items():
                    if isinstance(value, (dict, list)):
                        nested[key] = value  # rendered verbatim below, not in the table
                        continue
                    lines.append(f"| {key} | `{_format_config_value(value)}` |")
                lines.append("")
                if nested:
                    lines.append("##### Launch parameters (verbatim, from the stamped recipe)\n")
                    lines.append("```json")
                    lines.append(json.dumps(nested, indent=2, ensure_ascii=False))
                    lines.append("```")
                    lines.append("")

            # Per-Run Task Breakdown
            lines.append("#### Task Breakdown\n")
            lines.append("| Task ID | Type | Difficulty | Score | Matched Keywords |")
            lines.append("|---------|------|------------|-------|------------------|")

            for r in run_results:
                task_id = r.get("task_id", "?")
                task_type = r.get("task_type", "unknown")
                difficulty = r.get("difficulty", "medium")
                score = r.get("score")
                score_str = f"{score:.4f}" if score is not None else "N/A"
                matched = r.get("matched_keywords") or []
                kw_str = ", ".join(matched) if matched else "\u2014"
                # Add extra scoring details for code_execute tasks
                extra = ""
                if r.get("test_score") is not None:
                    extra = f" (test_patterns: {r['test_score']:.2f})"
                if r.get("constraint_score") is not None:
                    extra += f" (constraints: {r['constraint_score']:.2f})"
                if r.get("numeric_score") is not None:
                    extra += f" (numeric: {r['numeric_score']:.2f})"
                lines.append(f"| {task_id} | {task_type} | {difficulty} | {score_str}{extra} | {kw_str} |")

            lines.append("")

            # Per-Run Parameter Averages
            type_scores_run: dict[str, list[float]] = {}
            for r in run_results:
                tt = r.get("task_type", "unknown")
                sc = r.get("score")
                if sc is not None:
                    type_scores_run.setdefault(tt, []).append(sc)

            if type_scores_run:
                lines.append("##### Parameter Averages\n")
                lines.append("| Task Type | Average Score |")
                lines.append("|-----------|---------------|")
                for tt in sorted(type_scores_run):
                    scores = type_scores_run[tt]
                    lines.append(f"| {tt} | {_avg(scores):.4f} |")
                lines.append("")
            else:
                lines.append("##### Parameter Averages\n")
                lines.append("No scored results available.")
                lines.append("")

        # --- Global Parameter Averages ---
        lines.append("## Global Parameter Averages\n")
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
                lines.append(f"| {tt} | {_avg(scores):.4f} |")
        else:
            lines.append("No scored results available.")
        lines.append("")

        # --- Ranking ---
        lines.append("## Ranking\n")
        lines.append("| Run | Model | Average Score | Rank |")
        lines.append("|-----|-------|---------------|------|")

        run_id_to_number = {rid: i for i, rid in enumerate(run_list, 1)}

        run_avgs: list[tuple[str, str, float]] = []
        for run_id in run_list:
            run_results = runs[run_id]
            run_scores = [r.get("score") for r in run_results if r.get("score") is not None]
            avg = _avg(run_scores)

            # Prefer model name from result entries (embedded at generation time,
            # survives config file overwrites). Fall back to per-run config,
            # then to shared config dict.
            model_name = "unknown"
            # 1) Check result entries first (most reliable)
            models = [r.get("model") for r in run_results if r.get("model")]
            if models:
                model_name = models[0]
            else:
                # 2) Fall back to per-run config
                run_num = run_id_to_number.get(run_id, 0)
                if isinstance(configs, list) and len(configs) >= run_num:
                    cfg = configs[run_num - 1]
                    model_name = cfg.get("model", "unknown")
                elif isinstance(configs, dict):
                    model_name = configs.get("model", "unknown")

            run_avgs.append((run_id, model_name, avg))

        run_avgs.sort(key=lambda x: x[2], reverse=True)
        for rank, (run_id, model_name, avg) in enumerate(run_avgs, 1):
            run_num = run_id_to_number.get(run_id, "?")
            lines.append(f"| {run_num} | `{model_name}` | {avg:.4f} | {rank} |")
    else:
        lines.append("No results available.\n")

    lines.append("")

    markdown = "\n".join(lines)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")

    return markdown


def generate_report_from_file(
    results_path: str = "results/results.json",
    config_path: str = "configs/run.json",
    output_path: str = "results/results.md",
) -> str:
    """Load results and per-run configs from results.json and generate a report.

    If results.json contains "original_configs" (saved before detection),
    uses that for Configuration sections. Falls back to "configs" (post-detection).
    """
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both old format (flat list) and new format (dict with results+configs)
    if isinstance(data, dict):
        results = data.get("results", [])
        original_configs = data.get("original_configs", None)
        configs = data.get("configs", None)
    else:
        results = data
        original_configs = None
        configs = None

    # No fallback to config file: the file may have been overwritten
    # by a later run's detection, so its model name would be wrong
    # for historical runs. If a run has no config, show "unknown".

    return generate_report(results, configs, original_configs, output_path)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Regenerate results/results.md from results/results.json",
    )
    parser.add_argument(
        "--results",
        default="results/results.json",
        help="Path to the results JSON file",
    )
    parser.add_argument(
        "--output",
        default="results/results.md",
        help="Path of the markdown report to write",
    )
    args = parser.parse_args()

    markdown = generate_report_from_file(
        results_path=args.results,
        output_path=args.output,
    )
    print(f"Report written to {args.output} ({len(markdown)} chars)", file=sys.stderr)
