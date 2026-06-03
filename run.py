#!/usr/bin/env python3
"""CLI entrypoint — run the LLM benchmark against a llama.cpp server."""

import argparse
import json
import sys
from pathlib import Path

from runner import run_benchmark

_PROJECT_ROOT = Path(__file__).resolve().parent


def _print_banner(task_count: int, config: dict, server_url: str, limit: int | None) -> None:
    """Print a startup banner with config and task info to stderr."""
    sep = "=" * 55
    print("\n" + sep, file=sys.stderr)
    print("  LLM Benchmark — Starting", file=sys.stderr)
    print(sep, file=sys.stderr)
    if limit:
        print(f"  Tasks: {task_count} (limited to {limit})", file=sys.stderr)
    else:
        print(f"  Tasks: {task_count}", file=sys.stderr)
    print(f"  Server: {server_url}", file=sys.stderr)
    print("  Config:", file=sys.stderr)
    for key, value in config.items():
        print(f"    {key}: {value}", file=sys.stderr)
    print(sep + "\n", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LLM benchmark against a llama.cpp server.",
    )
    parser.add_argument(
        "--server",
        type=str,
        default=None,
        help="Override server URL (default: from config or http://localhost:8080)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N tasks (for debugging)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/run.json",
        help="Path to server config JSON (default: configs/run.json)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        default=False,
        help="Generate markdown report after benchmark run",
    )
    args = parser.parse_args()

    server_url = args.server or "http://localhost:8080"
    config_path = args.config

    # Resolve config path relative to project root if not absolute
    if not Path(config_path).is_absolute():
        config_path = str(_PROJECT_ROOT / config_path)

    # Load config to know task count and params before running
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Load tasks to count them
    tasks_path = _PROJECT_ROOT / "tasks" / "tasks.json"
    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    task_count = len(tasks)
    if args.limit is not None:
        task_count = min(args.limit, len(tasks))

    # Print startup banner
    _print_banner(task_count, config, server_url, args.limit)

    results = run_benchmark(
        config_path=config_path,
        server_url=server_url,
        limit=args.limit,
    )

    print(f"\nCompleted {len(results)} tasks.", file=sys.stderr)

    if args.report:
        from report import generate_report_from_file

        generate_report_from_file()
        print("Report written to results/results.md", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
