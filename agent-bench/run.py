#!/usr/bin/env python3
"""Agent Bench — CLI entrypoint for LLM benchmarking.

Usage:
    python run.py
    python run.py --server http://127.0.0.1:8080
    python run.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from runner import load_tasks, load_config, run_benchmark, generate_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark an OpenAI-compatible LLM server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--server",
        default=None,
        help="Server base URL (default: from configs/run.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks to run (for debugging)",
    )
    parser.add_argument(
        "--config",
        default="configs/run.json",
        help="Path to config file (default: configs/run.json)",
    )
    parser.add_argument(
        "--tasks",
        default="tasks/tasks.json",
        help="Path to tasks file (default: tasks/tasks.json)",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for results (default: results)",
    )

    args = parser.parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    # Override server URL if provided
    server_url = args.server or config.get("server", "http://127.0.0.1:8080")

    # Load tasks
    try:
        tasks = load_tasks(args.tasks)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"Error loading tasks: {e}", file=sys.stderr)
        sys.exit(1)

    # Run benchmark
    print(f"Running benchmark on {server_url}...")
    print(f"Tasks: {len(tasks)}, Limit: {args.limit or 'all'}")

    try:
        results = run_benchmark(
            server_url=server_url,
            config=config,
            tasks=tasks,
            results_dir=args.results_dir,
            limit=args.limit,
        )
    except Exception as e:
        print(f"Benchmark failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate report
    report_path = Path(args.results_dir) / "results.md"
    try:
        generate_report(results, report_path)
    except Exception as e:
        print(f"Report generation failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Print summary
    scores = [r["score"] for r in results if r["score"] > 0]
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\nBenchmark complete.")
    print(f"  Tasks run: {len(results)}")
    print(f"  Avg score: {avg:.2f}")
    print(f"  Results: {args.results_dir}/results.json")
    print(f"  Report:  {report_path}")


if __name__ == "__main__":
    main()
