#!/usr/bin/env python3
"""CLI entrypoint — run the LLM benchmark against a llama.cpp server."""

import argparse
import sys
from pathlib import Path

from runner import run_benchmark

_PROJECT_ROOT = Path(__file__).resolve().parent


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

    results = run_benchmark(
        config_path=config_path,
        server_url=server_url,
        limit=args.limit,
    )

    print(f"Ran {len(results)} tasks.", file=sys.stderr)

    if args.report:
        from report import generate_report_from_file

        generate_report_from_file()
        print("Report written to results/results.md", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
