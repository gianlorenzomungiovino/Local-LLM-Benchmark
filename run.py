#!/usr/bin/env python3
"""CLI entrypoint — run the LLM benchmark against a llama.cpp server."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from runner import run_benchmark

_PROJECT_ROOT = Path(__file__).resolve().parent


def detect_and_populate_config(
    config_path: str, server_url: str,
) -> tuple[dict, dict | None]:
    """Auto-detect model and server metadata from the llama.cpp server.

    If config['model'] is already set (not None), returns config unchanged.
    Otherwise queries /v1/models, updates config with model name and n_ctx,
    and writes the updated config back to disk.

    Args:
        config_path: Path to the JSON config file.
        server_url: llama.cpp server URL, e.g. 'http://localhost:8080'.

    Returns:
        Tuple of (config dict, detected metadata dict or None).
        Metadata includes: id, n_ctx, n_ctx_train, n_embd, n_params, n_vocab, size.
    """
    # Load config
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = _PROJECT_ROOT / config_file
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Skip detection if model is already set
    if config.get("model") is not None:
        return config, None

    # Query server for model info
    try:
        from llm_client import LLMClient

        client = LLMClient(server_url)
        model_info = asyncio.run(client.fetch_model_info())
        if model_info:
            config["model"] = model_info["id"]
            # Auto-set n_ctx from server if not already set
            if "n_ctx" not in config and model_info.get("n_ctx"):
                config["n_ctx"] = model_info["n_ctx"]
            # Write updated config back to disk
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            return config, model_info
        return config, None
    except Exception as exc:
        # Detection failed — return config unchanged
        print(
            f"  [warn] Model detection failed: {exc}",
            file=sys.stderr,
        )
        return config, None


def _print_banner(
    task_count: int,
    config: dict,
    server_url: str,
    limit: int | None,
    detected_model: str | None = None,
) -> None:
    """Print a startup banner with config and task info to stderr.

    Args:
        task_count: Number of tasks to run.
        config: Configuration dict.
        server_url: llama.cpp server URL.
        limit: Optional task limit.
        detected_model: Auto-detected model name (optional).
    """
    sep = "=" * 55
    print("\n" + sep, file=sys.stderr)
    print("  LLM Benchmark — Starting", file=sys.stderr)
    print(sep, file=sys.stderr)
    if detected_model:
        print(f"  Model: {detected_model}", file=sys.stderr)
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

    # Load config, auto-detect model if not set
    config, model_info = detect_and_populate_config(config_path, server_url)

    # Load tasks to count them
    tasks_path = _PROJECT_ROOT / "tasks" / "tasks.json"
    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    task_count = len(tasks)
    if args.limit is not None:
        task_count = min(args.limit, len(tasks))

    # Print startup banner (with detected model if available)
    _print_banner(task_count, config, server_url, args.limit, model_info)

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
