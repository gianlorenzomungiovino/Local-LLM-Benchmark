#!/usr/bin/env python3
"""CLI entrypoint — run the LLM benchmark against a local OpenAI-compatible server."""

import argparse
import json
import sys
from pathlib import Path

from runner import run_benchmark
from runner import _build_run_stamp

_PROJECT_ROOT = Path(__file__).resolve().parent

# Results history location (module-level so tests can redirect it)
_RESULTS_DIR = _PROJECT_ROOT / "results"
_RESULTS_PATH = _RESULTS_DIR / "results.json"


def _print_banner(
    task_count: int,
    run_stamp: dict,
    config_path: str,
    limit: int | None,
) -> None:
    """Print a startup banner with config and task info to stderr.

    Args:
        task_count: Number of tasks to run.
        run_stamp: Resolved run stamp (recipe + identity + reasoning choices).
        config_path: Path to models.json.
        limit: Optional task limit.
    """
    sep = "=" * 55
    print("\n" + sep, file=sys.stderr)
    print("  LLM Benchmark — Starting", file=sys.stderr)
    print(sep, file=sys.stderr)
    if run_stamp.get("model"):
        print(f"  Model: {run_stamp['model']}", file=sys.stderr)
    if limit:
        print(f"  Tasks: {task_count} (limited to {limit})", file=sys.stderr)
    else:
        print(f"  Tasks: {task_count}", file=sys.stderr)
    print(f"  Server: {run_stamp.get('baseUrl')}", file=sys.stderr)
    print("  Config:", file=sys.stderr)
    for key, value in run_stamp.items():
        print(f"    {key}: {value}", file=sys.stderr)
    print(sep + "\n", file=sys.stderr)


def _warn_served_model_mismatch(api_config: dict) -> None:
    """Sanity warning only — the configured models.json id stays the truth.

    If the server answers GET /v1/models and none of the served ids matches
    the configured model, print a loud warning so a mislabeled run is caught
    immediately (this happens when models.json was not updated along with the
    launch recipe). Never re-configures anything.
    """
    try:
        import httpx
        base = api_config["baseUrl"].rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        resp = httpx.get(base + "/v1/models", timeout=5.0)
        served = {m.get("id") for m in resp.json().get("data", []) if m.get("id")}
        configured = (api_config.get("models") or [{}])[0].get("id")
        if served and configured not in served:
            print(
                f"  WARNING: server serves {sorted(served)}; configured model is "
                f"{configured!r} — update configs/models.json before trusting this run.",
                file=sys.stderr,
            )
    except Exception:
        pass  # server unreachable or endpoint missing: stay silent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LLM benchmark against a local OpenAI-compatible server.",
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
        default="configs/models.json",
        help="Path to models.json API config (default: configs/models.json)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        default=False,
        help="Record this run in results/results.json and generate results/results.md",
    )
    args = parser.parse_args()

    config_path = args.config

    # Resolve config path relative to project root if not absolute
    if not Path(config_path).is_absolute():
        config_path = str(_PROJECT_ROOT / config_path)

    # Load API config and build the run stamp (recipe verbatim + resolved
    # identity/reasoning choices). This is what gets recorded in original_configs
    # so each run in results.md is traceable to the flags the server was started
    # with. Nothing is written back to models.json.
    with open(config_path, "r", encoding="utf-8") as f:
        api_config = json.load(f)
    run_stamp = _build_run_stamp(api_config)

    # Record the run stamp in original_configs (history bookkeeping).
    # Only runs launched with --report are kept: runs without it are
    # throwaway tests and must not pollute the history.
    if args.report:
        existing = None
        if _RESULTS_PATH.exists():
            with open(_RESULTS_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
        if isinstance(existing, dict):
            existing_originals = existing.get("original_configs", [])
            existing_originals.append(run_stamp)
            existing["original_configs"] = existing_originals
        else:
            # Old format: convert to new format, preserving results
            old_results = existing if isinstance(existing, list) else []
            existing = {"results": old_results, "original_configs": [run_stamp]}
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    # Load tasks to count them
    tasks_path = _PROJECT_ROOT / "tasks" / "tasks.json"
    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    task_count = len(tasks)
    if args.limit is not None:
        task_count = min(args.limit, len(tasks))

    # Print startup banner
    _print_banner(task_count, run_stamp, config_path, args.limit)
    if not args.report:
        print("  NOTE: this run is NOT recorded in results/ (pass --report to keep it)", file=sys.stderr)

    # Warn (never re-configure) if the served model id disagrees with the config
    _warn_served_model_mismatch(api_config)

    results = run_benchmark(
        config_path=config_path,
        limit=args.limit,
        record=args.report,
    )

    print(f"\nCompleted {len(results)} tasks.", file=sys.stderr)

    if args.report:
        from report import generate_report_from_file

        generate_report_from_file()
        print("Report written to results/results.md", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
