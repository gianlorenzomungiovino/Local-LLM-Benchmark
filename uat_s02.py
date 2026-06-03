#!/usr/bin/env python3
"""UAT script for S02: Auto-detect modello e populate config defaults."""
import json, sys, tempfile, os, io
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from run import detect_and_populate_config, _print_banner

results = {}

# ============================================================
# UAT-01: Auto-detection populates config when model is null
# ============================================================
print("=== UAT-01: Auto-detection populates config when model is null ===")
config = {"model": None, "temperature": 0.7, "top_k": 40, "top_p": 0.95, "min_p": 0.0, "repeat_penalty": 1.1, "presence_penalty": 0.0}
tmpdir = Path(tempfile.mkdtemp())
config_path = tmpdir / "run.json"
config_path.write_text(json.dumps(config, indent=4), encoding="utf-8")

mock_client = MagicMock()
mock_client.fetch_model_info = AsyncMock(return_value="llama-3-8b-instruct")

with patch("llm_client.LLMClient", return_value=mock_client):
    result_config, detected_model = detect_and_populate_config(str(config_path), "http://localhost:8080")

old_stderr = sys.stderr
sys.stderr = io.StringIO()
try:
    _print_banner(10, result_config, "http://localhost:8080", None, detected_model)
    banner_output = sys.stderr.getvalue()
finally:
    sys.stderr = old_stderr

u01_checks = [
    ("Config model updated in return value", result_config["model"] == "llama-3-8b-instruct"),
    ("Detected model returned", detected_model == "llama-3-8b-instruct"),
    ("Config file on disk updated", json.loads(config_path.read_text(encoding="utf-8"))["model"] == "llama-3-8b-instruct"),
    ("Banner shows 'Model: llama-3-8b-instruct'", "Model: llama-3-8b-instruct" in banner_output),
    ("Config file uses indent=4", '    "model"' in config_path.read_text(encoding="utf-8")),
]
u01_pass = all(c[1] for c in u01_checks)
for desc, passed in u01_checks:
    print(f"  [{'PASS' if passed else 'FAIL'}] {desc}")
results["UAT-01"] = {"pass": u01_pass, "checks": u01_checks}

# ============================================================
# UAT-02: No detection when model is already set
# ============================================================
print("\n=== UAT-02: No detection when model is already set ===")
config2 = {"model": "llama-3-70b", "temperature": 0.5, "top_k": 20}
config_path2 = tmpdir / "run2.json"
config_path2.write_text(json.dumps(config2, indent=4), encoding="utf-8")

mock_client2 = MagicMock()
mock_client2.fetch_model_info = AsyncMock(return_value="should-not-be-called")

with patch("llm_client.LLMClient", return_value=mock_client2) as MockLLMClient:
    result_config2, detected2 = detect_and_populate_config(str(config_path2), "http://localhost:8080")

old_stderr = sys.stderr
sys.stderr = io.StringIO()
try:
    _print_banner(5, result_config2, "http://localhost:8080", 3, detected2)
    banner_output2 = sys.stderr.getvalue()
finally:
    sys.stderr = old_stderr

u02_checks = [
    ("LLMClient was NOT instantiated", not MockLLMClient.called),
    ("Config model unchanged", result_config2["model"] == "llama-3-70b"),
    ("Detected model is None", detected2 is None),
    ("Config file on disk unchanged", json.loads(config_path2.read_text(encoding="utf-8"))["model"] == "llama-3-70b"),
    ("Banner does NOT show 'Model:' line", "Model:" not in banner_output2),
]
u02_pass = all(c[1] for c in u02_checks)
for desc, passed in u02_checks:
    print(f"  [{'PASS' if passed else 'FAIL'}] {desc}")
results["UAT-02"] = {"pass": u02_pass, "checks": u02_checks}

# ============================================================
# UAT-03: Graceful degradation when server is unavailable
# ============================================================
print("\n=== UAT-03: Graceful degradation when server is unavailable ===")
config3 = {"model": None, "temperature": 0.7}
config_path3 = tmpdir / "run3.json"
config_path3.write_text(json.dumps(config3, indent=4), encoding="utf-8")

mock_client3 = MagicMock()
mock_client3.fetch_model_info = AsyncMock(side_effect=Exception("Connection refused"))

with patch("llm_client.LLMClient", return_value=mock_client3):
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        result_config3, detected3 = detect_and_populate_config(str(config_path3), "http://localhost:8080")
    finally:
        stderr_output = sys.stderr.getvalue()
        sys.stderr = old_stderr

u03_checks = [
    ("Config model unchanged (None)", result_config3["model"] is None),
    ("Detected model is None", detected3 is None),
    ("Config file on disk unchanged", json.loads(config_path3.read_text(encoding="utf-8"))["model"] is None),
    ("Warning logged to stderr", "[warn]" in stderr_output or "Model detection failed" in stderr_output),
]
u03_pass = all(c[1] for c in u03_checks)
for desc, passed in u03_checks:
    print(f"  [{'PASS' if passed else 'FAIL'}] {desc}")
results["UAT-03"] = {"pass": u03_pass, "checks": u03_checks}

# ============================================================
# UAT-04: Banner displays detected model name
# ============================================================
print("\n=== UAT-04: Banner displays detected model name ===")
old_stderr = sys.stderr
sys.stderr = io.StringIO()
try:
    _print_banner(10, {"model": "llama-3-8b-instruct"}, "http://localhost:8080", None, "llama-3-8b-instruct")
    banner4 = sys.stderr.getvalue()
finally:
    sys.stderr = old_stderr

u04_checks = [
    ("Banner includes 'Model: llama-3-8b-instruct'", "Model: llama-3-8b-instruct" in banner4),
    ("Model line appears between separator and Tasks line", banner4.index("Model:") < banner4.index("Tasks:")),
]
u04_pass = all(c[1] for c in u04_checks)
for desc, passed in u04_checks:
    print(f"  [{'PASS' if passed else 'FAIL'}] {desc}")
results["UAT-04"] = {"pass": u04_pass, "checks": u04_checks}

# ============================================================
# UAT-05: Config file updated with indent=4
# ============================================================
print("\n=== UAT-05: Config file updated with indent=4 ===")
config5 = {"model": None, "temperature": 0.7, "top_k": 40, "top_p": 0.95}
config_path5 = tmpdir / "run5.json"
config_path5.write_text(json.dumps(config5, indent=4), encoding="utf-8")

mock_client5 = MagicMock()
mock_client5.fetch_model_info = AsyncMock(return_value="llama-3-8b-instruct")

with patch("llm_client.LLMClient", return_value=mock_client5):
    detect_and_populate_config(str(config_path5), "http://localhost:8080")

content5 = config_path5.read_text(encoding="utf-8")
parsed5 = json.loads(content5)
u05_checks = [
    ("Model field populated", parsed5["model"] == "llama-3-8b-instruct"),
    ("File formatted with 4-space indent", '    "model"' in content5),
    ("File is valid JSON", isinstance(parsed5, dict)),
]
u05_pass = all(c[1] for c in u05_checks)
for desc, passed in u05_checks:
    print(f"  [{'PASS' if passed else 'FAIL'}] {desc}")
results["UAT-05"] = {"pass": u05_pass, "checks": u05_checks}

# ============================================================
# Edge Cases
# ============================================================
print("\n=== Edge Cases ===")

# Edge: Server returns empty data array
config_edge = {"model": None}
config_path_edge = tmpdir / "run_edge.json"
config_path_edge.write_text(json.dumps(config_edge, indent=4), encoding="utf-8")

mock_client_edge = MagicMock()
mock_client_edge.fetch_model_info = AsyncMock(return_value=None)

with patch("llm_client.LLMClient", return_value=mock_client_edge):
    detect_and_populate_config(str(config_path_edge), "http://localhost:8080")

edge1 = json.loads(config_path_edge.read_text(encoding="utf-8"))["model"] is None
print(f"  [{'PASS' if edge1 else 'FAIL'}] Server returns empty data -> config unchanged")

# Edge: Server returns response without "data" key
# This is tested in test_llm_client.py::TestFetchModelInfo::test_fetch_model_info_missing_data_key_returns_none
# which already passed above

# Edge: Multiple models on server - only first is used
# This is tested in test_llm_client.py::TestFetchModelInfo::test_fetch_model_info_returns_first_model_id
# which already passed above

# Edge: Network timeout
config_edge2 = {"model": None}
config_path_edge2 = tmpdir / "run_edge2.json"
config_path_edge2.write_text(json.dumps(config_edge2, indent=4), encoding="utf-8")

mock_client_edge2 = MagicMock()
mock_client_edge2.fetch_model_info = AsyncMock(side_effect=Exception("Timeout"))

with patch("llm_client.LLMClient", return_value=mock_client_edge2):
    detect_and_populate_config(str(config_path_edge2), "http://localhost:8080")

edge2 = json.loads(config_path_edge2.read_text(encoding="utf-8"))["model"] is None
print(f"  [{'PASS' if edge2 else 'FAIL'}] Network timeout -> config unchanged")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("UAT SUMMARY")
print("=" * 60)
all_pass = True
for check_name, data in results.items():
    status = "PASS" if data["pass"] else "FAIL"
    if not data["pass"]:
        all_pass = False
    print(f"  {check_name}: {status}")
for edge_name, edge_pass in [("Empty data array", edge1), ("Network timeout", edge2)]:
    status = "PASS" if edge_pass else "FAIL"
    if not edge_pass:
        all_pass = False
    print(f"  Edge - {edge_name}: {status}")

print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
