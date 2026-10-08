# Local LLM Benchmark

CLI benchmark for testing LLMs served via OpenAI-compatible APIs (e.g., llama.cpp, vLLM, Ollama).

## Purpose

Compare performance and response quality across different local LLMs and inference configurations (quantization, temperature, speculative decoding) served via OpenAI-compatible APIs. Each run produces a report with per-type scores, model rankings, and difficulty breakdowns.

## Requirements

- Python 3.10+
- httpx (managed by `pyproject.toml`)
- An LLM server exposing an OpenAI-compatible API (`/v1/chat/completions`)

## Installation

```bash
pip install -e .
```

## Configuration

The tool **never starts the server**: you start it manually (with any engine), then configure the API connection in `configs/models.json`. That is the **only** file whose values reach the server API — one active configuration at a time, edited as needed.

```json
{
  "baseUrl": "http://127.0.0.1:8080/v1",
  "apiKey": "dummy",
  "api": "openai-completions",
  "compat": { "supportsReasoningEffort": true },
  "thinking_level": "medium",
  "reasoning_budget": 8192,
  "models": [
    {
      "id": "qwen3.8-flash-next-coder-iq1_m",
      "name": "Qwen3.8-Flash-Next Coder",
      "reasoning": true,
      "input": ["text", "image"],
      "contextWindow": 163840,
      "maxTokens": 16384,
      "thinkingLevelMap": {
        "off": "none", "minimal": "low", "low": "low",
        "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"
      }
    }
  ],
  "launch_config": "configs/run.json"
}
```

- `baseUrl` — server URL **and port** (no hardcoded default anymore; accept `http://host:port` or `http://host:port/v1`)
- `thinking_level` — resolved per request through the model's `thinkingLevelMap` into `reasoning_effort` (only if `compat.supportsReasoningEffort`)
- `reasoning_budget` — sent as-is in the request
- `maxTokens` — worst-case generation cap, used for the adaptive deadline
- `launch_config` — path to the **launch recipe** file, see below

### Launch recipes (`configs/*.json`)

A recipe documents **how you started the server** (engine flags for llama.cpp, strata, etc.). The benchmark never interprets it: it copies its content verbatim into each run's history stamp, so every result in `results.md` is traceable to the exact flags that were in place. Write one recipe per model/engine configuration and keep them in `configs/`.

### Guardrails (`configs/timeout.json`)

Unchanged: `idle_timeout_s` (stream watchdog), `max_chars` (runaway output cap), and the adaptive deadline parameters (the deadline is derived from `maxTokens` and the observed token rate).

### Supported Inference Engines

This benchmark works with **any server implementing the OpenAI-compatible API** (`POST /v1/chat/completions`). Below is a curated list of the most popular and actively maintained inference engines that expose this endpoint.

#### Desktop / Local-first

| Engine | GitHub Stars | Notes |
|--------|-------------|-------|
| [**Ollama**](https://github.com/ollama/ollama) | 166k+ | Go, one-command setup, 100+ models via `ollama pull`. Default on `localhost:11434`. |
| [**LM Studio**](https://lmstudio.ai) | — | Polished GUI, ships both llama.cpp and MLX engines. Server mode on `localhost:1234`. |
| [**Jan**](https://github.com/janhq/jan) | 40k+ | Cross-platform desktop app with model library and built-in API server. |
| [**GPT4All**](https://github.com/nomic-ai/gpt4all) | 20k+ | Beginner-friendly desktop app with local server mode. |

#### CLI / Server-grade

| Engine | GitHub Stars | Notes |
|--------|-------------|-------|
| [**llama.cpp**](https://github.com/ggerganov/llama.cpp) (`llama-server`) | 70k+ | Foundational C/C++ engine. GGUF native. Full sampling param support (`top_k`, `mirostat`, `typical_p`, `tfs_z`, etc.). |
| [**vLLM**](https://github.com/vllm-project/vllm) | 30k+ | Production-grade, PagedAttention, highest throughput. Best for serving large models on GPUs. |
| [**SGLang**](https://github.com/sgl-project/sglang) | 10k+ | Rising star. Outperforms vLLM on chat workloads and structured output. OpenAI-compatible API. |
| [**text-generation-webui**](https://github.com/oobabooga/text-generation-webui) (oobabooga) | 40k+ | Flexible multi-backend (Transformers, llama.cpp, ExLlamaV2, etc.). Rich extensions ecosystem. |
| [**LocalAI**](https://github.com/mudler/LocalAI) | 20k+ | Drop-in OpenAI API replacement. Multi-backend (llama.cpp, vLLM, transformers, MLX). Supports text, images, audio. |
| [**mistral.rs**](https://github.com/EricLBuehler/mistral.rs) | 5k+ | Rust-based, blazing fast. OpenAI-compatible with core sampling params. |
| [**Xinference**](https://github.com/xorbitsai/inference) | 8k+ | Multi-engine backend with REST API. Supports llama.cpp, vLLM, Transformers, and more. |
| [**Unsloth**](https://github.com/unslothai/unsloth) | 30k+ | Dynamic GGUF quantization, fast inference/training. OpenAI-compatible server. |
| [**TGI**](https://github.com/huggingface/text-generation-inference) | 9k+ | Hugging Face's production server. Supports vLLM-like scheduling. Some params differ from OpenAI spec. |

#### Proxy / Aggregation Layer

| Engine | Notes |
|--------|-------|
| [**LiteLLM**](https://github.com/BerriAI/litellm) | Proxy unifying 100+ providers. Can front any of the above engines and expose a unified OpenAI-compatible API. |
| [**Open WebUI**](https://github.com/open-webui/open-webui) | Chat interface (ChatGPT-like) that works with any of the engines above. |

> **Tip:** Pick the engine that matches your hardware and workflow. For quick local testing: **Ollama** or **LM Studio**. For maximum parameter control: **llama.cpp**. For GPU throughput: **vLLM** or **SGLang**. All work with this benchmark — just set `baseUrl` in `configs/models.json` to the right URL.

### What the client sends per request

The benchmark does **not** send sampling parameters — sampling belongs to the server, which you start with your own flags (recorded in the launch recipe). Per request the client sends only:

`messages` (system + user), `stream: true`, `model` (id from `models.json`), and — when declared — `reasoning_effort` (resolved via `thinkingLevelMap`) and `reasoning_budget`.

### CLI usage

```
python run.py [--config configs/models.json] [--limit N] [--report]
```

| Flag | Meaning |
|------|---------|
| *(no flags)* | Run all tasks against the server declared in `configs/models.json` |
| `--config <path>` | Use a different API config file (default: `configs/models.json`) |
| `--limit <N>` | Run only the first N tasks (debugging) |
| `--report` | Also generate `results/results.md` (rankings + per-run provenance) |

> There is no `--server` flag anymore: the API URL, port, model and reasoning level all live in `configs/models.json`, one file to edit per configuration you test.

### Basic run

```bash
python run.py
```

### With markdown report

```bash
python run.py --report
```

### Limited task set (debug)

```bash
python run.py --limit 3
```

### Custom config

```bash
python run.py --config configs/models-gpu.json --report
```

## Output

- **stderr**: banner, progress `[N/29] type:id → Xs, Y chars, score:Z`, summary
- **logs/client.log**: structured request/response logging (zero console output)
- **results/results.json**: all results, cumulative across runs
- **results/results.md**: markdown report with rankings (with `--report`)

## Tasks

29 tasks in `tasks/tasks.json` organized into 8 categories:

| Type                      | Count | Description                                    |
| ------------------------- | ----- | ---------------------------------------------- |
| **code_execute**          | 6     | Code generation with verifiable test cases     |
| **reasoning_math**        | 4     | Advanced mathematics with numeric verification |
| **reasoning_science**     | 4     | Scientific explanations with depth evaluation  |
| **instruction_following** | 4     | Specific constraints (formatting, rules)       |
| **code_refactoring**      | 2     | Improving existing code with explanations      |
| **code_debug**            | 2     | Bug identification and correction              |
| **reasoning_logic**       | 3     | Complex logic and puzzles                      |
| **qa_knowledge**          | 4     | Python technical knowledge                     |

Each task has a difficulty level: `easy`, `medium`, or `hard`.

## Scoring

Scoring is multi-dimensional and varies by task type:

| Type                      | Score Composition                                                      |
| ------------------------- | ---------------------------------------------------------------------- |
| **code_execute**          | code structure (50%) + keywords (30%) + algorithmic patterns (20%)     |
| **instruction_following** | structure (30%) + keywords (30%) + constraint compliance (40%)         |
| **reasoning_math**        | keywords (50%) + numeric verification (50%)                            |
| **reasoning_science**     | keywords (40%) + reasoning indicators (30%) + explanation length (30%) |
| **reasoning_logic**       | keywords (50%) + step-by-step indicators (50%)                         |
| **code_refactoring**      | structure (40%) + keywords (40%) + explanation quality (20%)           |
| **code_debug**            | structure (30%) + keywords (40%) + bug identification (30%)            |
| **qa_knowledge**          | keyword matching (100%)                                                |

## Benchmark Standard Alignment

Tasks are categorized according to common types used in industry benchmarks. Here is the mapping:

| Benchmark Category                | Reference Standard       | Notes                                                                                                                               |
| --------------------------------- | ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| **code_execute**                  | LiveCodeBench, HumanEval | Code generation with structural and keyword checks. **Note:** current scoring is text-based (keyword matching), not code execution. |
| **reasoning_math**                | AIME, MATH-500           | Math problems with exact or range-based numeric verification.                                                                       |
| **reasoning_science**             | GPQA Diamond             | Scientific explanations scored by keywords, reasoning indicators, and depth.                                                        |
| **instruction_following**         | IFEval                   | Formatting and constraint compliance verification.                                                                                  |
| **code_refactoring / code_debug** | SWE-bench                | Code improvement and debugging tasks.                                                                                               |
| **reasoning_logic**               | Formal logic / puzzles   | Step-by-step reasoning with structured indicators.                                                                                  |
| **qa_knowledge**                  | MMLU-Pro                 | Open-ended technical knowledge questions.                                                                                           |

> **Caveat:** This benchmark is designed for relative comparisons between configurations of the same model or between models of similar capability in a local environment. The keyword-based scoring system does not replace professional benchmarks with real code execution and automated verification. For rigorous comparison, integrate a code executor (e.g., Docker sandbox) for `code_execute` tasks.

## Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
run.py              # CLI entrypoint
runner.py           # Benchmark execution engine
llm_client.py       # Async HTTP client for OpenAI-compatible servers
evaluator.py        # Multi-dimensional scoring by task type
report.py           # Markdown report generation
configs/models.json # API connection + model capabilities + per-run choices (the only config sent to the server)
configs/run.json    # Launch recipe: how the server was started (stamped verbatim, never interpreted)
configs/timeout.json # Request timeouts and guardrails
tasks/tasks.json    # 29 tasks in 8 categories
tests/              # Unit and integration tests
```
