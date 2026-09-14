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

Edit `configs/run.json` with your server parameters. The JSON below is an **example** — the actual parameters depend on your model and inference engine.

```json
{
  "model": "your-model-name",
  "temperature": 0.3,
  "top_k": 20,
  "top_p": 0.95,
  "min_p": 0.0,
  "repeat_penalty": 1.0,
  "presence_penalty": 0.0,
  "threads": 6,
  "ctx_size": 57344,
  "ngl": "all",
  "flash_attn": "on"
}
```

The `model` field is optional: if omitted or `null`, the benchmark auto-detects it by querying `/v1/models` on the first run.

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

> **Tip:** Pick the engine that matches your hardware and workflow. For quick local testing: **Ollama** or **LM Studio**. For maximum parameter control: **llama.cpp**. For GPU throughput: **vLLM** or **SGLang**. All work with this benchmark — just point `--server` to the right URL.

### Supported API Parameters

The client maps the following config keys to the OpenAI-compatible API:

`temperature`, `top_k`, `top_p`, `min_p`, `repeat_penalty`, `presence_penalty`, `frequency_penalty`, `mirostat`, `mirostat_tau`, `mirostat_eta`, `typical_p`, `penalty_last_n`, `tfs_z`, `num_keep`, `seed`, `n_predict`, `logit_bias`

Parameters not recognized by your specific server will be ignored by the server (not the client).

## Commands

### Basic run

```bash
python run.py --server http://localhost:8080
```

### With markdown report

```bash
python run.py --server http://localhost:8080 --report
```

### Limited task set (debug)

```bash
python run.py --server http://localhost:8080 --limit 3
```

### Custom config

```bash
python run.py --config configs/run.json --server http://localhost:8080 --report
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
configs/run.json    # Server and inference parameters
configs/timeout.json # Request timeouts and guardrails
tasks/tasks.json    # 29 tasks in 8 categories
benchmark_analysis_report.md  # Critical analysis of the benchmark
tests/              # Unit and integration tests
```
