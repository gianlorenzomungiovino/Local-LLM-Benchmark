# Agent Bench

Minimal Python CLI benchmarking tool for locally running LLM servers (llama.cpp).

## Setup

```bash
cd agent-bench
pip install -r requirements.txt
```

## Usage

1. Start llama.cpp server manually:
   ```bash
   ./server -m model.gguf --temp 0.3 --top-k 30 --top-p 0.95 ...
   ```

2. Edit `configs/run.json` with your server parameters:
   ```json
   {
     "run_id": "run-20260601-0001",
     "model": "qwen3.6-35b",
     "server": "http://127.0.0.1:8080",
     "temperature": 0.3,
     "top_k": 30,
     "top_p": 0.95,
     "min_p": 0.0,
     "repeat_penalty": 1.0,
     "presence_penalty": 0.0
   }
   ```

3. Run the benchmark:
   ```bash
   python run.py
   ```

4. View results:
   - `results/results.json` — structured results
   - `results/results.md` — readable report with averages and ranking

## Options

```bash
python run.py --server http://127.0.0.1:8080
python run.py --limit 5          # debug: run only 5 tasks
python run.py --config configs/my-run.json
```

## Task Types

- **code** — Code generation, evaluated with AST validation + keyword match + quality metrics
- **qa** — Question answering, evaluated with substring match + keyword overlap
- **reasoning** — Code reasoning/debugging, evaluated with keyword overlap + logical check

## Structure

```
agent-bench/
  run.py              # CLI entrypoint
  llm_client.py       # HTTP client (httpx)
  evaluator.py        # Deterministic scoring
  runner.py           # Task execution + report generation
  configs/
    run.json          # Server parameters (edit before each run)
  tasks/
    tasks.json        # 15 benchmark tasks
  results/
    results.json      # Accumulated results
    results.md        # Markdown report
  tests/              # Unit tests
  requirements.txt    # Dependencies
```
