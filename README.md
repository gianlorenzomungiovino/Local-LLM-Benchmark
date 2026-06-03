# LLM Benchmark

Benchmark CLI per testare modelli LLM esposti tramite il server [llama.cpp](https://github.com/ggerganov/llama.cpp) con API OpenAI-compatible.

## Requisiti

- Python 3.12+
- httpx (gestito da `pyproject.toml`)

## Installazione

```bash
pip install -e .
```

## Avviare il server llama.cpp

```bash
llama-server -m modello.gguf --host 127.0.0.1 --port 8080   --temp 0.3 --top-k 30 --top-p 0.95 --repeat-penalty 1.0
```

## Configurazione

Modifica `configs/run.json` con i parametri desiderati:

```json
{
  "model": "nome-modello",
  "temperature": 0.3,
  "top_k": 30,
  "top_p": 0.95,
  "min_p": 0.0,
  "repeat_penalty": 1.0,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "mirostat": 0,
  "typical_p": 1.0,
  "seed": 42
}
```

## Comandi

### Esecuzione base

```bash
python run.py --server http://localhost:8080
```

### Con report markdown

```bash
python run.py --server http://localhost:8080 --report
```

### Limita i task (debug)

```bash
python run.py --server http://localhost:8080 --limit 3
```

### Config personalizzato

```bash
python run.py --config configs/run.json --server http://localhost:8080 --report
```

## Output

- **stderr**: banner, progresso `[N/15] type:id → Xs, Y chars, score:Z`, summary
- **logs/client.log**: log strutturato delle richieste/risposte
- **results/results.json**: tutti i risultati in JSON
- **results/results.md**: report markdown (con `--report`)

## Task

15 task in `tasks/tasks.json`:
- **code** (8): scrittura di codice Python con valutazione strutturale
- **qa** (4): domande di conoscenza con keyword matching
- **reasoning** (3): logica e ragionamento con keyword matching

## Test

```bash
python -m pytest tests/ -v
```

## Struttura

```
run.py          # CLI entrypoint
runner.py       # Motore di esecuzione benchmark
llm_client.py   # Client HTTP async per llama.cpp
evaluator.py    # Scoring (strutturale + keyword matching)
report.py       # Generazione report markdown
configs/run.json# Configurazione server
tasks/tasks.json# Task del benchmark
tests/          # Test unitari e integrazione
```
