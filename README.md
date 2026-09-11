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
llama-server -m modello.gguf --host 127.0.0.1 --port 8080 --temp 0.3 --top-k 30 --top-p 0.95 --repeat-penalty 1.0
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

- **stderr**: banner, progresso `[N/39] type:id → Xs, Y chars, score:Z`, summary
- **logs/client.log**: log strutturato delle richieste/risposte
- **results/results.json**: tutti i risultati in JSON
- **results/results.md**: report markdown (con `--report`)

## Task

39 task in `tasks/tasks.json` organizzati per tipo e difficoltà:

### Tipologie di Task

| Tipo | N. | Descrizione |
|------|-----|-------------|
| **code_execute** | 8 | Generazione codice con test case verificabili |
| **reasoning_math** | 5 | Matematica avanzata (AIME-level) con verifica numerica |
| **reasoning_science** | 5 | Scienza PhD-level (GPQA-level) con spiegazioni approfondite |
| **instruction_following** | 5 | Vincoli specifici (formattazione, constraints) |
| **code_refactoring** | 3 | Miglioramento codice esistente con spiegazioni |
| **code_debug** | 3 | Identificazione e correzione di bug |
| **reasoning_logic** | 5 | Logica complessa e puzzle |
| **qa_knowledge** | 5 | Conoscenze Python avanzate |

### Difficoltà

Ogni task ha un livello di difficoltà: `easy`, `medium`, o `hard`. Il report mostra i punteggi medi per difficoltà.

## Scoring

Il sistema di scoring è multi-dimensionale:

- **code_execute**: struttura codice (50%) + keyword (30%) + pattern algoritmici (20%)
- **instruction_following**: struttura (30%) + keyword (30%) + compliance vincoli (40%)
- **reasoning_math**: keyword (50%) + verifica numerica (50%)
- **reasoning_science**: keyword (40%) + indicatori ragionamento (30%) + lunghezza spiegazione (30%)
- **reasoning_logic**: keyword (50%) + indicatori step-by-step (50%)

## Benchmark di Riferimento

Il benchmark è progettato per allinearsi con gli standard di settore:

- **LiveCodeBench** → code_execute (difficoltà competitiva)
- **GPQA Diamond** → reasoning_science (livello PhD)
- **AIME** → reasoning_math (matematica olimpica)
- **IFEval** → instruction_following (vincoli verificabili)
- **SWE-bench** → code_debug/refactoring (ingegneria software)

## Test

```bash
python -m pytest tests/ -v
```

## Struttura

```
run.py              # CLI entrypoint
runner.py           # Motore di esecuzione benchmark
llm_client.py       # Client HTTP async per llama.cpp
evaluator.py        # Scoring multi-dimensionale per task type
report.py           # Generazione report markdown
configs/run.json    # Configurazione server
tasks/tasks.json    # 39 task in 8 categorie
benchmark_analysis_report.md  # Analisi critica del benchmark
tests/              # Test unitari e integrazione
```
