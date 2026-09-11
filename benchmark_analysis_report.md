# 📊 Report di Analisi Benchmark — AGENT BENCH

**Data:** Settembre 2026  
**Oggetto:** Valutazione critica della batteria di test attuale e proposte di miglioramento

---

## 1. Panoramica della Batteria Attuale

### Struttura attuale

| Categoria  | N. Task | Tipo di Valutazione                  |
| ---------- | ------- | ------------------------------------ |
| Code       | 8       | Keyword matching + structural checks |
| QA         | 4       | Keyword matching                     |
| Reasoning  | 3       | Keyword matching                     |
| **Totale** | **15**  | —                                    |

### Risultati medi osservati

| Run             | Overall | Code   | QA     | Reasoning |
| --------------- | ------- | ------ | ------ | --------- |
| 1 (Qwen3.6-35B) | 0.8419  | 0.8723 | 0.8750 | 0.7167    |
| 2 (Qwen3.8-27B) | 0.8530  | 0.8723 | 0.8542 | 0.8000    |

**Differenza tra modelli: ~1.1 punti percentuali** — esattamente il problema segnalato.

---

## 2. Analisi Critica: Perché i Punteggi Non Distinguono i Modelli

### 🔴 Debolezza #1: Keyword Matching come Metrica Primaria

Il sistema valuta le risposte basandosi sulla **presenza di parole chiave** nella risposta. Questo è problematico perché:

- **Un modello "bravino" può soddisfare tutte le keyword senza davvero risolvere il problema**
- **Un modello eccellente può formulare la risposta in modo diverso e perdere keyword**
- **La soglia di discriminazione è troppo bassa**: quasi tutti i modelli moderni (27B-35B+) riescono a includere le keyword attese

**Esempio reale dai tuoi dati:**

- Task 3 (palindrome): entrambi i modelli ottengono **0.8750** — identici
- Task 1 (linked list): entrambi ottengono **0.8889** — identici
- Task 4 (Stack): entrambi ottengono **1.0000** — perfetto per entrambi

Questi sono task "facili" dove quasi qualsiasi modello moderno passa il test delle keyword.

### 🔴 Debolezza #2: Assenza di Esecuzione Reale del Codice

I task di code generation **non eseguono il codice prodotto**. La valutazione è puramente testuale:

- Controlla se ci sono `def`, type hints, docstring, keyword
- **NON verifica se il codice funziona**

Nel mondo reale, la differenza tra un modello da 27B e uno da 70B si vede proprio nella **correttezza del codice generato**, non nella sua sintassi superficiale.

### 🔴 Debolezza #3: Task Troppo Semplici / Conoscenza di Base

I task attuali coprono algoritmi da libro di testo:

- Reverse linked list → O(1) space, 5 righe
- Binary search → algoritmo elementare
- Palindrome check → 3 righe
- Stack con get_min → esercizio da esame universitario

Questi sono task che **qualsiasi modello pre-addestrato su codice Python** (e tutti lo sono) risolve bene. Non c'è complessità reale.

### 🔴 Debolezza #4: Reasoning Superficiale

I 3 task di reasoning sono:

1. Sillogismo logico base ("all cats are animals")
2. Problema di treno a velocità diverse (algebra di scuola media)
3. Fallacia logica "affirming the consequent"

Questi sono **troppo facili** per distinguere modelli di fascia media-alta.

### 🔴 Debolezza #5: Dimensione del Campione Troppo Piccola

15 task totali → **varianza statistica alta**. Con così pochi task, le differenze reali tra modelli vengono "diluite" dal rumore.

---

## 3. Benchmark di Riferimento (Stato dell'Arte 2026)

Dalle ricerche web, ecco i benchmark più rilevanti:

### 🏆 Coding Benchmarks

| Benchmark              | Descrizione                                                                              | Perché è diverso dal tuo                                 |
| ---------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| **LiveCodeBench**      | Problemi competitivi da LeetCode/AtCoder, aggiornati in tempo reale, anti-contaminazione | Testa reasoning algoritmico reale, non esercizi da libro |
| **HumanEval-Pro**      | Estensione di HumanEval con auto-invoking code (funzioni che si chiamano tra loro)       | Valuta composizione, non funzioni isolate                |
| **SWE-bench Verified** | Risoluzione di issue reali su repo GitHub Python                                         | Testa ingegneria del software, non snippets              |
| **BigCodeBench**       | Problemi pratici con chiamate a librerie esterne                                         | Valuta knowledge delle API, non solo algoritmi           |
| **CRUXEval**           | Previsione input/output di funzioni Python                                               | Testa reasoning sul codice, non generazione              |

### 🧠 Reasoning Benchmarks

| Benchmark        | Descrizione                                                       | Perché è diverso                                  |
| ---------------- | ----------------------------------------------------------------- | ------------------------------------------------- |
| **GPQA Diamond** | Domande di PhD-level in fisica/chimica/biologia                   | Distingue modelli profondi da quelli superficiali |
| **MMLU-Pro**     | Estensione difficile di MMLU, 10 opzioni, ragionamento multi-step | Più discriminante del QA a risposta aperta        |
| **AIME 2025**    | Problemi di matematica olimpica (30 problemi)                     | Testa reasoning matematico avanzato               |
| **MATH-500**     | 500 problemi di matematica competitiva                            | Copertura multi-dominio                           |

### 📋 Other Important Benchmarks

| Benchmark                                  | Cosa misura                                             |
| ------------------------------------------ | ------------------------------------------------------- |
| **IFEval**                                 | Instruction following (formattazione, vincoli)          |
| **WildBench**                              | Task reali da utenti, 1024 task difficili               |
| **Artificial Analysis Intelligence Index** | Composite score: agents + coding + general + scientific |
| **Terminal-Bench**                         | Agentic workflows (uso di terminal, tool)               |

---

## 4. Confronto Diretto: Il Tuo Benchmark vs Standard di Settore

| Dimensione             | Il Tuo Benchmark           | Standard di Settore             |
| ---------------------- | -------------------------- | ------------------------------- |
| **Code correctness**   | Keyword matching testuale  | Esecuzione reale (pass@1)       |
| **Code complexity**    | Algoritmi da libro         | Problemi competitivi/reali      |
| **Reasoning depth**    | Logica elementare          | PhD-level / matematica olimpica |
| **Knowledge**          | QA a risposta aperta       | Multiple-choice strutturato     |
| **Sample size**        | 15 task                    | 100-1000+ task                  |
| **Contamination risk** | Altissimo (task statici)   | Controllato (LiveCodeBench)     |
| **Discriminazione**    | ~1% differenza tra modelli | 10-30% differenza tra modelli   |

---

## 5. Proposte di Miglioramento

### 🥇 Priorità Alta: Esecuzione Reale del Codice

**Aggiungere un runner di codice** che esegua effettivamente il codice generato e verifichi i test case:

```python
# Nuovo tipo di task: "code_execute"
{
    "id": 101,
    "type": "code_execute",
    "user_prompt": "Write a function that sorts a list using merge sort...",
    "test_cases": [
        {"input": [3,1,4,1,5,9,2,6], "expected": [1,1,2,3,4,5,6,9]},
        {"input": [], "expected": []},
        {"input": [1], "expected": [1]}
    ],
    "timeout": 10
}
```

**Impatto stimato:** Differenziazione tra modelli da 5-15%. Un modello che scrive codice syntatticamente corretto ma logicamente sbagliato viene penalizzato.

### 🥈 Priorità Alta: Task di Code Complessi

Sostituire gli esercizi da libro con problemi di livello competitivo:

```python
# Esempio LiveCodeBench-style
{
    "id": 102,
    "type": "code_execute",
    "user_prompt": "Given an array of integers nums and an integer target,
                   return indices of the two numbers such that they add up to target.
                   You may assume that each input would have exactly one solution.",
    "test_cases": [
        {"input": {"nums": [2,7,11,15], "target": 9}, "expected": [0,1]},
        {"input": {"nums": [3,2,4], "target": 6}, "expected": [1,2]}
    ],
    "difficulty": "medium"
}
```

### 🥉 Priorità Media: Reasoning Avanzato

Inserire task di livello GPQA/AIME:

```python
# Matematica avanzata
{
    "id": 201,
    "type": "reasoning_math",
    "user_prompt": "Let f(x) = x³ - 3x + 1. Find the number of real roots
                   of f(f(f(x))) = 0.",
    "expected_answer": 27,
    "verification": "numeric"
}

# Scienza PhD-level
{
    "id": 202,
    "type": "reasoning_science",
    "user_prompt": "In a hydrogen atom, what is the ratio of the de Broglie
                   wavelength of an electron in the n=2 state to that in
                   the n=1 state?",
    "expected_answer": "2",
    "verification": "numeric"
}
```

### 🏅 Priorità Media: Instruction Following

Testare la capacità del modello di seguire vincoli specifici:

```python
{
    "id": 301,
    "type": "instruction_following",
    "user_prompt": "Write a Python function to calculate factorial.
                   Constraints: (1) Use exactly 3 lines of code,
                   (2) Include no comments, (3) Use recursion,
                   (4) No imports allowed.",
    "constraints": [
        {"type": "line_count", "max": 3},
        {"type": "no_comments": True},
        {"type": "uses_recursion": True},
        {"type": "no_imports": True}
    ]
}
```

### 📊 Priorità Bassa: Aumentare il Numero di Task

Portare da 15 a almeno **50-100 task** per:

- Ridurre la varianza statistica
- Coprire più domini
- Permettere analisi per difficoltà (easy/medium/hard)

---

## 6. Implementazione Pratica: Roadmap

### Fase 1 (1-2 giorni): Runner di Codice

- Aggiungere `code_execute` type
- Sandbox sicura per esecuzione (timeout, memory limit)
- Test case verificabili

### Fase 2 (2-3 giorni): Nuovo Set di Task

- 20 task code_execute (difficoltà media-alta)
- 10 task reasoning_math/science
- 10 task instruction_following
- 5 task code_refactoring (migliorare codice esistente)

### Fase 3 (1 giorno): Evaluator Migliorato

- Supporto per `numeric` verification (approssimazione)
- Supporto per test case multipli
- Punteggio weighted per difficoltà

### Fase 4 (continuo): Aggiornamento Periodico

- Integrazione con LiveCodeBench API per task sempre nuovi
- Anti-contaminazione: task che cambiano periodicamente

---

## 7. Metriche Attese Dopo i Miglioramenti

| Scenario             | Differenza Attuale | Differenza Stimata |
| -------------------- | ------------------ | ------------------ |
| 27B vs 35B           | ~1%                | **5-10%**          |
| 27B vs 70B           | ~1%                | **10-20%**         |
| Q8_0 vs IQ2_XS       | ~1%                | **15-25%**         |
| temp=0.3 vs temp=1.0 | ~1%                | **5-15%**          |

---

## 8. Conclusioni

### Punti di Forza del Progetto

- ✅ Architettura pulita e ben strutturata
- ✅ Supporto per configurazione flessibile
- ✅ Report markdown chiaro
- ✅ Sistema di scoring estensibile
- ✅ Supporto per parallel/speculative decoding

### Problematiche Principali

- ❌ **Metrica di scoring troppo debole** (keyword matching)
- ❌ **Nessuna esecuzione reale del codice**
- ❌ **Task troppo facili** per distinguere modelli di fascia media-alta
- ❌ **Sample size insufficiente** (15 task)
- ❌ **Alto rischio di contamination** (task statici)

### Raccomandazione Finale

Il tuo framework è **solido nell'infrastruttura** ma **debole nella valutazione**. La differenza tra un benchmark utile e uno inutile non è nel runner o nel client HTTP, ma nel **design dei task e nel sistema di scoring**.

Con le modifiche proposte (soprattutto esecuzione reale del codice + task più complessi), il benchmark diventerebbe uno strumento **realmente discriminante** per confrontare modelli e configurazioni di inferenza.
