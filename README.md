# LLM Evaluation & Prompt Optimization System

A scalable framework for evaluating, comparing, and optimizing Large Language Model (LLM) prompt variants.  
The system supports bulk prompt testing, automated metric computation, LLM-based judging, and interactive analytics for data-driven prompt engineering.

---

## Overview

This project enables systematic evaluation of prompt variants by sending them to a local LLM, computing quantitative metrics, and generating a composite score. Results are stored for comparison and visualized through interactive dashboards.

---

## Features

- Bulk CSV prompt import (200+ variants)
- Parallel batch evaluation
- LLM-as-judge scoring
- Composite scoring system
- SQLite experiment tracking
- Interactive analytics dashboards
- Batch comparison and leaderboard
- CSV export functionality
- Per-variant deep analysis
- Score distribution insights

---

## Why This Project

- Eliminates manual prompt testing  
- Supports large-scale prompt evaluation  
- Combines NLP metrics with LLM judging  
- Runs locally without API cost  
- Parallel execution for faster evaluation  
- Experiment tracking for reproducibility  
- Interactive visual analytics  
- Modular and extensible architecture  

---

## Architecture

User Prompts (CSV / Manual Input)
            │
            ▼
   ┌─────────────────────────┐
   │  Batch Evaluation Pipeline │
   │                         │
   │  ├─ LLM Response Gen    │  ← Ollama (llama3 / any local model)
   │  ├─ Metric Computation  │  ← BLEU, ROUGE-L, Semantic Similarity
   │  ├─ LLM Judge Scoring   │  ← Faithfulness, Clarity, Usefulness
   │  └─ Composite Score     │  ← Weighted aggregate [0–100]
   └─────────────────────────┘
            │
            ▼
       SQLite Storage
            │
            ▼
   Analytics Dashboard (Streamlit)
```

---

## Installation

```bash
pip install -r requirements.txt
```

Start local model:

```bash
ollama serve
```

Pull model:

```bash
ollama pull llama3
```

Run application:

```bash
streamlit run ui/app.py
```

---

## Usage

1. Upload prompt CSV  
2. Select model  
3. Configure concurrency  
4. Run evaluation  
5. Analyze results  
6. Compare batches  
7. Export results  

---

## CSV Format

Required column:

| Column | Description |
|--------|-------------|
| text | Prompt text |

Optional columns:

| Column | Description |
|--------|-------------|
| name | Variant label |
| tags | Comma-separated tags |

Example:

```csv
name,text,tags
Simple,"Explain AI simply.",basic
Detailed,"Explain AI with examples.",advanced
```

---

## Metrics

| Metric | Range | Description |
|--------|------|-------------|
| BLEU | 0–1 | N-gram overlap |
| ROUGE-L | 0–1 | Sequence similarity |
| Semantic Similarity | 0–1 | Embedding cosine similarity |
| Coherence | 0–1 | Response structure |
| Faithfulness | 0–1 | LLM judge accuracy |
| Completeness | 0–1 | Coverage |
| Clarity | 0–1 | Readability |
| Usefulness | 0–1 | Practical value |
| Composite Score | 0–100 | Weighted score |

---

## Composite Score Formula

```
Composite =
0.40 * Semantic Similarity +
0.15 * BLEU +
0.15 * ROUGE-L +
0.15 * Coherence +
0.15 * Faithfulness
```

---

## Concurrency

| Threads | Description |
|---------|-------------|
| 1 | Sequential execution |
| 2 | Balanced performance |
| 4 | Parallel execution |

---

## Project Structure

```
llm-eval-pro/
├── ui/
│   ├── app.py
│   └── pages/
├── src/
│   ├── evaluator/
│   ├── prompt_manager/
│   └── models/
├── sample_data/
├── data/
└── requirements.txt
```

---

## Tech Stack

- Python  
- Streamlit  
- Ollama  
- Sentence Transformers  
- Plotly  
- SQLite  
- Pandas  
- NumPy  
- Scikit-learn  

---

## Workflow

1. Import prompts  
2. Run evaluation  
3. Compute metrics  
4. Generate composite score  
5. Store results  
6. Visualize analytics  
7. Compare runs  

---

## Performance

- Supports 200+ prompt variants  
- Parallel execution improves speed  
- Local inference reduces latency  
- Bulk database writes  

---

## Future Improvements

- Automatic prompt optimization  
- Multi-model comparison  
- REST API support  
- Cloud deployment  
- Prompt versioning  
- Hyperparameter tuning  

---

## Author

Vaishnavi Deshpande
