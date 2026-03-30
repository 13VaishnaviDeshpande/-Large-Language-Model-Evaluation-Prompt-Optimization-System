# ⚡ LLM Eval Pro

A production-ready system for evaluating and comparing LLM prompt variants at scale —
supporting **200+ variants** via CSV import, per-variant deep analysis, and full analytics.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Ollama and pull a model
ollama serve
ollama pull llama3

# 3. Launch the app
streamlit run ui/app.py
```

---

## ✨ What's New (vs original)

| Feature | Original | Pro |
|---|---|---|
| Max variants | 5 (hard-coded) | **Unlimited** (200+ tested) |
| CSV import | ❌ | ✅ with preview & validation |
| Per-variant analysis | Tabs only | **Paginated expanders** with inline metrics + progress bars |
| Batch tracking | ❌ | ✅ SQLite batch runs table |
| Concurrent evaluation | ❌ | ✅ configurable thread pool |
| Bulk DB writes | One-by-one | ✅ single transaction |
| Export results | ❌ | ✅ CSV download per batch |
| Score insights | Basic | ✅ spread, discriminating metric, strengths |
| Metric visualization | Radar only | Radar + Box plot + Scatter + Heatmap |
| Batch Analysis page | ❌ | ✅ full page with all charts |

---

## 📂 CSV Import Format

Minimum required column:

| Column name(s) accepted | Maps to |
|---|---|
| `text`, `prompt`, `prompt_text`, `content` | Prompt text (**required**) |
| `name`, `prompt_name`, `variant_name` | Display label |
| `tags`, `category` | Tags (comma-separated in cell) |

**Example:**
```csv
name,text,tags
Simple,"Explain X simply.",beginner
Expert,"You are an expert. Explain X with examples.",expert
```

A sample CSV is included at `sample_data/sample_prompts.csv`.

---

## 🗺️ Pages

| Page | Description |
|---|---|
| 🏠 Home (`app.py`) | Stats overview, navigation guide, quick start |
| 🔬 Prompt Lab | Manual + CSV input, run evaluation, per-variant analysis |
| 🧪 Batch Analysis | Deep dive into any batch: distribution, heatmap, scatter, all variants |
| ⚖️ Compare | Side-by-side two runs with diff |
| 🏆 Leaderboard | Top all-time runs |
| 📊 Analytics | Trends, correlations, model comparison |

---

## 🏗️ Architecture

```
llm-eval-pro/
├── ui/
│   ├── app.py                    # Home dashboard
│   └── pages/
│       ├── 01_prompt_lab.py      # Main eval + CSV import
│       ├── 02_batch_analysis.py  # Per-batch deep analysis ← NEW
│       ├── 03_compare.py         # Side-by-side comparison
│       ├── 04_leaderboard.py     # Rankings
│       └── 05_analytics.py       # Trends & correlations
├── src/
│   ├── evaluator/
│   │   ├── pipeline.py           # Batch eval + variant report
│   │   ├── metrics.py            # BLEU, ROUGE-L, Semantic, Coherence
│   │   └── llm_judge.py          # LLM-as-judge scoring
│   ├── prompt_manager/
│   │   └── store.py              # SQLite store + CSV import/export
│   └── models/
│       └── ollama_connector.py   # Ollama HTTP wrapper
├── sample_data/
│   └── sample_prompts.csv        # 20-row sample for testing
├── data/                         # SQLite DB lives here (auto-created)
└── requirements.txt
```

---

## ⚙️ Concurrency Settings

In the sidebar:
- **1 (Sequential)** — safest, works with any local Ollama setup
- **2** — 2× faster, light on RAM
- **4** — fastest, needs Ollama able to handle parallel requests

For 200+ variants without a judge, sequential at ~2s/variant ≈ ~7 minutes.
With concurrency=4, same batch ≈ ~2 minutes.

---

## 📊 Metrics

| Metric | Range | Description |
|---|---|---|
| BLEU | 0–1 | N-gram overlap with reference (requires reference answer) |
| ROUGE-L | 0–1 | Longest common subsequence F1 |
| Semantic Similarity | 0–1 | Cosine similarity via sentence-transformers |
| Coherence | 0–1 | Heuristic: sentence structure, transitions, uniqueness |
| Faithfulness | 0–1 | LLM judge: addresses the task without hallucination |
| Completeness | 0–1 | LLM judge: covers all aspects |
| Clarity | 0–1 | LLM judge: clear and well-structured |
| Usefulness | 0–1 | LLM judge: genuinely helpful |
| **Composite** | 0–100 | Weighted combination of all above |
