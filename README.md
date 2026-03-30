# 🚀 LLM Evaluation & Prompt Optimization System

A **production-ready framework** for evaluating, comparing, and optimizing Large Language Model (LLM) prompt variants at scale.  
The system supports **bulk prompt testing (200+ variants)**, automated metrics, LLM-as-judge scoring, and rich analytics for data-driven prompt engineering.

---

## ✨ Key Features

- 📥 Bulk CSV prompt import (200+ variants)
- ⚡ Concurrent batch evaluation
- 📊 Advanced visual analytics (Radar, Scatter, Heatmap, Box plots)
- 🧠 LLM-as-judge scoring (faithfulness, clarity, usefulness)
- 📈 Batch comparison & leaderboard
- 💾 SQLite experiment tracking
- 📤 Export evaluation results to CSV
- 🔬 Per-variant deep analysis
- 📉 Score spread & discriminative insights

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama
ollama serve

# Pull model
ollama pull llama3

# Run application
streamlit run ui/app.py
```

---

## 📂 CSV Import Format

Minimum required column:

| Column | Description |
|--------|-------------|
| text | Prompt text (required) |
| name | Variant name (optional) |
| tags | Comma-separated tags |

Example:

```csv
name,text,tags
Simple,"Explain X simply.",beginner
Expert,"Explain X with examples.",expert
```

Sample dataset available in:
```
sample_data/sample_prompts.csv
```

---

## 🗺️ Application Pages

| Page | Description |
|------|-------------|
| Home | Overview dashboard |
| Prompt Lab | Run evaluations |
| Batch Analysis | Deep metrics visualization |
| Compare | Compare two runs |
| Leaderboard | Top scoring prompts |
| Analytics | Trends & correlations |

---

## 📊 Evaluation Metrics

| Metric | Range | Description |
|--------|------|-------------|
| BLEU | 0–1 | N-gram overlap |
| ROUGE-L | 0–1 | Sequence similarity |
| Semantic Similarity | 0–1 | Embedding cosine similarity |
| Coherence | 0–1 | Structural quality |
| Faithfulness | 0–1 | LLM judge |
| Completeness | 0–1 | LLM judge |
| Clarity | 0–1 | LLM judge |
| Usefulness | 0–1 | LLM judge |
| Composite Score | 0–100 | Weighted score |

---

## ⚙️ Concurrency Settings

| Threads | Description |
|---------|-------------|
| 1 | Sequential (safe) |
| 2 | Balanced performance |
| 4 | Fastest (parallel requests) |

---

## 🏗️ Project Structure

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

## 📈 Visualizations

- Radar charts
- Scatter plots
- Heatmaps
- Box plots
- Score distributions
- Batch comparisons

---

## 🧪 Typical Workflow

1. Upload prompt CSV  
2. Select model  
3. Configure concurrency  
4. Run evaluation  
5. Analyze metrics  
6. Compare batches  
7. Export results  

---

## 🛠️ Tech Stack

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

## 🎯 Use Cases

- Prompt engineering optimization  
- LLM benchmarking  
- Research experimentation  
- A/B prompt testing  
- Model comparison  
- Evaluation dashboards  

---

## 📄 License
MIT License

---

## 👩‍💻 Author

**Vaishnavi Deshpande**
