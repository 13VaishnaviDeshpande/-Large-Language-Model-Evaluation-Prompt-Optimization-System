# LLM Evaluation & Prompt Optimization System

## Overview
The LLM Evaluation Tool is a comprehensive framework designed to evaluate and compare Large Language Model (LLM) outputs using deterministic metrics, LLM-based judging, and heuristic fallback scoring.

It provides a structured workflow for prompt experimentation, batch evaluation, model comparison, and performance tracking. Results are stored in a persistent SQLite database and visualized through a Streamlit-based dashboard.

---

## Key Features

- Interactive Prompt Lab for prompt experimentation
- Batch evaluation using CSV input
- Deterministic metric-based scoring
- LLM-as-a-Judge evaluation
- Heuristic fallback scoring
- Model comparison dashboard
- Leaderboard with weighted composite scoring
- SQLite-based persistent storage
- Concurrency support for faster evaluations
- Streamlit-based UI for visualization

---

## Project Structure

    .
    ├── pipeline.py
    ├── metrics.py
    ├── llm_judge.py
    ├── store.py
    ├── app.py
    ├── data/
    ├── outputs/
    └── README.md

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

## Workflow

### 1. Prompt Lab

The Prompt Lab allows interactive testing of prompts against selected models.

Steps:
- Enter a prompt
- Select a model
- Generate output
- Evaluate response using metrics
- Store results

#### Screenshot
<img width="1888" height="905" alt="Screenshot 2026-03-31 183932" src="https://github.com/user-attachments/assets/71754d4a-5085-4071-9f21-7f08fef0a7e1" />


---

### 2. Run Evaluation & Results

After running evaluation, the system ranks prompt variants and computes scores.

Displayed metrics include:
- Composite score
- Semantic similarity
- Latency
- Error tracking

#### Screenshot
![Evaluation Results](./images/evaluation_results.png)

---

### 3. Results Overview Dashboard

Provides aggregated insights across all evaluated prompts.

Includes:
- Total variants
- Best score
- Average score
- Score spread
- Failure count

#### Screenshot
![Results Overview](./images/results_overview.png)

---

### 4. Metric Radar Visualization

A radar chart comparing multiple evaluation dimensions:

- Semantic similarity
- Coherence
- Faithfulness
- Completeness
- Clarity
- BLEU / ROUGE

#### Screenshot
![Metric Radar](./images/metric_radar.png)

---

### 5. Per-Variant Analysis

Detailed breakdown of each prompt variant:

- Individual metric scores
- Prompt text
- LLM response
- Judge reasoning
- Latency and token usage

#### Screenshot
![Per Variant Analysis](./images/per_variant.png)

---

### 6. Batch Analysis

Batch Analysis enables evaluation of multiple prompts using CSV input or manual runs.

#### Overview
- Total variants processed
- Completed evaluations
- Source type (Manual/CSV)
- Model used
- Task description

#### Screenshot
![Batch Analysis](./images/batch_analysis.png)

#### Score Summary
- Best Score: 76.2
- Worst Score: 69.3
- Mean Score: 74.2
- Score Spread: 6.8
- Failed: 0

#### Features
- Score distribution visualization
- Metric correlation heatmap
- Performance consistency analysis

---

## Metrics

### Deterministic Metrics

- Exact Match
- Token Overlap
- Length Ratio
- Keyword Coverage
- Response Time

### LLM Judge Metrics

- Relevance
- Correctness
- Completeness
- Clarity
- Coherence

### Heuristic Fallback

If LLM judge fails, rule-based scoring is applied.

---

## CSV Input Format

    prompt,reference
    What is AI?,Artificial Intelligence definition
    Explain recursion,Definition of recursion

Fields:
- prompt: Input prompt
- reference: Expected output (optional but recommended)

---

## Concurrency

The evaluation pipeline supports concurrent execution.

Benefits:
- Reduced evaluation time
- Parallel model execution
- Efficient batch processing

---

## API Reference

### pipeline.py

Functions:
- run_single_prompt()
- run_batch()
- evaluate_output()
- compute_scores()

Responsibilities:
- Manage evaluation workflow
- Call metrics
- Invoke LLM judge
- Store results

---

### metrics.py

Functions:
- exact_match()
- token_overlap()
- length_ratio()
- keyword_coverage()

---

### llm_judge.py

Functions:
- judge_response()
- parse_judge_output()
- fallback_heuristic()

---

### store.py

Functions:
- init_db()
- insert_result()
- fetch_results()
- fetch_leaderboard()

---

## Database

SQLite is used for persistent storage.

### Tables

#### results
- id
- prompt
- model
- output
- metrics
- timestamp

#### leaderboard
- model
- composite_score
- total_runs

---

## Installation

    git clone <repository-url>
    cd llm-evaluation-tool
    pip install -r requirements.txt

---

## Running the Application

    streamlit run app.py

---

## Usage

- Launch the application
- Use Prompt Lab for testing prompts
- Upload CSV for batch evaluation
- Compare models using dashboard
- View leaderboard rankings

---

## Output

The system generates:

- Prompt-level scores
- Model-level aggregates
- Leaderboard rankings
- Stored evaluation history

---

## Extending Metrics

To add a new metric:

1. Implement function in metrics.py  
2. Register it in pipeline.py  
3. Add weight in composite score calculation  

---

## Error Handling

- LLM timeout handling
- Heuristic fallback scoring
- Partial result storage
- Retry logic

---

## Future Improvements

- Additional evaluation metrics
- Multi-dataset support
- Export to JSON and Excel
- Enhanced visualizations
- CI-based regression testing

---

## License

MIT License

---

## Author

Vaishnavi Deshpande
