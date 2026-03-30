"""
LLM Evaluation & Prompt Optimization System — Pro
Main entry point.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.prompt_manager.store import get_stats, init_db
from src.models.ollama_connector import is_ollama_running, list_models

st.set_page_config(
    page_title="LLM Eval Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
  .hero { background: linear-gradient(135deg, #0d0d1f 0%, #1a0a2e 50%, #0d0d1f 100%);
          border: 1px solid rgba(167,139,250,0.2); border-radius: 16px;
          padding: 2.5rem 3rem; margin-bottom: 2rem; }
  .hero-title { font-size: 2.8rem; font-weight: 800; color: white;
                background: linear-gradient(135deg,#a78bfa,#60a5fa); -webkit-background-clip:text;
                -webkit-text-fill-color:transparent; margin-bottom: 0.4rem; }
  .hero-sub { color: rgba(255,255,255,0.5); font-size: 1.05rem; }
  .stat-card { background: #12122a; border: 1px solid rgba(255,255,255,0.08);
               border-radius: 12px; padding: 1.4rem; text-align: center; }
  .stat-val  { font-size: 2.4rem; font-weight: 800; color: #a78bfa; }
  .stat-lbl  { color: rgba(255,255,255,0.45); font-size: 0.82rem;
               text-transform: uppercase; letter-spacing: 1px; }
  .feature-card { background: #12122a; border: 1px solid rgba(255,255,255,0.07);
                  border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 0.8rem; }
  .feature-title { color: #a78bfa; font-weight: 700; font-size: 1rem; margin-bottom: 0.3rem; }
  .feature-body  { color: rgba(255,255,255,0.55); font-size: 0.88rem; line-height: 1.6; }
  .ollama-ok  { color: #34d399; font-weight: 600; }
  .ollama-err { color: #f87171; font-weight: 600; }
  .stButton > button { background: linear-gradient(135deg,#7c3aed,#a78bfa) !important;
                        color: white !important; border: none !important;
                        border-radius: 8px !important; font-family:'Syne',sans-serif !important;
                        font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

init_db()
stats   = get_stats()
ollama  = is_ollama_running()
models  = list_models() if ollama else []

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">⚡ LLM Eval Pro</div>
  <div class="hero-sub">
    Evaluate &amp; compare prompt variants at scale — manual entry or CSV import (200+ variants),
    per-variant analysis, leaderboards, and actionable insights.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Status ─────────────────────────────────────────────────────────────────────
if ollama:
    st.markdown(f'<span class="ollama-ok">● Ollama running</span> — {len(models)} model(s) available: '
                f'`{"`, `".join(models[:5])}`', unsafe_allow_html=True)
else:
    st.markdown('<span class="ollama-err">● Ollama not detected</span> — '
                'Start with `ollama serve` and pull a model with `ollama pull llama3`',
                unsafe_allow_html=True)

st.markdown("---")

# ── Stats ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in [
    (c1, stats.get("total_runs") or 0,                              "Total Evaluations"),
    (c2, stats.get("unique_prompts") or 0,                          "Unique Prompts"),
    (c3, f"{stats.get('best_score') or 0:.1f}",                     "Best Score /100"),
    (c4, f"{stats.get('avg_composite') or 0:.1f}",                  "Avg Score /100"),
]:
    col.markdown(f"""
    <div class="stat-card">
      <div class="stat-val">{val}</div>
      <div class="stat-lbl">{lbl}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ── Features ───────────────────────────────────────────────────────────────────
st.markdown("### 🗺️ Navigation Guide")
left, right = st.columns(2)

features_left = [
    ("🔬 Prompt Lab",
     "Write or import prompt variants. Supports 200+ at once via CSV upload. "
     "Real-time progress bar, live score table, per-variant deep analysis, and export."),
    ("🧪 Batch Analysis",
     "Drill into any completed batch run. Score distribution, metric box plots, "
     "latency scatter, correlation heatmap, and paginated per-variant details."),
    ("⚖️ Compare",
     "Side-by-side comparison of any two evaluation runs with metric bar charts "
     "and word-level response diff."),
]
features_right = [
    ("🏆 Leaderboard",
     "All-time top runs ranked by composite score. Podium view for top 3, "
     "full sortable table, and score distribution histogram."),
    ("📊 Analytics",
     "Score trends over time, metric correlations, latency vs score scatter, "
     "model comparison, and auto-generated insights."),
    ("📂 CSV Import Format",
     "Required column: `text` or `prompt`. Optional: `name`, `tags`. "
     "Upload up to 200+ rows, preview before running, export results back to CSV."),
]

with left:
    for title, body in features_left:
        st.markdown(f"""
        <div class="feature-card">
          <div class="feature-title">{title}</div>
          <div class="feature-body">{body}</div>
        </div>""", unsafe_allow_html=True)

with right:
    for title, body in features_right:
        st.markdown(f"""
        <div class="feature-card">
          <div class="feature-title">{title}</div>
          <div class="feature-body">{body}</div>
        </div>""", unsafe_allow_html=True)

# ── Quick Start ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🚀 Quick Start")
st.code("""# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull a model (if not already)
ollama pull llama3

# 3. Run
streamlit run ui/app.py
""", language="bash")

st.markdown("**CSV tip:** Column named `text` or `prompt` is all you need. "
            "Add `name` for readable labels and `tags` for filtering.")
