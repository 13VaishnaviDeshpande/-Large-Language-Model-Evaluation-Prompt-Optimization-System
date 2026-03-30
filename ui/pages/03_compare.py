"""
Compare Page — Page 2
Side-by-side comparison of two evaluation runs with diff highlighting.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import plotly.graph_objects as go
import difflib

from src.prompt_manager.store import get_results

st.set_page_config(page_title="Compare Prompts", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
  .diff-add { background: rgba(52,211,153,0.15); color: #34d399; }
  .diff-rem { background: rgba(248,113,113,0.15); color: #f87171; }
  .cmp-card { background: #12122a; border: 1px solid rgba(255,255,255,0.08);
              border-radius: 12px; padding: 1.5rem; height: 100%; }
  .cmp-title { font-size: 1.1rem; font-weight: 700; color: #a78bfa; margin-bottom: 1rem; }
  .cmp-score { font-size: 2.5rem; font-weight: 800; }
  .score-win { color: #34d399; }
  .score-lose { color: rgba(255,255,255,0.4); }
  .response-pre { font-family: 'JetBrains Mono', monospace; font-size: 0.83rem;
                  color: #e2e8f0; white-space: pre-wrap; max-height: 280px;
                  overflow-y: auto; line-height: 1.6; }
  .stButton > button { background: linear-gradient(135deg, #7c3aed, #a78bfa) !important;
                        color: white !important; border: none !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("## ⚖️ Compare Evaluations")
st.caption("Select two evaluation runs to compare side-by-side")

all_results = get_results(limit=500)

if len(all_results) < 2:
    st.info("You need at least 2 evaluation runs to compare. Head to **Prompt Lab** first!")
    st.stop()

result_labels = {
    f"{r['prompt_name']} — {r['timestamp'][:16]} (score: {r['composite']:.1f})": r
    for r in all_results
}
labels = list(result_labels.keys())

col1, col2 = st.columns(2)
with col1:
    left_label = st.selectbox("Left prompt", labels, index=0)
with col2:
    right_label = st.selectbox("Right prompt", labels, index=min(1, len(labels)-1))

left = result_labels[left_label]
right = result_labels[right_label]

st.markdown("---")

# ── Score Comparison ──────────────────────────────────────────────────────────
lc, _, rc = st.columns([5, 1, 5])

def score_class(a, b):
    return "score-win" if a >= b else "score-lose"

with lc:
    cls = score_class(left['composite'], right['composite'])
    st.markdown(f"""
    <div class="cmp-card">
      <div class="cmp-title">🔵 {left['prompt_name']}</div>
      <div class="cmp-score {cls}">{left['composite']:.1f}<span style="font-size:1rem;opacity:.5">/100</span></div>
      <br/>
      <b>Prompt:</b><br/>
      <div class="response-pre" style="background:#0d1117;padding:0.8rem;border-radius:8px;margin-top:0.5rem">
        {left['prompt_text'] or '—'}
      </div>
    </div>""", unsafe_allow_html=True)

with rc:
    cls = score_class(right['composite'], left['composite'])
    st.markdown(f"""
    <div class="cmp-card">
      <div class="cmp-title">🟣 {right['prompt_name']}</div>
      <div class="cmp-score {cls}">{right['composite']:.1f}<span style="font-size:1rem;opacity:.5">/100</span></div>
      <br/>
      <b>Prompt:</b><br/>
      <div class="response-pre" style="background:#0d1117;padding:0.8rem;border-radius:8px;margin-top:0.5rem">
        {right['prompt_text'] or '—'}
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ── Metric Bar Chart ──────────────────────────────────────────────────────────
metrics = ["bleu", "rouge_l", "semantic_similarity", "coherence", "faithfulness"]
labels_m = ["BLEU", "ROUGE-L", "Semantic", "Coherence", "Faithfulness"]

fig = go.Figure(data=[
    go.Bar(name=left['prompt_name'], x=labels_m,
           y=[left.get(m, 0) for m in metrics],
           marker_color="#60a5fa", opacity=0.85),
    go.Bar(name=right['prompt_name'], x=labels_m,
           y=[right.get(m, 0) for m in metrics],
           marker_color="#a78bfa", opacity=0.85),
])
fig.update_layout(
    barmode="group",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="rgba(255,255,255,0.7)"),
    yaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.06)"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=20, b=20),
    title="Metric Breakdown",
)
st.plotly_chart(fig, width="stretch")

# ── Response Diff ─────────────────────────────────────────────────────────────
st.markdown("### 📝 Response Diff")
lc2, rc2 = st.columns(2)

left_resp = (left.get("llm_response") or "").strip()
right_resp = (right.get("llm_response") or "").strip()

with lc2:
    st.markdown(f"**🔵 {left['prompt_name']}**")
    st.markdown(f'<div class="response-pre" style="background:#0d1117;padding:1rem;border-radius:8px">{left_resp}</div>',
                unsafe_allow_html=True)

with rc2:
    st.markdown(f"**🟣 {right['prompt_name']}**")
    st.markdown(f'<div class="response-pre" style="background:#0d1117;padding:1rem;border-radius:8px">{right_resp}</div>',
                unsafe_allow_html=True)

# Unified diff
st.markdown("### 🔄 Unified Diff (word-level)")
diff = list(difflib.unified_diff(
    left_resp.split(), right_resp.split(),
    lineterm="", n=3,
))
if diff:
    diff_html = []
    for line in diff[:80]:
        if line.startswith("+") and not line.startswith("+++"):
            diff_html.append(f'<span class="diff-add">+ {line[1:]}</span>')
        elif line.startswith("-") and not line.startswith("---"):
            diff_html.append(f'<span class="diff-rem">- {line[1:]}</span>')
        else:
            diff_html.append(f'<span style="color:rgba(255,255,255,0.4)">  {line}</span>')
    st.markdown(
        f'<pre style="font-family:JetBrains Mono,monospace;font-size:0.82rem;background:#0d1117;padding:1rem;border-radius:8px;overflow-x:auto">'
        + "\n".join(diff_html) + "</pre>",
        unsafe_allow_html=True,
    )
else:
    st.info("Responses are identical.")

# ── Judge Reasoning ──────────────────────────────────────────────────────────
if left.get("judge_reasoning") or right.get("judge_reasoning"):
    st.markdown("### 💬 Judge Reasoning")
    lc3, rc3 = st.columns(2)
    with lc3:
        st.info(left.get("judge_reasoning") or "No judge reasoning available")
    with rc3:
        st.info(right.get("judge_reasoning") or "No judge reasoning available")
