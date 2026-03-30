"""
Leaderboard — Page 3
Ranks all prompt runs by composite score.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import plotly.express as px

from src.prompt_manager.store import get_leaderboard, get_stats

st.set_page_config(page_title="Leaderboard", page_icon="🏆", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
  .rank-1 { color: #fbbf24; font-size: 1.5rem; }
  .rank-2 { color: #94a3b8; font-size: 1.3rem; }
  .rank-3 { color: #d97706; font-size: 1.2rem; }
  .podium-card { background: #12122a; border: 1px solid rgba(255,255,255,0.08);
                 border-radius: 12px; padding: 1.5rem; text-align: center; }
  .podium-score { font-size: 2.8rem; font-weight: 800; color: #a78bfa; }
  .podium-name { font-weight: 700; color: white; font-size: 1.1rem; margin-bottom: 0.3rem; }
  .podium-meta { color: rgba(255,255,255,0.4); font-size: 0.8rem; }
  .stButton > button { background: linear-gradient(135deg, #7c3aed, #a78bfa) !important;
                        color: white !important; border: none !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🏆 Leaderboard")
st.caption("All-time best prompt runs ranked by composite score")

records = get_leaderboard(limit=100)
stats = get_stats()

if not records:
    st.info("No evaluation runs yet. Head to **Prompt Lab** to run your first evaluation!")
    st.stop()

# ── Stats Strip ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Runs", stats.get("total_runs", 0))
c2.metric("Unique Prompts", stats.get("unique_prompts", 0))
c3.metric("Best Score", f"{stats.get('best_score', 0):.1f}/100")
c4.metric("Avg Score", f"{(stats.get('avg_composite') or 0):.1f}/100")

st.markdown("---")

# ── Podium ─────────────────────────────────────────────────────────────────────
if len(records) >= 3:
    st.markdown("### 🥇 Top 3")
    medals = ["🥇", "🥈", "🥉"]
    cols = st.columns(3)
    for i, (col, rec) in enumerate(zip(cols, records[:3])):
        with col:
            st.markdown(f"""
            <div class="podium-card">
              <div style="font-size:2rem">{medals[i]}</div>
              <div class="podium-name">{rec['prompt_name']}</div>
              <div class="podium-score">{rec['composite']:.1f}</div>
              <div class="podium-meta">Model: {rec.get('model','—')}</div>
              <div class="podium-meta">{rec['timestamp'][:16]}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("---")

# ── Full Table ─────────────────────────────────────────────────────────────────
st.markdown("### 📋 Full Rankings")

df = pd.DataFrame(records)
display_cols = ["prompt_name", "composite", "semantic_similarity",
                "faithfulness", "coherence", "bleu", "rouge_l", "model", "timestamp"]
display_cols = [c for c in display_cols if c in df.columns]
df_display = df[display_cols].copy()
df_display.insert(0, "Rank", range(1, len(df_display) + 1))
df_display.columns = ["Rank", "Prompt", "Score/100", "Semantic", "Faithfulness",
                       "Coherence", "BLEU", "ROUGE-L", "Model", "Timestamp"][:len(df_display.columns)]
df_display["Score/100"] = df_display["Score/100"].round(1)
for col in ["Semantic", "Faithfulness", "Coherence", "BLEU", "ROUGE-L"]:
    if col in df_display.columns:
        df_display[col] = df_display[col].round(3)

st.dataframe(
    df_display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Score/100": st.column_config.ProgressColumn(
            "Score/100", min_value=0, max_value=100, format="%.1f"
        ),
    },
)

# ── Score Distribution ─────────────────────────────────────────────────────────
st.markdown("### 📊 Score Distribution")
fig = px.histogram(
    df, x="composite", nbins=20,
    color_discrete_sequence=["#a78bfa"],
    labels={"composite": "Composite Score"},
    title="Distribution of All Eval Scores",
)
fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="rgba(255,255,255,0.7)"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    margin=dict(t=40, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# ── Export ─────────────────────────────────────────────────────────────────────
st.markdown("### 💾 Export")
csv = df.to_csv(index=False)
st.download_button(
    label="⬇️ Download Full Results as CSV",
    data=csv,
    file_name="eval_leaderboard.csv",
    mime="text/csv",
)