"""
Batch Analysis — Page 2
Deep per-variant analysis for any completed batch run.
Supports CSV-imported batches with 200+ variants.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.prompt_manager.store import (
    list_batches, get_results, get_variant_analysis,
    export_results_to_csv,
)

st.set_page_config(page_title="Batch Analysis", page_icon="🧪", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
  .metric-chip { display: inline-block; background: rgba(167,139,250,0.12);
                  border: 1px solid rgba(167,139,250,0.3); border-radius: 6px;
                  padding: 3px 10px; font-size: 0.8rem; color: #a78bfa;
                  font-family: 'JetBrains Mono', monospace; margin: 2px; }
  .rank-1 { color: #fbbf24; }
  .rank-2 { color: #94a3b8; }
  .rank-3 { color: #d97706; }
  .stButton > button { background: linear-gradient(135deg, #7c3aed, #a78bfa) !important;
                        color: white !important; border: none !important;
                        border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🧪 Batch Analysis")
st.caption("Per-variant deep dive for any evaluation batch")

batches = list_batches(limit=50)
if not batches:
    st.info("No batch runs yet. Run an evaluation in **Prompt Lab** first.")
    st.stop()

# Batch selector
batch_labels = {
    f"{b['batch_id']} — {b['source'].upper()} — {b['total_variants']} variants — {(b['started_at'] or '')[:16]}": b
    for b in batches
}
selected_label = st.selectbox("Select Batch Run", list(batch_labels.keys()))
batch = batch_labels[selected_label]

st.markdown("---")

# Batch metadata
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Variants", batch["total_variants"])
c2.metric("Completed",      batch["completed"])
c3.metric("Source",         batch["source"].upper())
c4.metric("Model",          batch.get("model", "—"))

st.markdown(f"**Task:** {batch.get('task_description','—')}")
if batch.get("reference_answer"):
    st.markdown(f"**Reference:** {batch['reference_answer'][:200]}")

st.markdown("---")

# Load results
rows = get_variant_analysis(batch["batch_id"])
if not rows:
    st.warning("No results found for this batch.")
    st.stop()

df = pd.DataFrame(rows)
numeric_cols = ["composite", "bleu", "rouge_l", "semantic_similarity",
                "coherence", "faithfulness", "completeness", "clarity", "usefulness"]
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# ── Summary Stats ──────────────────────────────────────────────────────────────
st.markdown("### 📊 Score Summary")
s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("Best",   f"{df['composite'].max():.1f}/100")
s2.metric("Worst",  f"{df['composite'].min():.1f}/100")
s3.metric("Mean",   f"{df['composite'].mean():.1f}/100")
s4.metric("Spread", f"{df['composite'].max()-df['composite'].min():.1f}")
s5.metric("Failed", int((df.get("error","") != "").sum()) if "error" in df.columns else 0)

# ── Score Distribution ────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### 📉 Score Distribution")
    fig = px.histogram(df, x="composite", nbins=min(30, len(df)//2 or 1),
                       color_discrete_sequence=["#a78bfa"],
                       labels={"composite": "Composite Score"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="rgba(255,255,255,0.7)"),
                      xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                      yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                      margin=dict(t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.markdown("### 🔗 Metric Correlation")
    corr_cols = [c for c in numeric_cols if c in df.columns]
    corr = df[corr_cols].corr()
    fig2 = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="Purples", text=corr.round(2).values,
        texttemplate="%{text}", showscale=True,
    ))
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="rgba(255,255,255,0.7)"),
                       margin=dict(t=20,b=20,l=20,r=20))
    st.plotly_chart(fig2, use_container_width=True)

# ── Metric Spread Box Plot ────────────────────────────────────────────────────
st.markdown("### 📦 Metric Distribution Across Variants")
melt = df[corr_cols].melt(var_name="Metric", value_name="Score")
fig3 = px.box(melt, x="Metric", y="Score",
              color="Metric",
              color_discrete_sequence=["#a78bfa","#34d399","#f59e0b",
                                       "#f87171","#60a5fa","#fb923c","#4ade80"])
fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                   font=dict(color="rgba(255,255,255,0.7)"),
                   yaxis=dict(range=[0,1], gridcolor="rgba(255,255,255,0.06)"),
                   xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                   showlegend=False, margin=dict(t=20,b=20))
st.plotly_chart(fig3, use_container_width=True)

# ── Scatter: Latency vs Score ─────────────────────────────────────────────────
if "latency_ms" in df.columns:
    st.markdown("### ⚡ Latency vs Score")
    import numpy as np
    fig4 = px.scatter(df, x="latency_ms", y="composite",
                      hover_data=["prompt_name"],
                      color_discrete_sequence=["#a78bfa"],
                      labels={"latency_ms": "Latency (ms)", "composite": "Composite"})
    if len(df) >= 2:
        m, b = np.polyfit(df["latency_ms"], df["composite"], 1)
        x_r = np.linspace(df["latency_ms"].min(), df["latency_ms"].max(), 40)
        fig4.add_trace(go.Scatter(x=x_r, y=m*x_r+b, mode="lines",
                                  name="Trend", line=dict(color="#f59e0b", dash="dash")))
    fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="rgba(255,255,255,0.7)"),
                       xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                       yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                       margin=dict(t=20,b=20))
    st.plotly_chart(fig4, use_container_width=True)

# ── Full Ranked Table ─────────────────────────────────────────────────────────
st.markdown("### 🏅 All Variants Ranked")

disp_cols = ["prompt_name", "composite", "semantic_similarity", "faithfulness",
             "coherence", "completeness", "clarity", "bleu", "rouge_l", "latency_ms"]
disp_cols = [c for c in disp_cols if c in df.columns]
disp_df = df[disp_cols].copy()
disp_df.insert(0, "Rank", range(1, len(disp_df)+1))

rename = {
    "prompt_name": "Prompt", "composite": "Score/100",
    "semantic_similarity": "Semantic", "faithfulness": "Faith.",
    "coherence": "Coherence", "completeness": "Complete.",
    "clarity": "Clarity", "bleu": "BLEU", "rouge_l": "ROUGE-L",
    "latency_ms": "Latency(ms)",
}
disp_df.rename(columns=rename, inplace=True)

st.dataframe(
    disp_df,
    use_container_width=True, hide_index=True,
    column_config={
        "Score/100": st.column_config.ProgressColumn(
            "Score/100", min_value=0, max_value=100, format="%.1f"),
    },
)

# Export
csv_export = export_results_to_csv(rows)
st.download_button("⬇️ Export Batch Results as CSV",
                   data=csv_export, file_name=f"batch_{batch['batch_id']}.csv",
                   mime="text/csv")

# ── Per-Variant Detail Expanders ──────────────────────────────────────────────
st.markdown("### 🔍 Per-Variant Detail")
st.caption("Expand any variant to see full response, prompt, and judge reasoning")

page_size = 25
total_pages = (len(rows) + page_size - 1) // page_size
if total_pages > 1:
    page = st.selectbox("Page", list(range(1, total_pages+1)),
                        format_func=lambda p: f"Page {p} ({(p-1)*page_size+1}–{min(p*page_size,len(rows))})")
    page_rows = rows[(page-1)*page_size: page*page_size]
else:
    page_rows = rows

for row in page_rows:
    rank = row.get("rank", "?")
    score = row.get("composite", 0) or 0
    score_color = "#34d399" if score >= 70 else "#f59e0b" if score >= 45 else "#f87171"
    with st.expander(f"#{rank}  {row.get('prompt_name','—')}  —  {score:.1f}/100"):
        if row.get("error"):
            st.error(f"Error: {row['error']}")
        ca, cb = st.columns([1, 2])
        with ca:
            st.markdown(f"""
            <div style="text-align:center;margin-bottom:1rem">
              <span style="font-size:2.5rem;font-weight:800;color:{score_color}">{score:.1f}</span>
              <div style="color:rgba(255,255,255,0.4);font-size:0.8rem">/ 100</div>
            </div>
            """, unsafe_allow_html=True)
            metrics_show = [
                ("Semantic",     row.get("semantic_similarity", 0)),
                ("Coherence",    row.get("coherence", 0)),
                ("Faithfulness", row.get("faithfulness", 0)),
                ("Completeness", row.get("completeness", 0)),
                ("Clarity",      row.get("clarity", 0)),
                ("BLEU",         row.get("bleu", 0)),
                ("ROUGE-L",      row.get("rouge_l", 0)),
            ]
            for lbl, val in metrics_show:
                val = val or 0
                bar_w = int(val * 100)
                bc = "#34d399" if val >= 0.7 else "#f59e0b" if val >= 0.4 else "#f87171"
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;padding:4px 0;
                            border-bottom:1px solid rgba(255,255,255,0.05)">
                  <span style="color:rgba(255,255,255,0.65);font-size:0.85rem">{lbl}</span>
                  <span style="font-family:'JetBrains Mono',monospace;color:#a78bfa;font-size:0.85rem">{val:.4f}</span>
                </div>
                <div style="height:2px;background:rgba(255,255,255,0.06);margin-bottom:3px">
                  <div style="height:2px;width:{bar_w}%;background:{bc}"></div>
                </div>
                """, unsafe_allow_html=True)
            st.caption(f"Latency: {row.get('latency_ms',0)}ms")
        with cb:
            if row.get("prompt_text"):
                st.markdown("**Prompt:**")
                st.code(row["prompt_text"], language=None)
            st.markdown("**Response:**")
            resp = row.get("llm_response") or "—"
            st.markdown(f"""<div style="background:#0d1117;border:1px solid rgba(255,255,255,0.1);
                            border-radius:8px;padding:1rem;font-family:'JetBrains Mono',monospace;
                            font-size:0.83rem;color:#e2e8f0;white-space:pre-wrap;
                            max-height:220px;overflow-y:auto">{resp}</div>""",
                        unsafe_allow_html=True)
            if row.get("judge_reasoning"):
                st.markdown("**Judge:**")
                st.markdown(f"""<div style="background:rgba(167,139,250,0.08);
                                border-left:3px solid #a78bfa;border-radius:0 8px 8px 0;
                                padding:0.8rem;color:rgba(255,255,255,0.7);font-size:0.85rem;
                                font-style:italic">{row['judge_reasoning']}</div>""",
                            unsafe_allow_html=True)
