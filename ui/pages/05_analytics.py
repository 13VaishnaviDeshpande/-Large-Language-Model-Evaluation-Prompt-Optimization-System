"""
Analytics — Page 4
Score trends over time, metric distributions, model comparisons.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.prompt_manager.store import get_results

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
  .insight-card { background: rgba(167,139,250,0.08); border: 1px solid rgba(167,139,250,0.2);
                  border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.8rem; }
  .insight-title { color: #a78bfa; font-weight: 700; margin-bottom: 0.3rem; }
  .insight-body { color: rgba(255,255,255,0.65); font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 📊 Analytics")
st.caption("Score trends, metric correlations, and model comparisons across all runs")

records = get_results(limit=500)
if not records:
    st.info("No evaluation runs yet. Head to **Prompt Lab** to run your first evaluation!")
    st.stop()

df = pd.DataFrame(records)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

numeric_metrics = ["composite", "semantic_similarity", "faithfulness",
                   "coherence", "bleu", "rouge_l"]
for col in numeric_metrics:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# ── Score Over Time ────────────────────────────────────────────────────────────
st.markdown("### 📈 Score Over Time")
fig = px.line(
    df, x="timestamp", y="composite",
    color="model" if "model" in df.columns else None,
    markers=True,
    labels={"composite": "Composite Score", "timestamp": "Time"},
    color_discrete_sequence=["#a78bfa", "#34d399", "#f59e0b", "#f87171"],
)
fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="rgba(255,255,255,0.7)"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", range=[0, 100]),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=20, b=20),
)
st.plotly_chart(fig, width="stretch")

# ── Metric Correlations ────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔗 Metric Correlations")
    corr_metrics = [m for m in numeric_metrics if m in df.columns]
    corr_df = df[corr_metrics].corr()
    fig2 = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale="Purples",
        text=corr_df.round(2).values,
        texttemplate="%{text}",
        showscale=True,
    ))
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.7)"),
        margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(fig2, width="stretch")

with col2:
    st.markdown("### 🕰️ Latency vs Score")
    if "latency_ms" in df.columns:
        # Manual trendline via numpy (no statsmodels dependency)
        import numpy as np
        x_vals = df["latency_ms"].astype(float)
        y_vals = df["composite"].astype(float)
        fig3 = px.scatter(
            df, x="latency_ms", y="composite",
            color="model" if "model" in df.columns else None,
            labels={"latency_ms": "Latency (ms)", "composite": "Score"},
            color_discrete_sequence=["#a78bfa", "#34d399", "#f59e0b"],
        )
        # Add numpy trendline manually
        if len(x_vals) >= 2:
            m, b = np.polyfit(x_vals, y_vals, 1)
            x_range = np.linspace(x_vals.min(), x_vals.max(), 50)
            fig3.add_trace(go.Scatter(
                x=x_range, y=m * x_range + b,
                mode="lines", name="Trend",
                line=dict(color="#f59e0b", width=1.5, dash="dash"),
            ))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.7)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig3, width="stretch")

# ── Model Comparison ──────────────────────────────────────────────────────────
if "model" in df.columns and df["model"].nunique() > 1:
    st.markdown("### 🤖 Model Comparison")
    model_avg = df.groupby("model")[numeric_metrics].mean().reset_index()
    fig4 = px.bar(
        model_avg.melt(id_vars="model", value_vars=numeric_metrics),
        x="variable", y="value", color="model", barmode="group",
        labels={"variable": "Metric", "value": "Avg Score"},
        color_discrete_sequence=["#a78bfa", "#34d399", "#f59e0b", "#f87171"],
    )
    fig4.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.7)"),
        yaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.06)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig4, width="stretch")

# ── Auto Insights ──────────────────────────────────────────────────────────────
st.markdown("### 💡 Auto Insights")

insights = []
if len(df) >= 3:
    best = df.loc[df["composite"].idxmax()]
    worst = df.loc[df["composite"].idxmin()]
    insights.append({
        "title": "Best Performing Prompt",
        "body": f"'{best['prompt_name']}' scored {best['composite']:.1f}/100 with {best['semantic_similarity']:.3f} semantic similarity"
    })
    insights.append({
        "title": "Score Improvement Over Time",
        "body": f"Average score in first half: {df.head(len(df)//2)['composite'].mean():.1f} vs second half: {df.tail(len(df)//2)['composite'].mean():.1f}"
    })
    high_sem = df["semantic_similarity"].corr(df["composite"])
    insights.append({
        "title": "Most Predictive Metric",
        "body": f"Semantic similarity has {high_sem:.2f} correlation with composite score — it's your most reliable signal"
    })
    avg_latency = df["latency_ms"].mean() if "latency_ms" in df else 0
    insights.append({
        "title": "Performance",
        "body": f"Average inference latency: {avg_latency:.0f}ms. Faster doesn't mean worse — see the scatter chart above."
    })

for ins in insights:
    st.markdown(f"""
    <div class="insight-card">
      <div class="insight-title">💡 {ins['title']}</div>
      <div class="insight-body">{ins['body']}</div>
    </div>""", unsafe_allow_html=True)
