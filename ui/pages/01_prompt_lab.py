"""
Prompt Lab — Enhanced
Supports manual entry (unlimited variants), CSV import (200+),
real-time progress, and detailed per-variant analysis.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import io

from src.evaluator.pipeline import run_batch_evaluation, generate_variant_report
from src.prompt_manager.store import (
    save_prompt, save_prompts_bulk, save_result, list_prompts,
    import_prompts_from_csv, export_results_to_csv,
    create_batch, update_batch_progress,
)
from src.models.ollama_connector import is_ollama_running, list_models

st.set_page_config(page_title="Prompt Lab", page_icon="🔬", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
  .score-ring { text-align: center; }
  .score-value { font-size: 3rem; font-weight: 800; color: #a78bfa; }
  .score-label { font-size: 0.85rem; color: rgba(255,255,255,0.5); text-transform: uppercase; letter-spacing: 1px; }
  .metric-row { display: flex; justify-content: space-between; align-items: center;
                padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.06); }
  .metric-name { color: rgba(255,255,255,0.7); font-size: 0.88rem; }
  .metric-val  { font-family: 'JetBrains Mono', monospace; color: #a78bfa; font-weight: 600; }
  .response-box { background: #0d1117; border: 1px solid rgba(255,255,255,0.1);
                  border-radius: 10px; padding: 1.2rem; font-family: 'JetBrains Mono', monospace;
                  font-size: 0.85rem; color: #e2e8f0; white-space: pre-wrap;
                  max-height: 260px; overflow-y: auto; margin-top: 0.5rem; }
  .judge-box  { background: rgba(167,139,250,0.08); border-left: 3px solid #a78bfa;
                border-radius: 0 8px 8px 0; padding: 1rem; color: rgba(255,255,255,0.7);
                font-size: 0.88rem; margin-top: 0.5rem; font-style: italic; }
  .rank-badge { display: inline-block; background: linear-gradient(135deg,#7c3aed,#a78bfa);
                color: white; border-radius: 20px; padding: 2px 10px;
                font-size: 0.78rem; font-weight: 700; margin-right: 6px; }
  .variant-card { background: #12122a; border: 1px solid rgba(255,255,255,0.07);
                  border-radius: 12px; padding: 1.2rem; margin-bottom: 0.8rem; }
  .stButton > button { background: linear-gradient(135deg, #7c3aed, #a78bfa) !important;
                        color: white !important; border: none !important;
                        border-radius: 8px !important; font-family: 'Syne',sans-serif !important;
                        font-weight: 600 !important; }
  .csv-hint { background: rgba(52,211,153,0.07); border: 1px solid rgba(52,211,153,0.2);
              border-radius: 8px; padding: 0.8rem 1rem; font-size: 0.85rem;
              color: rgba(255,255,255,0.7); margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🔬 Prompt Lab")
st.caption("Evaluate unlimited prompt variants — manual entry or CSV import")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Eval Config")
    models = list_models() if is_ollama_running() else []
    model = st.selectbox("Model", models or ["llama3"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    use_judge = st.checkbox("Use LLM-as-Judge", value=True)
    max_workers = st.selectbox("Concurrency", [1, 2, 4], index=0,
                               help="1 = sequential (safe), 2-4 = parallel (faster but needs more RAM)")
    st.markdown("---")
    st.markdown("### 💾 Saved Prompts")
    saved = list_prompts(limit=6)
    if saved:
        for p in saved:
            with st.expander(f"📝 {p['name'][:28]}"):
                st.code(p["text"][:180] + ("…" if len(p["text"]) > 180 else ""), language=None)
    else:
        st.caption("No saved prompts yet.")

# ── Task Setup ────────────────────────────────────────────────────────────────
st.markdown("### 1️⃣ Define the Task")
c1, c2 = st.columns([2, 1])
with c1:
    task_desc = st.text_area("Task Description",
                             placeholder="e.g. Explain gradient descent to a beginner", height=80)
with c2:
    reference = st.text_area("Reference Answer (optional)",
                             placeholder="Ideal answer for BLEU/ROUGE scoring…", height=80)

# ── Input Mode Tabs ────────────────────────────────────────────────────────────
st.markdown("### 2️⃣ Add Prompt Variants")
tab_manual, tab_csv = st.tabs(["✏️ Manual Entry", "📂 CSV Import"])

# ─ Manual Entry Tab ───────────────────────────────────────────────────────────
with tab_manual:
    if "prompt_variants" not in st.session_state:
        st.session_state.prompt_variants = [
            {"name": "Variant A", "text": ""},
            {"name": "Variant B", "text": ""},
        ]

    n = len(st.session_state.prompt_variants)
    st.caption(f"{n} variant(s) — add as many as you need")

    # Render in 3-column grid for space efficiency
    cols_per_row = 3
    for row_start in range(0, n, cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, col in enumerate(cols):
            i = row_start + col_idx
            if i >= n:
                break
            with col:
                variant = st.session_state.prompt_variants[i]
                name = st.text_input("Name", value=variant["name"], key=f"name_{i}")
                text = st.text_area("Prompt", value=variant["text"],
                                    height=140, key=f"text_{i}",
                                    placeholder=f"Variant {chr(65+i%26)}…")
                st.session_state.prompt_variants[i]["name"] = name
                st.session_state.prompt_variants[i]["text"] = text

    btn_col1, btn_col2, btn_col3, _ = st.columns([1, 1, 1, 4])
    with btn_col1:
        if st.button("➕ Add Variant"):
            n2 = len(st.session_state.prompt_variants)
            st.session_state.prompt_variants.append(
                {"name": f"Variant {n2+1}", "text": ""})
            st.rerun()
    with btn_col2:
        if st.button("➕ Add 10") and n < 500:
            n2 = len(st.session_state.prompt_variants)
            for j in range(10):
                st.session_state.prompt_variants.append(
                    {"name": f"Variant {n2+j+1}", "text": ""})
            st.rerun()
    with btn_col3:
        if st.button("− Remove Last") and len(st.session_state.prompt_variants) > 1:
            st.session_state.prompt_variants.pop()
            st.rerun()

    manual_active = True

# ─ CSV Import Tab ─────────────────────────────────────────────────────────────
with tab_csv:
    st.markdown("""
    <div class="csv-hint">
    📋 <strong>CSV Format</strong><br/>
    Required column: <code>text</code> or <code>prompt</code> or <code>prompt_text</code><br/>
    Optional columns: <code>name</code> (variant label), <code>tags</code> (comma-separated)<br/>
    Supports <strong>200+ rows</strong> in one upload.
    </div>
    """, unsafe_allow_html=True)

    # Download sample CSV
    sample_csv = "name,text,tags\nVariant A,Explain {topic} simply.,beginner\nVariant B,You are an expert. Explain {topic} with examples.,detailed\n"
    st.download_button("⬇️ Download Sample CSV", data=sample_csv,
                       file_name="sample_prompts.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"],
                                help="UTF-8 encoded CSV with a 'text' column")

    if uploaded:
        csv_content = uploaded.read().decode("utf-8")
        parsed_rows, parse_errors = import_prompts_from_csv(csv_content)

        if parse_errors:
            for e in parse_errors[:5]:
                st.warning(e)

        if parsed_rows:
            st.success(f"✅ Parsed **{len(parsed_rows)}** variants from CSV")
            preview_df = pd.DataFrame(parsed_rows)[["name", "text"]].head(10)
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
            if len(parsed_rows) > 10:
                st.caption(f"…and {len(parsed_rows)-10} more rows")

            st.session_state.csv_variants = parsed_rows
        else:
            st.session_state.csv_variants = []
    else:
        st.session_state.setdefault("csv_variants", [])

# ── Run Evaluation ────────────────────────────────────────────────────────────
st.markdown("### 3️⃣ Run Evaluation")

# Determine active variants
csv_variants   = st.session_state.get("csv_variants", [])
manual_variants = [v for v in st.session_state.prompt_variants if v["text"].strip()]

use_csv    = len(csv_variants) > 0
use_manual = len(manual_variants) > 0

if use_csv and use_manual:
    src_choice = st.radio("Which variants to evaluate?",
                          ["CSV variants", "Manual variants", "Both"],
                          horizontal=True)
    if src_choice == "CSV variants":
        active_variants = csv_variants
    elif src_choice == "Manual variants":
        active_variants = manual_variants
    else:
        active_variants = manual_variants + csv_variants
elif use_csv:
    active_variants = csv_variants
    st.info(f"Using **{len(csv_variants)}** CSV variants")
else:
    active_variants = manual_variants

st.metric("Variants queued", len(active_variants))

if st.button("🚀 Run Evaluation", use_container_width=True, type="primary"):
    if not task_desc.strip():
        st.error("Please enter a task description first.")
    elif not active_variants:
        st.error("Please add at least one prompt variant (or upload a CSV).")
    else:
        total = len(active_variants)
        progress_bar = st.progress(0, text="Starting…")
        status_text  = st.empty()
        live_table   = st.empty()

        results = []
        live_rows = []

        def on_progress(completed, total, r):
            pct = completed / total
            progress_bar.progress(pct, text=f"Evaluating {completed}/{total}…")
            status_text.markdown(
                f"**{r.prompt_name}** → composite **{r.composite:.1f}**/100"
                + (f" ⚠️ {r.error}" if r.error else "")
            )
            live_rows.append({
                "Rank": completed,
                "Prompt": r.prompt_name[:35],
                "Composite": f"{r.composite:.1f}",
                "Semantic": f"{r.semantic_similarity:.3f}",
                "Latency": f"{r.latency_ms}ms",
                "Error": r.error[:40] if r.error else "",
            })
            live_table.dataframe(
                pd.DataFrame(live_rows).tail(15),
                use_container_width=True, hide_index=True,
            )

        # Create batch record
        source = "csv" if use_csv else "manual"
        batch_id = create_batch(task_desc, reference, model, temperature, total, source)

        results = run_batch_evaluation(
            prompts=active_variants,
            task_description=task_desc,
            reference_answer=reference,
            model=model,
            temperature=temperature,
            use_llm_judge=use_judge,
            max_workers=max_workers,
            progress_callback=on_progress,
        )

        # Bulk save
        for i, r in enumerate(results):
            pid = save_prompt(r.prompt_name, r.prompt_text, source=source)
            r.prompt_id = pid
            save_result(r, batch_id=batch_id, csv_row=i)

        update_batch_progress(batch_id, len(results), finished=True)

        progress_bar.progress(1.0, text="✅ Complete!")
        status_text.empty()
        st.session_state.last_results = results
        st.session_state.last_batch_id = batch_id
        st.success(f"Evaluated **{len(results)}** variants | batch `{batch_id}`")

# ── Results ───────────────────────────────────────────────────────────────────
if "last_results" in st.session_state and st.session_state.last_results:
    results = st.session_state.last_results
    report  = generate_variant_report(results)

    st.markdown("---")
    st.markdown("### 📊 Results Overview")

    m1, m2, m3, m4, m5 = st.columns(5)
    sr = report.get("score_range", {})
    m1.metric("Total Variants", report.get("total_variants", 0))
    m2.metric("Best Score",  f"{sr.get('max', 0):.1f}/100")
    m3.metric("Avg Score",   f"{sr.get('mean', 0):.1f}/100")
    m4.metric("Score Spread",f"{sr.get('spread', 0):.1f}")
    m5.metric("Failed",      report.get("failed", 0))

    # Summary table with all variants
    rows = []
    for r in results:
        rows.append({
            "Prompt": r.prompt_name,
            "Composite": r.composite,
            "Semantic": round(r.semantic_similarity, 3),
            "Coherence": round(r.coherence, 3),
            "Faithfulness": round(r.faithfulness, 3),
            "Completeness": round(r.completeness, 3),
            "BLEU": round(r.bleu, 3),
            "ROUGE-L": round(r.rouge_l, 3),
            "Latency (ms)": r.latency_ms,
            "Error": r.error or "",
        })
    df = pd.DataFrame(rows)
    df.insert(0, "Rank", range(1, len(df) + 1))

    st.dataframe(
        df.drop(columns=["Rank"], errors="ignore"),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Composite": st.column_config.ProgressColumn(
                "Composite /100", min_value=0, max_value=100, format="%.1f"),
        },
    )

    # Export button
    csv_out = export_results_to_csv([r.to_dict() for r in results])
    st.download_button("⬇️ Export All Results as CSV",
                       data=csv_out, file_name="eval_results.csv", mime="text/csv")

    # Radar chart (up to 8 variants for readability)
    st.markdown("### 📡 Metric Radar")
    metric_keys   = ["bleu", "rouge_l", "semantic_similarity",
                     "coherence", "faithfulness", "completeness", "clarity"]
    metric_labels = ["BLEU", "ROUGE-L", "Semantic", "Coherence",
                     "Faithfulness", "Completeness", "Clarity"]
    colors = ["#a78bfa","#34d399","#f59e0b","#f87171","#60a5fa",
              "#fb923c","#4ade80","#e879f9"]

    display_results = results[:8]
    fig = go.Figure()
    for i, r in enumerate(display_results):
        vals = [getattr(r, k) for k in metric_keys]
        vals_c  = vals + [vals[0]]
        labels_c = metric_labels + [metric_labels[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_c, theta=labels_c, fill="toself",
            name=r.prompt_name[:25],
            line_color=colors[i % len(colors)], opacity=0.65,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1],
                                   color="rgba(255,255,255,0.3)")),
        showlegend=True, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
        margin=dict(t=40, b=40),
        title=f"Top {len(display_results)} variants" + (" (showing first 8)" if len(results) > 8 else ""),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Per-Variant Deep Analysis ─────────────────────────────────────────────
    st.markdown("### 🔍 Per-Variant Analysis")
    st.caption(f"Showing all {len(results)} variants — click a tab to inspect")

    # Use expanders for 200+ variants (tabs don't scale)
    page_size = 20
    total_pages = (len(results) + page_size - 1) // page_size
    if total_pages > 1:
        page = st.selectbox(
            f"Page (showing {page_size} variants per page)",
            list(range(1, total_pages + 1)), index=0,
            format_func=lambda p: f"Page {p} — Variants {(p-1)*page_size+1}–{min(p*page_size, len(results))}"
        )
        page_results = results[(page-1)*page_size: page*page_size]
    else:
        page_results = results

    for rank, result in enumerate(page_results, start=1 if total_pages <= 1 else (page-1)*page_size+1):
        score_color = "#34d399" if result.composite >= 70 else \
                      "#f59e0b"  if result.composite >= 45 else "#f87171"
        with st.expander(
            f"#{rank}  {result.prompt_name}  —  "
            f"Composite: {result.composite:.1f}/100", expanded=(rank == 1)
        ):
            if result.error:
                st.error(f"Error: {result.error}")
                st.code(result.prompt_text, language=None)
                continue

            ca, cb = st.columns([1, 2])
            with ca:
                st.markdown(f"""
                <div class="score-ring">
                  <div class="score-value" style="color:{score_color}">{result.composite:.1f}</div>
                  <div class="score-label">Composite / 100</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("")
                detail_metrics = [
                    ("Semantic Similarity", result.semantic_similarity),
                    ("Coherence",           result.coherence),
                    ("Faithfulness",        result.faithfulness),
                    ("Completeness",        result.completeness),
                    ("Clarity",             result.clarity),
                    ("Usefulness",          result.usefulness),
                    ("BLEU",                result.bleu),
                    ("ROUGE-L",             result.rouge_l),
                ]
                for label, val in detail_metrics:
                    bar_w = int(val * 100)
                    bar_c = "#34d399" if val >= 0.7 else "#f59e0b" if val >= 0.4 else "#f87171"
                    st.markdown(f"""
                    <div class="metric-row">
                      <span class="metric-name">{label}</span>
                      <span class="metric-val">{val:.4f}</span>
                    </div>
                    <div style="height:3px;background:rgba(255,255,255,0.06);border-radius:2px;margin-bottom:4px">
                      <div style="height:3px;width:{bar_w}%;background:{bar_c};border-radius:2px"></div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-row">
                  <span class="metric-name">Latency</span>
                  <span class="metric-val">{result.latency_ms}ms</span>
                </div>
                <div class="metric-row">
                  <span class="metric-name">Response words</span>
                  <span class="metric-val">{len(result.llm_response.split())}</span>
                </div>
                """, unsafe_allow_html=True)

            with cb:
                st.markdown("**Prompt Text:**")
                st.code(result.prompt_text, language=None)
                st.markdown("**LLM Response:**")
                st.markdown(f'<div class="response-box">{result.llm_response or "No response"}</div>',
                            unsafe_allow_html=True)
                if result.judge_reasoning:
                    st.markdown("**Judge Reasoning:**")
                    st.markdown(f'<div class="judge-box">💬 {result.judge_reasoning}</div>',
                                unsafe_allow_html=True)

    # ── Auto Insights ─────────────────────────────────────────────────────────
    st.markdown("### 💡 Auto Insights")
    bv = report.get("best_variant", {})
    disc = report.get("most_discriminating_metric", "semantic_similarity")

    insights = [
        ("🏆 Best Variant",
         f"**{bv.get('name','—')}** scored **{bv.get('score',0):.1f}/100**. "
         f"Key strengths: {', '.join(bv.get('key_strengths', ['—'])) or '—'}"),
        ("📊 Score Spread",
         f"Scores ranged from **{sr.get('min',0):.1f}** to **{sr.get('max',0):.1f}** "
         f"(spread = **{sr.get('spread',0):.1f}** points). "
         + ("Large spread → prompt wording has major impact." if sr.get('spread',0) > 20
            else "Small spread → variants are fairly equivalent.")),
        ("🔬 Most Discriminating Metric",
         f"**{disc}** showed the highest variance across variants — "
         "tune your prompts to improve this metric first."),
        ("⚡ Performance",
         f"Average latency: **{int(sum(r.latency_ms for r in results)/max(len(results),1))}ms** "
         f"across {len(results)} evaluations."),
    ]
    for title, body in insights:
        st.info(f"**{title}**\n\n{body}")
