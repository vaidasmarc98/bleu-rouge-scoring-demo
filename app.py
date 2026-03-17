"""
app.py
Streamlit dashboard for visualizing LLM evaluation results.
Run with: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from files3.scorer import run_scoring

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LLM Eval Dashboard",
    page_icon="🧪",
    layout="wide",
)

# ── styles ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .pass { color: #28a745; font-weight: bold; }
    .fail { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── header ────────────────────────────────────────────────────────────────────

st.title("🧪 LLM Evaluation Dashboard")
st.markdown("BLEU & ROUGE scoring of Claude responses against reference answers.")
st.divider()

# ── run / cache scoring ───────────────────────────────────────────────────────

@st.cache_data(show_spinner="Querying Claude and scoring responses...")
def get_results():
    return run_scoring()


if st.button("▶ Run Evaluation", type="primary"):
    st.cache_data.clear()

results = get_results()
df = pd.DataFrame(results)

# ── summary metrics ───────────────────────────────────────────────────────────

st.subheader("Summary")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    pass_rate = df["passed"].mean()
    st.metric("Pass Rate", f"{pass_rate:.0%}")

with col2:
    st.metric("Avg BLEU", f"{df['bleu'].mean():.1f}")

with col3:
    st.metric("Avg ROUGE-1", f"{df['rouge1'].mean():.3f}")

with col4:
    st.metric("Avg ROUGE-2", f"{df['rouge2'].mean():.3f}")

with col5:
    st.metric("Avg ROUGE-L", f"{df['rougeL'].mean():.3f}")

st.divider()

# ── charts ────────────────────────────────────────────────────────────────────

st.subheader("Score Distributions")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig_bleu = px.histogram(
        df,
        x="bleu",
        nbins=10,
        title="BLEU Score Distribution",
        color_discrete_sequence=["#4C78A8"],
        labels={"bleu": "BLEU Score (0–100)"},
    )
    fig_bleu.add_vline(
        x=15.0, line_dash="dash", line_color="red",
        annotation_text="Pass threshold (15)",
        annotation_position="top right",
    )
    fig_bleu.update_layout(showlegend=False)
    st.plotly_chart(fig_bleu, use_container_width=True)

with chart_col2:
    rouge_df = df[["id", "rouge1", "rouge2", "rougeL"]].melt(
        id_vars="id", var_name="Metric", value_name="Score"
    )
    fig_rouge = px.box(
        rouge_df,
        x="Metric",
        y="Score",
        title="ROUGE Score Distribution",
        color="Metric",
        color_discrete_map={
            "rouge1": "#F58518",
            "rouge2": "#E45756",
            "rougeL": "#72B7B2",
        },
    )
    fig_rouge.update_layout(showlegend=False)
    st.plotly_chart(fig_rouge, use_container_width=True)

# ── scores by category ────────────────────────────────────────────────────────

st.subheader("Performance by Category")

cat_df = df.groupby("category").agg(
    avg_bleu=("bleu", "mean"),
    avg_rouge1=("rouge1", "mean"),
    avg_rouge2=("rouge2", "mean"),
    avg_rougeL=("rougeL", "mean"),
    pass_rate=("passed", "mean"),
    count=("id", "count"),
).reset_index()

fig_cat = px.bar(
    cat_df,
    x="category",
    y=["avg_rouge1", "avg_rouge2", "avg_rougeL"],
    title="Avg ROUGE Scores by Category",
    barmode="group",
    labels={"value": "Score", "variable": "Metric"},
    color_discrete_map={
        "avg_rouge1": "#F58518",
        "avg_rouge2": "#E45756",
        "avg_rougeL": "#72B7B2",
    },
)
st.plotly_chart(fig_cat, use_container_width=True)

# ── radar chart ───────────────────────────────────────────────────────────────

st.subheader("Metric Radar by Category")

radar_fig = go.Figure()

for _, row in cat_df.iterrows():
    radar_fig.add_trace(go.Scatterpolar(
        r=[row["avg_rouge1"], row["avg_rouge2"], row["avg_rougeL"], row["avg_bleu"] / 100],
        theta=["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU (norm)"],
        fill="toself",
        name=row["category"],
    ))

radar_fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    title="Normalized Score Radar by Category",
)
st.plotly_chart(radar_fig, use_container_width=True)

st.divider()

# ── per-sample results table ──────────────────────────────────────────────────

st.subheader("Per-Sample Results")

display_df = df[["id", "category", "prompt", "bleu", "rouge1", "rouge2", "rougeL", "passed"]].copy()
display_df["passed"] = display_df["passed"].map({True: "✓ PASS", False: "✗ FAIL"})
display_df.columns = ["ID", "Category", "Prompt", "BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Result"]

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "BLEU": st.column_config.NumberColumn(format="%.1f"),
        "ROUGE-1": st.column_config.NumberColumn(format="%.3f"),
        "ROUGE-2": st.column_config.NumberColumn(format="%.3f"),
        "ROUGE-L": st.column_config.NumberColumn(format="%.3f"),
    },
)

# ── response inspector ────────────────────────────────────────────────────────

st.divider()
st.subheader("Response Inspector")

selected_id = st.selectbox(
    "Select a sample to inspect",
    options=df["id"].tolist(),
    format_func=lambda x: f"{x} — {df[df['id'] == x]['prompt'].values[0][:60]}...",
)

if selected_id:
    row = df[df["id"] == selected_id].iloc[0]
    ins_col1, ins_col2 = st.columns(2)

    with ins_col1:
        st.markdown("**Prompt**")
        st.info(row["prompt"])
        st.markdown("**Reference Answer**")
        st.success(row["reference"])

    with ins_col2:
        st.markdown("**Model Response**")
        st.warning(row["model_response"])
        st.markdown("**Scores**")
        score_data = {
            "Metric": ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L"],
            "Score": [row["bleu"], row["rouge1"], row["rouge2"], row["rougeL"]],
            "Threshold": [15.0, 0.35, 0.15, 0.30],
        }
        score_df = pd.DataFrame(score_data)
        score_df["Pass"] = score_df.apply(
            lambda r: "✓" if r["Score"] >= r["Threshold"] else "✗", axis=1
        )
        st.dataframe(score_df, hide_index=True, use_container_width=True)
