import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="The Price of Discovery", layout="wide")

# ── Load data ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

@st.cache_data
def load_data():
    org = pd.read_csv(os.path.join(OUTPUT_DIR, "org_stats_reliable.csv"))
    return org

org_raw = load_data()

# Filter to valid cost-per-impact rows
org = org_raw[org_raw["cost_per_impact"].notna() & (org_raw["cost_per_impact"] > 0)].copy()
if "dominant_field" not in org.columns:
    org["dominant_field"] = "Other"

# Funding bracket
org["funding_bracket"] = pd.cut(
    org["total_funding"],
    bins=[0, 1e6, 1e7, 1e8, 1e9, np.inf],
    labels=["<$1M", "$1M–10M", "$10M–100M", "$100M–1B", ">$1B"],
)

# CMU flag
org["is_cmu"] = org["RESEARCH_ORG_CLEAN"].str.contains("CARNEGIE MELLON", case=False, na=False)
cmu = org[org["is_cmu"]]
cmu_row = cmu.iloc[0] if len(cmu) > 0 else None

# Fixed y-axis range (computed once from full data so it doesn't jump)
Y_RANGE = [np.log10(org["cost_per_impact"].min()) - 0.2, np.log10(org["cost_per_impact"].max()) + 0.2]

# ── Header ─────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; margin-bottom:0'>The Price of Discovery</h1>"
    "<p style='text-align:center; font-size:1.2em; color:gray; margin-top:0'>"
    "Does more research funding buy more scientific impact?"
    "</p>",
    unsafe_allow_html=True,
)

# ── KPI row ────────────────────────────────────────────────────────────
if cmu_row is not None:
    rank = int((org["cost_per_impact"] <= cmu_row["cost_per_impact"]).sum())
    total = len(org)
    funding_m = cmu_row["total_funding"] / 1e6
    st.markdown(
        f"""
        <div style="text-align:center; margin-bottom:0.3rem; font-size:0.8rem; color:gray; font-weight:600; letter-spacing:0.05em;">CARNEGIE MELLON UNIVERSITY</div>
        <div style="display:flex; justify-content:center; gap:2.5rem; flex-wrap:wrap; font-size:0.85rem;">
            <div style="text-align:center"><span style="color:gray">Funding</span><br><b>${funding_m:,.1f}M</b></div>
            <div style="text-align:center"><span style="color:gray">Publications</span><br><b>{cmu_row['total_publications']:,.0f}</b></div>
            <div style="text-align:center"><span style="color:gray">Avg FCR</span><br><b>{cmu_row['avg_field_citation_ratio']:.1f}</b></div>
            <div style="text-align:center"><span style="color:gray">Cost / Impact</span><br><b>${cmu_row['cost_per_impact']:,.0f}</b></div>
            <div style="text-align:center"><span style="color:gray">Efficiency Rank</span><br><b>{rank} / {total}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ── Filter (inline) ───────────────────────────────────────────────────
selected_org = "CARNEGIE MELLON UNIVERSITY"
min_pubs = st.slider("Min publications", 5, 500, 50, step=5)

# Apply filters
filtered = org[org["total_publications"] >= min_pubs].copy()
filtered["highlight"] = filtered["RESEARCH_ORG_CLEAN"] == selected_org
sel_row = filtered[filtered["highlight"]]

# ── Chart 1 & 2: side by side ─────────────────────────────────────────
col1, col2 = st.columns(2)

# ── CHART 1: Cost per Impact vs Total Funding (scatter) ────────────────
with col1:
    st.markdown("<h3 style='margin-bottom:0'>Cost per Impact vs. Total Funding</h3>", unsafe_allow_html=True)

    FIELD_COLORS = {
        "Biomedical": "#e74c3c", "Engineering & CS": "#3498db",
        "Physical Sciences": "#2ecc71", "Life Sciences": "#9b59b6",
        "Social Sciences": "#f39c12", "Defense": "#7f8c8d",
        "Agriculture": "#1abc9c", "Other": "#bdc3c7",
    }
    others = filtered[~filtered["highlight"]]
    plot_others = others.rename(columns={
        "total_funding": "Total Funding",
        "cost_per_impact": "Cost per Impact",
        "total_publications": "Total Publications",
        "dominant_field": "Field",
        "RESEARCH_ORG_CLEAN": "Institution",
    })
    fig1 = px.scatter(
        plot_others,
        x="Total Funding",
        y="Cost per Impact",
        color="Field",
        color_discrete_map=FIELD_COLORS,
        size="Total Publications",
        size_max=40,
        hover_name="Institution",
        hover_data={"Total Funding": ":$,.0f", "Cost per Impact": ":$,.0f", "Total Publications": ":,.0f", "Field": True},
        log_x=True,
        log_y=True,
        opacity=0.5,
    )

    # Trend line (log-log)
    log_x = np.log10(filtered["total_funding"].values)
    log_y = np.log10(filtered["cost_per_impact"].values)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    x_fit = np.linspace(log_x.min(), log_x.max(), 100)
    fig1.add_trace(go.Scatter(
        x=10**x_fit, y=10**(slope * x_fit + intercept),
        mode="lines", line=dict(color="rgba(255,0,0,0.5)", dash="dash", width=2),
        showlegend=False,
    ))

    # Highlighted institution
    if len(sel_row) > 0:
        sr = sel_row.iloc[0]
        fig1.add_trace(go.Scatter(
            x=sel_row["total_funding"],
            y=sel_row["cost_per_impact"],
            mode="markers+text",
            marker=dict(size=14, color="red", line=dict(width=2, color="black")),
            text=sel_row["RESEARCH_ORG_CLEAN"].str.title(),
            textposition="top center",
            textfont=dict(color="red", size=12),
            showlegend=False,
            customdata=[[sr["dominant_field"], f"${sr['total_funding']:,.0f}", f"${sr['cost_per_impact']:,.0f}", f"{sr['total_publications']:,.0f}"]],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Field = %{customdata[0]}<br>"
                "Total Funding = %{customdata[1]}<br>"
                "Cost per Impact = %{customdata[2]}<br>"
                "Total Publications = %{customdata[3]}"
                "<extra></extra>"
            ),
        ))

    fig1.update_layout(
        xaxis_title="Total Funding (USD)",
        yaxis_title="Cost per Impact ($ per FCR·Pub)",
        xaxis=dict(dtick=1),
        yaxis=dict(range=Y_RANGE, dtick=1),
        legend=dict(title="Field", font_size=10, yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(t=10, b=40),
        height=450,
    )
    st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

# ── CHART 2: Box plot by funding bracket ───────────────────────────────
with col2:
    st.markdown("<h3 style='margin-bottom:0'>Cost per Impact by Funding Bracket</h3>", unsafe_allow_html=True)

    box_data = filtered.copy()
    fig2 = px.box(
        box_data,
        x="funding_bracket",
        y="cost_per_impact",
        log_y=True,
        color_discrete_sequence=["steelblue"],
        category_orders={"funding_bracket": ["<$1M", "$1M–10M", "$10M–100M", "$100M–1B", ">$1B"]},
    )
    fig2.update_traces(boxpoints=False)

    # Highlight selected institution
    if len(sel_row) > 0:
        sr = sel_row.iloc[0]
        fig2.add_trace(go.Scatter(
            x=[sr["funding_bracket"]],
            y=[sr["cost_per_impact"]],
            mode="markers+text",
            marker=dict(size=14, color="red", line=dict(width=2, color="black")),
            text=[selected_org.title()],
            textposition="top center",
            textfont=dict(color="red", size=11),
            name=selected_org.title(),
            showlegend=False,
        ))
        fig2.add_hline(
            y=sr["cost_per_impact"],
            line_dash="dash", line_color="red", line_width=1.5,
            annotation_text=f"${sr['cost_per_impact']:,.0f}",
            annotation_position="right",
            annotation_font_color="red",
        )

    fig2.update_layout(
        xaxis_title="Total Funding Bracket",
        yaxis_title="Cost per Impact ($ per FCR·Pub)",
        yaxis=dict(range=Y_RANGE, dtick=1),
        margin=dict(t=10, b=40),
        height=450,
        showlegend=False,  # legend already on chart 1
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

# ── Takeaway ───────────────────────────────────────────────────────────
st.divider()
st.markdown("""
We consider the average relative citations per field (RCF) of each institution. We then calculate the cost per impact, or total funding divided by RCF. We realize the cost per impact does not increase linearly per funding: a 10x increase in funding leads to 1.8x increase in cost per impact. While the inefficiencies grow slower than the funding itself, larger institutions still are generally less efficient dollar-for-dollar.

Carnegie Mellon University sits above our regression line, indicating it's cost per impact is higher than expected for the amount of funding it receives. CMU's research dollars are used less efficiently (citation-wise) compared to peer institutions.
""")
