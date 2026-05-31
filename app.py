"""
Software License Utilization Analytics Dashboard
Vendors: Petrel & Landmark
Author: Analytics Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import re

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="License Utilization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THEME / STYLES
# ─────────────────────────────────────────────
COLORS = {
    "Petrel":   "#4C9BE8",
    "Landmark": "#F4845F",
    "accent":   "#6C63FF",
    "good":     "#2ECC71",
    "warn":     "#F39C12",
    "danger":   "#E74C3C",
    "bg":       "#0E1117",
    "card":     "#1C2333",
}

st.markdown("""
<style>
/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0E1117;
    color: #E8EAF0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background: #161B27;
    border-right: 1px solid #2A3244;
}

/* ── KPI Cards ── */
.kpi-row { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 160px;
    background: linear-gradient(135deg, #1C2333 60%, #232B40);
    border: 1px solid #2A3244;
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 4px 20px rgba(0,0,0,.4);
}
.kpi-label {
    font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1.2px;
    color: #8892A4; margin-bottom: 8px;
}
.kpi-value {
    font-size: 34px; font-weight: 800;
    color: #E8EAF0; line-height: 1;
}
.kpi-sub  { font-size: 12px; color: #5A6478; margin-top: 6px; }
.kpi-good { color: #2ECC71; }
.kpi-warn { color: #F39C12; }
.kpi-danger { color: #E74C3C; }

/* ── Section headers ── */
.section-title {
    font-size: 18px; font-weight: 700;
    color: #C8D0E0; margin: 30px 0 14px;
    padding-left: 10px;
    border-left: 4px solid #4C9BE8;
}

/* ── Divider ── */
.divider { height: 1px; background: #2A3244; margin: 28px 0; }

/* ── Pill badge ── */
.badge {
    display: inline-block;
    padding: 3px 10px; border-radius: 99px;
    font-size: 11px; font-weight: 600;
}
.badge-petrel   { background: rgba(76,155,232,.18); color: #4C9BE8; }
.badge-landmark { background: rgba(244,132,95,.18);  color: #F4845F; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────

def extract_numeric(value: str) -> float:
    """Extract the first integer from strings like '171 (171.00)' → 171.0"""
    if pd.isna(value):
        return np.nan
    match = re.match(r"[\s\"]*(\d+)", str(value))
    return float(match.group(1)) if match else np.nan


def load_csv(filepath: str, vendor: str) -> pd.DataFrame:
    """Load a vendor workstation CSV, skip metadata rows, parse columns."""
    # Find the header row (contains "Feature")
    header_row = None
    with open(filepath, "r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            if "Feature" in line and "Date" in line:
                header_row = i
                break

    if header_row is None:
        raise ValueError(f"Could not find header in {filepath}")

    df = pd.read_csv(filepath, skiprows=header_row, quotechar='"')

    # Standardise column names
    df.columns = [c.strip().strip('"') for c in df.columns]

    # Keep only needed columns
    col_map = {}
    for col in df.columns:
        lc = col.lower()
        if "feature" in lc:
            col_map[col] = "Feature"
        elif "date" in lc or "bucket" in lc:
            col_map[col] = "Date"
        elif "avg used" in lc:
            col_map[col] = "Avg_Used_Raw"
        elif "avg total" in lc:
            col_map[col] = "Avg_Total_Raw"
        elif col.lower() == "max total":
            col_map[col] = "Max_Total_Raw"

    df = df.rename(columns=col_map)
    needed = ["Feature", "Date", "Avg_Used_Raw", "Avg_Total_Raw", "Max_Total_Raw"]
    df = df[[c for c in needed if c in df.columns]].copy()

    # Drop rows where Feature is NaN (footer junk)
    df.dropna(subset=["Feature"], inplace=True)
    df = df[df["Feature"].astype(str).str.strip() != ""]

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip().str.strip('"'), errors="coerce")
    df.dropna(subset=["Date"], inplace=True)

    # Extract numeric values
    df["Avg_Used"]  = df["Avg_Used_Raw"].apply(extract_numeric)
    df["Avg_Total"] = df["Avg_Total_Raw"].apply(extract_numeric)
    df["Max_Total"] = df["Max_Total_Raw"].apply(extract_numeric) if "Max_Total_Raw" in df.columns else np.nan

    # Derived metrics
    df["Utilization_Pct"] = np.where(
        df["Avg_Total"] > 0,
        (df["Avg_Used"] / df["Avg_Total"] * 100).round(2),
        np.nan,
    )

    df["Vendor"] = vendor
    df["Month"]  = df["Date"].dt.to_period("M").astype(str)
    df["Feature"] = df["Feature"].astype(str).str.strip().str.strip('"')

    return df[[
        "Vendor", "Feature", "Date", "Month",
        "Avg_Used", "Avg_Total", "Max_Total", "Utilization_Pct"
    ]]


@st.cache_data(show_spinner="Loading data…")
def load_all_data(petrel_path: str, landmark_path: str) -> pd.DataFrame:
    dfs = []
    for path, vendor in [(petrel_path, "Petrel"), (landmark_path, "Landmark")]:
        if os.path.exists(path):
            dfs.append(load_csv(path, vendor))
    if not dfs:
        st.error("No data files found!")
        st.stop()
    return pd.concat(dfs, ignore_index=True)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
BASE = os.path.dirname(__file__)
PETREL_FILE   = os.path.join(BASE, "workstation_report_Petrel_2025-10-01_to_2026-03-31.csv")
LANDMARK_FILE = os.path.join(BASE, "workstation_report_Landmark_2025-10-01_to_2026-03-31.csv")

df_raw = load_all_data(PETREL_FILE, LANDMARK_FILE)

# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 Filters")
    st.markdown("---")

    # Vendor
    vendors = sorted(df_raw["Vendor"].unique())
    sel_vendors = st.multiselect("Vendor", vendors, default=vendors)

    # Date range
    min_date = df_raw["Date"].min().date()
    max_date = df_raw["Date"].max().date()
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Feature search
    all_features = sorted(df_raw["Feature"].unique())
    feature_search = st.text_input("🔍 Search Feature", "")
    filtered_features_list = [f for f in all_features if feature_search.lower() in f.lower()]
    sel_features = st.multiselect("Feature", filtered_features_list, default=[])

    # Utilization threshold
    util_threshold = st.slider(
        "Underutilized threshold (%)", 0, 50, 20, step=5,
        help="Features below this utilisation are flagged as underutilised"
    )

    st.markdown("---")
    st.markdown("### 📁 Data Files")
    st.caption(f"**Petrel:** {os.path.basename(PETREL_FILE)}")
    st.caption(f"**Landmark:** {os.path.basename(LANDMARK_FILE)}")
    st.markdown(f"**Total rows:** `{len(df_raw):,}`")

# ─────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────
df = df_raw.copy()
if sel_vendors:
    df = df[df["Vendor"].isin(sel_vendors)]

if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    df = df[(df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)]

if sel_features:
    df = df[df["Feature"].isin(sel_features)]

if df.empty:
    st.warning("No data matches the selected filters. Please adjust the sidebar filters.")
    st.stop()

# ─────────────────────────────────────────────
# COMPUTED AGGREGATES
# ─────────────────────────────────────────────
feature_agg = (
    df.groupby(["Vendor", "Feature"])
    .agg(
        Avg_Used    = ("Avg_Used",        "mean"),
        Avg_Total   = ("Avg_Total",       "mean"),
        Max_Total   = ("Max_Total",       "max"),
        Util_Pct    = ("Utilization_Pct", "mean"),
        Days        = ("Date",            "count"),
    )
    .reset_index()
)
feature_agg["Util_Pct"] = np.where(
    feature_agg["Avg_Total"] > 0,
    (feature_agg["Avg_Used"] / feature_agg["Avg_Total"] * 100).round(2),
    np.nan,
)

# ─────────────────────────────────────────────
# KPI METRICS
# ─────────────────────────────────────────────
avg_util      = df["Utilization_Pct"].mean()
total_lic     = feature_agg["Avg_Total"].sum()
peak_usage    = df["Avg_Used"].max()
under_count   = (feature_agg["Util_Pct"] < util_threshold).sum()
total_features = feature_agg["Feature"].nunique()

def util_color(val):
    if val >= 70: return "kpi-good"
    if val >= 30: return "kpi-warn"
    return "kpi-danger"

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:24px;">
  <h1 style="font-size:28px;font-weight:900;color:#E8EAF0;margin:0;">
    📊 Software License Utilization Dashboard
  </h1>
  <p style="color:#5A6478;font-size:14px;margin-top:4px;">
    Petrel &amp; Landmark · Oct 2025 – Mar 2026 · Interactive Analytics
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
uc = util_color(avg_util)
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric("⚡ Avg Utilization", f"{avg_util:.1f}%",
              help="Average utilization across all features & dates in view")
with k2:
    st.metric("🔑 Total Licenses", f"{int(total_lic):,}",
              help="Sum of average available licenses across all features")
with k3:
    st.metric("📈 Peak Usage", f"{int(peak_usage):,}",
              help="Maximum single-day avg-used value observed")
with k4:
    st.metric("⚠️ Underutilized", f"{under_count}",
              help=f"Features with avg utilization < {util_threshold}%")
with k5:
    st.metric("🧩 Features", f"{total_features}",
              help="Distinct license features in current selection")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ROW 1: Line Chart + Donut
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">📅 Usage Trends & Vendor Distribution</div>', unsafe_allow_html=True)
col_line, col_donut = st.columns([3, 1])

with col_line:
    daily = (
        df.groupby(["Date", "Vendor"])["Avg_Used"]
        .sum().reset_index()
    )
    fig_line = px.line(
        daily, x="Date", y="Avg_Used", color="Vendor",
        title="License Usage Trend Over Time",
        labels={"Avg_Used": "Total Avg Licenses Used", "Date": ""},
        color_discrete_map={"Petrel": COLORS["Petrel"], "Landmark": COLORS["Landmark"]},
        template="plotly_dark",
    )
    fig_line.update_traces(line_width=2.2, mode="lines")
    fig_line.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#C8D0E0"),
        legend=dict(orientation="h", y=-0.15),
        hovermode="x unified",
        margin=dict(t=40, b=10, l=0, r=0),
        title_font_size=14,
        xaxis=dict(gridcolor="#2A3244"),
        yaxis=dict(gridcolor="#2A3244"),
    )
    st.plotly_chart(fig_line, use_container_width=True)

with col_donut:
    vendor_totals = (
        df.groupby("Vendor")["Avg_Used"].sum().reset_index()
    )
    fig_donut = go.Figure(go.Pie(
        labels=vendor_totals["Vendor"],
        values=vendor_totals["Avg_Used"],
        hole=0.58,
        marker_colors=[COLORS["Petrel"], COLORS["Landmark"]],
        textinfo="label+percent",
        textfont_size=12,
        hovertemplate="%{label}<br>Avg Used: %{value:,.0f}<extra></extra>",
    ))
    fig_donut.update_layout(
        title="Vendor Distribution",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#C8D0E0"),
        showlegend=False,
        margin=dict(t=40, b=10, l=10, r=10),
        title_font_size=14,
        annotations=[dict(
            text="Usage<br>Share",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color="#8892A4"),
        )],
    )
    st.plotly_chart(fig_donut, use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ROW 2: Top Features Bar + Utilization Histogram
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">🏆 Feature Usage & Utilization Distribution</div>', unsafe_allow_html=True)
col_bar, col_hist = st.columns([3, 2])

with col_bar:
    top_n = 15
    top_feat = (
        feature_agg.nlargest(top_n, "Avg_Used")
        .sort_values("Avg_Used")
    )
    fig_bar = px.bar(
        top_feat, x="Avg_Used", y="Feature", color="Vendor",
        orientation="h",
        title=f"Top {top_n} Features by Average Usage",
        labels={"Avg_Used": "Avg Licenses Used", "Feature": ""},
        color_discrete_map={"Petrel": COLORS["Petrel"], "Landmark": COLORS["Landmark"]},
        template="plotly_dark",
        text="Avg_Used",
    )
    fig_bar.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#C8D0E0"),
        legend=dict(orientation="h", y=-0.12),
        margin=dict(t=40, b=10, l=0, r=60),
        title_font_size=14,
        xaxis=dict(gridcolor="#2A3244"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        bargap=0.25,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_hist:
    fig_hist = px.histogram(
        feature_agg.dropna(subset=["Util_Pct"]),
        x="Util_Pct",
        color="Vendor",
        nbins=20,
        title="Utilization % Distribution",
        labels={"Util_Pct": "Utilization (%)", "count": "# Features"},
        color_discrete_map={"Petrel": COLORS["Petrel"], "Landmark": COLORS["Landmark"]},
        template="plotly_dark",
        barmode="overlay",
        opacity=0.75,
    )
    fig_hist.add_vline(
        x=util_threshold, line_dash="dash", line_color="#E74C3C",
        annotation_text=f"Threshold {util_threshold}%",
        annotation_font_color="#E74C3C",
    )
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#C8D0E0"),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=40, b=10, l=0, r=0),
        title_font_size=14,
        xaxis=dict(gridcolor="#2A3244"),
        yaxis=dict(gridcolor="#2A3244"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ROW 3: Monthly Trend Heatmap
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">🗓️ Monthly Utilization Heatmap</div>', unsafe_allow_html=True)

monthly_feat = (
    df.groupby(["Month", "Vendor"])["Utilization_Pct"]
    .mean().reset_index()
    .pivot(index="Vendor", columns="Month", values="Utilization_Pct")
    .round(1)
)

fig_heat = go.Figure(go.Heatmap(
    z=monthly_feat.values,
    x=monthly_feat.columns.tolist(),
    y=monthly_feat.index.tolist(),
    colorscale=[
        [0.0,  "#1C2333"],
        [0.2,  "#1A3A5C"],
        [0.5,  "#2F7FC1"],
        [0.75, "#2ECC71"],
        [1.0,  "#F39C12"],
    ],
    text=monthly_feat.values,
    texttemplate="%{text:.1f}%",
    hovertemplate="Vendor: %{y}<br>Month: %{x}<br>Avg Util: %{z:.1f}%<extra></extra>",
    showscale=True,
    colorbar=dict(title="Util %", tickfont=dict(color="#C8D0E0")),
))
fig_heat.update_layout(
    title="Monthly Avg Utilization % by Vendor",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#C8D0E0"),
    margin=dict(t=50, b=20, l=80, r=20),
    title_font_size=14,
    height=200,
)
st.plotly_chart(fig_heat, use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ROW 4: Underutilized Features Table + Scatter
# ─────────────────────────────────────────────
st.markdown(f'<div class="section-title">⚠️ Underutilized Features  (< {util_threshold}% utilization)</div>', unsafe_allow_html=True)

col_tbl, col_scatter = st.columns([2, 3])

with col_tbl:
    under = (
        feature_agg[feature_agg["Util_Pct"] < util_threshold]
        .sort_values("Util_Pct")
        [["Vendor", "Feature", "Util_Pct", "Avg_Used", "Avg_Total"]]
        .rename(columns={
            "Util_Pct": "Util %",
            "Avg_Used": "Avg Used",
            "Avg_Total": "Avg Total",
        })
    )
    under["Util %"]   = under["Util %"].map(lambda x: f"{x:.1f}%")
    under["Avg Used"] = under["Avg Used"].map(lambda x: f"{x:.1f}")
    under["Avg Total"]= under["Avg Total"].map(lambda x: f"{x:.0f}")

    st.dataframe(
        under,
        use_container_width=True,
        hide_index=True,
        height=380,
    )
    st.caption(f"💡 {len(under)} features are below the {util_threshold}% threshold — potential cost savings.")

with col_scatter:
    fig_scatter = px.scatter(
        feature_agg.dropna(subset=["Util_Pct", "Avg_Total"]),
        x="Avg_Total", y="Util_Pct", color="Vendor",
        size="Avg_Used", size_max=28,
        hover_name="Feature",
        title="License Pool Size vs Utilization %",
        labels={
            "Avg_Total": "Avg Total Licenses",
            "Util_Pct":  "Utilization (%)",
        },
        color_discrete_map={"Petrel": COLORS["Petrel"], "Landmark": COLORS["Landmark"]},
        template="plotly_dark",
        opacity=0.75,
    )
    fig_scatter.add_hline(
        y=util_threshold, line_dash="dash", line_color="#E74C3C",
        annotation_text=f"Underutil threshold {util_threshold}%",
        annotation_font_color="#E74C3C",
    )
    fig_scatter.add_hline(
        y=80, line_dash="dot", line_color="#2ECC71",
        annotation_text="80% target",
        annotation_font_color="#2ECC71",
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#C8D0E0"),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=40, b=10, l=0, r=0),
        title_font_size=14,
        xaxis=dict(gridcolor="#2A3244"),
        yaxis=dict(gridcolor="#2A3244"),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ROW 5: Monthly feature trend + Top underused waste
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">📉 Monthly Breakdown & Cost Opportunity</div>', unsafe_allow_html=True)
col_m1, col_m2 = st.columns(2)

with col_m1:
    monthly_vendor = (
        df.groupby(["Month", "Vendor"])
        .agg(Avg_Used=("Avg_Used", "sum"), Avg_Total=("Avg_Total", "sum"))
        .reset_index()
    )
    monthly_vendor["Util_Pct"] = (
        monthly_vendor["Avg_Used"] / monthly_vendor["Avg_Total"] * 100
    ).round(2)

    fig_mbar = px.bar(
        monthly_vendor, x="Month", y="Util_Pct", color="Vendor",
        barmode="group",
        title="Monthly Avg Utilization % by Vendor",
        labels={"Util_Pct": "Utilization (%)", "Month": ""},
        color_discrete_map={"Petrel": COLORS["Petrel"], "Landmark": COLORS["Landmark"]},
        template="plotly_dark",
    )
    fig_mbar.add_hline(y=80, line_dash="dot", line_color="#2ECC71",
                       annotation_text="80% target", annotation_font_color="#2ECC71")
    fig_mbar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#C8D0E0"),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=40, b=10, l=0, r=0),
        title_font_size=14,
        xaxis=dict(gridcolor="#2A3244", tickangle=-30),
        yaxis=dict(gridcolor="#2A3244"),
    )
    st.plotly_chart(fig_mbar, use_container_width=True)

with col_m2:
    # "Waste" = unused licenses (Avg_Total - Avg_Used) for under-used features
    waste = feature_agg.copy()
    waste["Unused"] = (waste["Avg_Total"] - waste["Avg_Used"]).clip(lower=0)
    top_waste = waste.nlargest(12, "Unused")[["Vendor", "Feature", "Unused", "Util_Pct"]].sort_values("Unused")

    fig_waste = px.bar(
        top_waste, x="Unused", y="Feature", color="Vendor",
        orientation="h",
        title="Top Features by Unused Licenses (Cost Opportunity)",
        labels={"Unused": "Avg Unused Licenses", "Feature": ""},
        color_discrete_map={"Petrel": COLORS["Petrel"], "Landmark": COLORS["Landmark"]},
        template="plotly_dark",
        text="Unused",
    )
    fig_waste.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig_waste.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#C8D0E0"),
        legend=dict(orientation="h", y=-0.18),
        margin=dict(t=40, b=10, l=0, r=60),
        title_font_size=14,
        xaxis=dict(gridcolor="#2A3244"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        bargap=0.25,
    )
    st.plotly_chart(fig_waste, use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RAW DATA EXPLORER
# ─────────────────────────────────────────────
with st.expander("🔍 Raw Data Explorer", expanded=False):
    show_df = df.copy()
    show_df["Date"] = show_df["Date"].dt.strftime("%Y-%m-%d")
    show_df["Utilization_Pct"] = show_df["Utilization_Pct"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "–")
    st.dataframe(show_df, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(show_df):,} rows after filters.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#3A4255;font-size:12px;margin-top:40px;padding:20px 0;">
  License Utilization Dashboard · Petrel &amp; Landmark · Built with Streamlit &amp; Plotly
</div>
""", unsafe_allow_html=True)
