import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px
import streamlit.components.v1 as components

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NS Healthcare Access",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #1F3864;
    margin-bottom: 0.2rem;
  }
  .rq-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.54rem;
    color: #1F3864;
    margin-bottom: 0.4rem;
  }
  .subtitle {
    font-size: 1rem;
    color: #6b7280;
    margin-bottom: 1.5rem;
  }
  .rq-box {
    background: #EEF2FF;
    border-left: 4px solid #2E75B6;
    padding: 0.8rem 1.2rem;
    border-radius: 0 6px 6px 0;
    font-style: italic;
    color: #1F3864;
    margin-bottom: 1.5rem;
  }
  .metric-card {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  .metric-value {
    font-size: 1.9rem;
    font-weight: 600;
    color: #1F3864;
    line-height: 1.1;
  }
  .metric-label {
    font-size: 0.78rem;
    color: #6b7280;
    margin-top: 0.2rem;
  }
  .vuln-badge {
    display:inline-block;
    background:#FEE2E2;
    color:#991B1B;
    border-radius:4px;
    padding:2px 8px;
    font-size:0.75rem;
    font-weight:600;
  }
  .ok-badge {
    display:inline-block;
    background:#D1FAE5;
    color:#065F46;
    border-radius:4px;
    padding:2px 8px;
    font-size:0.75rem;
    font-weight:600;
  }
  .section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.35rem;
    color: #1F3864;
    border-bottom: 2px solid #2E75B6;
    padding-bottom: 0.3rem;
    margin-bottom: 1rem;
  }
  .insight-box {
    background: #F0F7FF;
    border: 1px solid #BFDBFE;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
    font-size: 0.88rem;
    color: #1e3a5f;
  }
  .warning-box {
    background: #FFFBEB;
    border: 1px solid #FCD34D;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
    font-size: 0.88rem;
    color: #78350f;
  }

  /* Reduce top page padding so content starts higher */
  header, div[data-testid="stAppViewContainer"] > div, div[data-testid="stAppViewContainer"] .main > .block-container {
    padding-top: 0 !important;
    margin-top: 0 !important;
  }
  body, html {
    padding-top: 0 !important;
    margin-top: 0 !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    scored     = pd.read_csv("cleaned_data/vulnerability_scored.csv", index_col="Rank")
    final      = pd.read_csv("cleaned_data/merged_clean.csv")
    report     = pd.read_csv("cleaned_data/report_table.csv", index_col="Rank")
    return scored, final, report

scored, final_clean, report_table = load_data()

dist_median  = final_clean["Distance_km"].median()
share_median = final_clean["pct_65_plus"].median()

vulnerable = final_clean[
    (final_clean["Distance_km"] > dist_median) &
    (final_clean["pct_65_plus"] > share_median)
]

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("<div class='main-title'>🏥 NS Healthcare Access — Elderly Vulnerability Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='rq-title'>Research Question: Which Nova Scotian communities have older populations that live far from hospitals?</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Data Sources: Nova Scotia Community Health Clusters · 2021 Census · 43 Acute Care Hospitals</div>", unsafe_allow_html=True)

# ── KPI METRICS ───────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5, col6 = st.columns(6)
kpis = [
    (col1, "54",  "Community clusters"),
    (col2, "43",  "Acute care hospitals"),
    (col3, f"{len(vulnerable)}", "Vulnerable clusters"),
    (col4, f"{final_clean['Distance_km'].median():.1f} km", "Median distance"),
    (col5, f"{final_clean['pct_65_plus'].median():.1f}%", "Median 65+ share"),
    (col6, f"{final_clean['pct_65_plus'].max():.1f}%", "Highest 65+ share"),
]
for col, val, label in kpis:
    with col:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-value'>{val}</div>
          <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ Map", "📊 Rankings", "🔍 Explore", "📈 Analysis", "💡 Insights", "📋 Data"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — INTERACTIVE MAP
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<div class='section-header'>Interactive Vulnerability Map</div>", unsafe_allow_html=True)
    st.caption("Bubble colour = vulnerability score (green → red). Bubble size = 65+ share. Hover any bubble for details.")
    try:
        with open("cleaned_data/vulnerability_map.html", "r", encoding="utf-8") as f:
            html_data = f.read()
        components.html(html_data, height=680, scrolling=False)
    except FileNotFoundError:
        st.warning("vulnerability_map.html not found. Run the analysis script first.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — RANKINGS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-header'>Community Vulnerability Rankings</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem; color:#6b7280; margin-bottom:0.75rem;'>vulnerability_score = 0.5 × normalised_distance + 0.5 × normalised_65_share</div>", unsafe_allow_html=True)

    st.markdown("**Full ranking — all 54 clusters**")
    full_rank = scored[["Cluster", "Nearest_Hospital", "Distance_km", "pct_65_plus", "vulnerability_score"]].rename(
        columns={
            "Distance_km": "Dist (km)",
            "pct_65_plus": "65+%",
            "vulnerability_score": "Score",
            "Nearest_Hospital": "Nearest Hospital"
        }
    )

    st.dataframe(
        full_rank.style
        .background_gradient(subset=["Score"], cmap="RdYlGn_r")
        .format({"Dist (km)": "{:.1f}", "65+%": "{:.1f}", "Score": "{:.3f}"}),
        use_container_width=True, height=600
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — EXPLORE / FILTER
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-header'>Explore by Filter</div>", unsafe_allow_html=True)

    min_dist = st.slider("Min distance to hospital (km)", 0, 40, 0, step=5, key="tab3_min_dist")
    min_share = st.slider("Min 65+ share (%)", 0, 40, 0, step=5, key="tab3_min_share")

    filtered = final_clean[
        (final_clean["Distance_km"] >= min_dist) &
        (final_clean["pct_65_plus"] >= min_share)
    ].copy()

    st.caption(f"Showing {len(filtered)} of 54 clusters matching the selected filters.")

    if len(filtered) == 0:
        st.warning("No clusters match the current filters. Adjust the sliders above.")
    else:
        # Distance vs 65+ share scatter (interactive hover with cluster names)
        plot_df = final_clean.copy()
        plot_df["vulnerability_score"] = plot_df["Cluster"].map(
            scored.reset_index().set_index("Cluster")["vulnerability_score"]
        )
        plot_df["Selected"] = plot_df["Cluster"].isin(filtered["Cluster"]).map({True: "Filtered", False: "Other"})
        plot_df = plot_df.sort_values("Selected")

        fig = px.scatter(
            plot_df,
            x="Distance_km",
            y="pct_65_plus",
            color="vulnerability_score",
            color_continuous_scale="RdYlGn_r",
            hover_name="Cluster",
            hover_data={
                "Distance_km": ":.1f",
                "pct_65_plus": ":.1f",
                "vulnerability_score": ":.3f",
                "Selected": False
            },
            opacity=plot_df["Selected"].map({"Filtered": 1.0, "Other": 0.25}),
            size=plot_df["Selected"].map({"Filtered": 12, "Other": 7}),
            labels={
                "Distance_km": "Distance to Nearest Hospital (km)",
                "pct_65_plus": "65+ Share (%)",
                "vulnerability_score": "Vulnerability Score"
            },
            title="Distance vs Elderly Share — filtered clusters highlighted",
            template="plotly_white"
        )

        fig.update_traces(marker=dict(line=dict(width=0.5, color="white"), symbol="circle"))
        fig.update_layout(
            plot_bgcolor="#F8FAFC",
            paper_bgcolor="#F8FAFC",
            font=dict(family="DM Sans", size=12, color="#1F3864"),
            title=dict(font=dict(size=22)),
            xaxis=dict(
                showgrid=True,
                gridcolor="#E5E7EB",
                zeroline=False,
                showline=True,
                linecolor="black",
                linewidth=1,
                tickfont=dict(size=12),
                title=dict(font=dict(size=14))
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#E5E7EB",
                zeroline=False,
                showline=True,
                linecolor="black",
                linewidth=1,
                tickfont=dict(size=12),
                title=dict(font=dict(size=14))
            ),
            margin=dict(l=50, r=50, t=80, b=50),
            coloraxis_colorbar=dict(title="Vulnerability Score"),
            legend=dict(bordercolor="black", borderwidth=1)
        )

        # Add median lines like the original Matplotlib chart
        fig.add_shape(
            type="line",
            x0=dist_median,
            x1=dist_median,
            y0=plot_df["pct_65_plus"].min(),
            y1=plot_df["pct_65_plus"].max(),
            line=dict(color="#6B7280", dash="dash")
        )
        fig.add_shape(
            type="line",
            x0=plot_df["Distance_km"].min(),
            x1=plot_df["Distance_km"].max(),
            y0=share_median,
            y1=share_median,
            line=dict(color="#6B7280", dash="dash")
        )
        fig.add_annotation(
            x=dist_median,
            y=plot_df["pct_65_plus"].min(),
            text=f"Median {dist_median:.1f} km",
            showarrow=False,
            yshift=-10,
            font=dict(size=10, color="#6B7280")
        )
        fig.add_annotation(
            x=plot_df["Distance_km"].min(),
            y=share_median,
            text=f"Median {share_median:.1f}%",
            showarrow=False,
            xshift=-10,
            font=dict(size=10, color="#6B7280")
        )

        st.plotly_chart(fig, use_container_width=True, height=700)

        st.dataframe(
            filtered[["Cluster", "Distance_km", "pct_65_plus", "Population_65_plus", "Nearest_Hospital"]]
            .sort_values("Distance_km", ascending=False)
            .rename(columns={
                "Distance_km": "Dist (km)", "pct_65_plus": "65+ %",
                "Population_65_plus": "65+ Pop", "Nearest_Hospital": "Nearest Hospital"
            })
            .style.format({"Dist (km)": "{:.1f}", "65+ %": "{:.1f}", "65+ Pop": "{:,.0f}"}),
            use_container_width=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — ANALYSIS CHARTS
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("<div class='section-header'>Analysis Charts</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        # Distance distribution histogram
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#F8FAFC")
        ax.set_facecolor("#F8FAFC")
        ax.hist(final_clean["Distance_km"], bins=15, color="#2E75B6", edgecolor="white", linewidth=0.6)
        ax.axvline(dist_median, color="#C55A11", linestyle="--", linewidth=1.5, label=f"Median: {dist_median:.1f} km")
        ax.set_xlabel("Distance to Nearest Hospital (km)")
        ax.set_ylabel("Number of clusters")
        ax.set_title("Distribution of Hospital Distances", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")
        st.pyplot(fig, use_container_width=True)

    with c2:
        # 65+ share distribution
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#F8FAFC")
        ax.set_facecolor("#F8FAFC")
        ax.hist(final_clean["pct_65_plus"], bins=15, color="#C55A11", edgecolor="white", linewidth=0.6)
        ax.axvline(share_median, color="#2E75B6", linestyle="--", linewidth=1.5, label=f"Median: {share_median:.1f}%")
        ax.set_xlabel("65+ Share (%)")
        ax.set_ylabel("Number of clusters")
        ax.set_title("Distribution of Elderly Share", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")
        st.pyplot(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        # Top 15 by vulnerability score bar chart
        top15 = scored.head(15).copy()
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#F8FAFC")
        ax.set_facecolor("#F8FAFC")
        colours = ["#991B1B" if c in vulnerable["Cluster"].values else "#2E75B6"
                   for c in top15["Cluster"]]
        bars = ax.barh(top15["Cluster"][::-1], top15["vulnerability_score"][::-1],
                       color=colours[::-1], edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Vulnerability Score")
        ax.set_title("Top 15 by Vulnerability Score", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3, linestyle="--", axis="x")
        ax.tick_params(labelsize=7)
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor="#991B1B", label="Vulnerable (◄)"),
                           Patch(facecolor="#2E75B6", label="Not flagged")]
        ax.legend(handles=legend_elements, fontsize=7, loc="lower right")
        st.pyplot(fig, use_container_width=True)

    with c4:
        # Hospital coverage — which hospitals serve the most clusters
        hosp_counts = final_clean["Nearest_Hospital"].value_counts().head(12)
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#F8FAFC")
        ax.set_facecolor("#F8FAFC")
        ax.barh(hosp_counts.index[::-1], hosp_counts.values[::-1],
                color="#1F3864", edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Number of clusters served")
        ax.set_title("Hospitals by Number of Clusters Served", fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, linestyle="--", axis="x")
        st.pyplot(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("<div class='section-header'>Key Insights</div>", unsafe_allow_html=True)

    # Compute insight statistics
    n_beyond_15  = (final_clean["Distance_km"] > 15).sum()
    n_beyond_30  = (final_clean["Distance_km"] > 30).sum()
    top1         = scored.iloc[0]
    farthest      = final_clean.loc[final_clean["Distance_km"].idxmax()]
    pop_beyond_15 = final_clean[final_clean["Distance_km"] > 15]["Population_65_plus"].sum()

    st.markdown(f"""
    <div class='insight-box'>
    <b>Executive summary</b><br>
    • <b>{len(vulnerable)} of 54 clusters</b> are classified as vulnerable (above-median on both distance and elderly share).
    • <b>{n_beyond_15} clusters</b> are more than 15 km from the nearest hospital, affecting roughly <b>{pop_beyond_15:,.0f}</b> residents aged 65+.
    • <b>{n_beyond_30} clusters</b> are more than 30 km away; the most remote is <b>{farthest['Cluster']}</b> at <b>{farthest['Distance_km']:.1f} km</b> from its nearest hospital.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**🏆 Vulnerability & high-priority clusters**")
        st.markdown(f"""
        <div class='insight-box'>
        <b>{top1['Cluster']}</b> ranks highest by vulnerability score ({top1['vulnerability_score']:.3f}) and combines significant distance from hospital
        ({top1['Distance_km']:.1f} km) with a high elderly share ({top1['pct_65_plus']:.1f}%).
        </div>
        <div class='warning-box'>
        The strongest compound-risk signal is in the communities of Chester and Area,
        Annapolis Royal, and Digby/Clare/Weymouth — all of which score highly on both
        remoteness and elderly population share.
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("**📍 Geographic access & coverage**")
        st.markdown(f"""
        <div class='insight-box'>
        Rural clusters drive the access gap: {n_beyond_15} clusters sit more than 15 km from the nearest hospital,
        signalling potential service and transport vulnerabilities for seniors.
        </div>
        <div class='insight-box'>
        {n_beyond_30} clusters are more than 30 km away, with the most isolated cluster being
        <b>{farthest['Cluster']}</b> ({farthest['Distance_km']:.1f} km to {farthest['Nearest_Hospital']}).
        </div>
        """, unsafe_allow_html=True)

    st.markdown("**📊 System-level takeaway**")
    vuln_hosps = vulnerable["Nearest_Hospital"].value_counts()
    st.markdown(f"""
    <div class='insight-box'>
    <b>Hospital coverage:</b> {vuln_hosps.index[0]} serves the most vulnerable clusters ({vuln_hosps.iloc[0]}),
    which may indicate pressure points in regional access and capacity.
    </div>
    """, unsafe_allow_html=True)

    # Correlation between distance and 65+ share
    corr = final_clean[["Distance_km", "pct_65_plus"]].corr().iloc[0, 1]
    direction = "positive" if corr > 0 else "negative"
    st.markdown(f"""
    <div class='insight-box'>
    <b>Distance–Age correlation:</b> There is a <b>{direction} correlation
    (r = {corr:.2f})</b> between distance to hospital and elderly share.
    {"Communities farther from hospitals tend to have older populations — a compounding disadvantage." if corr > 0.15 else "Distance and elderly share are largely independent dimensions, meaning each must be considered separately."}
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — RAW DATA
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown("<div class='section-header'>Full Dataset</div>", unsafe_allow_html=True)

    search = st.text_input("Search by community name", "")
    display_df = final_clean.copy()
    if search:
        display_df = display_df[display_df["Cluster"].str.contains(search, case=False, na=False)]

    display_df["Vulnerable"] = display_df["Cluster"].isin(vulnerable["Cluster"]).map(
        {True: "◄ Yes", False: ""}
    )

    st.dataframe(
        display_df[[
            "Cluster", "Distance_km", "pct_65_plus", "Population_65_plus",
            "Total_pop_est", "Nearest_Hospital", "Census_CSD", "Vulnerable"
        ]].rename(columns={
            "Distance_km": "Dist (km)",
            "pct_65_plus": "65+ %",
            "Population_65_plus": "65+ Pop",
            "Total_pop_est": "Total Pop",
            "Nearest_Hospital": "Nearest Hospital",
            "Census_CSD": "CSD",
        })
        .sort_values("Dist (km)", ascending=False)
        .style
        .format({"Dist (km)": "{:.1f}", "65+ %": "{:.1f}",
                 "65+ Pop": "{:,.0f}", "Total Pop": "{:,.0f}"}),
        use_container_width=True,
        height=500
    )

    col1, col2 = st.columns(2)
    with col1:
        csv = display_df.to_csv(index=False)
        st.download_button("⬇️ Download filtered data as CSV", csv,
                           "ns_healthcare_filtered.csv", "text/csv")
    with col2:
        st.caption(f"Showing {len(display_df)} of 54 clusters")
