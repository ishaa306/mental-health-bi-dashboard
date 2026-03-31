
import streamlit as st
import pandas as pd
import plotly.express as px

# ───────────────────────────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Algorithm Comparison",
    page_icon="⚖️",
    layout="wide"
)

# ───────────────────────────────────────────────────────────
# CSS
# ───────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}

.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    border-left: 4px solid #4e9af1;
    padding-left: 12px;
    margin: 30px 0 18px 0;
}

.metric-card {
    background: linear-gradient(135deg,#1a1f2e,#2d3561);
    border: 1px solid #3d4f7c;
    border-radius: 14px;
    padding: 24px;
    text-align: center;
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #63b3ed;
}

.metric-label {
    margin-top: 10px;
    color: #cbd5e0;
    font-size: 1rem;
}

.summary-box {
    background: linear-gradient(135deg,#1a1f2e,#2d3561);
    border-left: 5px solid #48bb78;
    border-radius: 12px;
    padding: 20px;
    margin-top: 16px;
}
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────
# TITLE
# ───────────────────────────────────────────────────────────
st.title("⚖️ Algorithm Comparison")
st.markdown(
    "Comparing the performance of clustering, classification and association rule mining on the mental health dataset."
)

# ───────────────────────────────────────────────────────────
# FINAL VALUES FROM PREVIOUS PAGES
# ───────────────────────────────────────────────────────────
clustering_score = 0.41
classification_accuracy = 61.2
association_lift = 1.114

# ───────────────────────────────────────────────────────────
# TOP METRIC CARDS
# ───────────────────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>🏆 Best Result from Each Technique</div>",
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{clustering_score:.2f}</div>
        <div class='metric-label'>Best K-Means Silhouette Score</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{classification_accuracy:.1f}%</div>
        <div class='metric-label'>Best Classification Accuracy (KNN)</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{association_lift:.3f}</div>
        <div class='metric-label'>Strongest Association Rule Lift</div>
    </div>
    """, unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────
# PERFORMANCE COMPARISON CHART
# ───────────────────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>📊 Technique Performance Comparison</div>",
    unsafe_allow_html=True
)

comparison_df = pd.DataFrame({
    "Technique": [
        "Clustering (K-Means)",
        "Classification (KNN)",
        "Association Rules"
    ],
    "Score": [
        clustering_score * 100,
        classification_accuracy,
        association_lift * 50
    ],
    "Displayed As": [
        "41.0%",
        "61.2%",
        "1.114 Lift"
    ]
})

fig = px.bar(
    comparison_df,
    x="Technique",
    y="Score",
    color="Technique",
    text="Displayed As",
    template="plotly_dark"
)

fig.update_traces(textposition="outside")

fig.update_layout(
    paper_bgcolor="#1a1f2e",
    plot_bgcolor="#1a1f2e",
    font_color="white",
    showlegend=False,
    yaxis_title="Performance Score",
    xaxis_title=""
)

st.plotly_chart(fig, use_container_width=True)

# ───────────────────────────────────────────────────────────
# DETAILED COMPARISON TABLE
# ───────────────────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>📋 Final Technique Summary</div>",
    unsafe_allow_html=True
)

summary_df = pd.DataFrame({
    "Technique": [
        "Clustering",
        "Classification",
        "Association Rules"
    ],
    "Best Algorithm": [
        "K-Means",
        "KNN",
        "Apriori"
    ],
    "Main Result": [
        "4 patient groups discovered",
        "61.2% accuracy",
        "High Anxiety + Low Sleep → High Risk"
    ],
    "Important Factors": [
        "Sleep, anxiety, depression",
        "Days of treatment, sleep, age",
        "Low sleep, anxiety, low activity"
    ]
})

st.dataframe(summary_df, use_container_width=True)

# ───────────────────────────────────────────────────────────
# FINAL INSIGHT
# ───────────────────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>💡 Final Conclusion</div>",
    unsafe_allow_html=True
)

st.markdown("""
<div class='summary-box'>
<b>Classification using KNN performed best overall</b> with approximately 61.2% accuracy.
<br><br>
However, the most valuable insight is that the same factors repeatedly appeared in every technique:
<ul>
<li>Low sleep hours</li>
<li>High anxiety and depression</li>
<li>Long treatment duration</li>
<li>Low physical activity</li>
</ul>
These patterns consistently indicated higher mental health risk in the dataset.
</div>
""", unsafe_allow_html=True)

