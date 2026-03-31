import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ───────────────────────────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="K-Means Clustering",
    page_icon="🔵",
    layout="wide"
)

# ───────────────────────────────────────────────────────────
# CUSTOM CSS
# ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }

    section[data-testid="stSidebar"] {
        background-color: #1a1f2e;
    }

    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        border-left: 4px solid #4e9af1;
        padding-left: 12px;
        margin: 28px 0 18px 0;
    }

    .info-box {
        background: linear-gradient(135deg, #1a1f2e, #2d3561);
        border-left: 4px solid #4e9af1;
        border-radius: 10px;
        padding: 20px;
        margin: 18px 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #2d3561);
        border: 1px solid #3d4f7c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }

    .metric-label {
        color: #a0aec0;
        margin-top: 6px;
    }

    .result-box {
        border-radius: 14px;
        padding: 30px;
        text-align: center;
        margin-top: 24px;
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────
# LOAD DATA
# ───────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("Global_Mental_Health_Dataset_2025.csv")

df_raw = load_data()

# ───────────────────────────────────────────────────────────
# PREPROCESSING
# ───────────────────────────────────────────────────────────
stress_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
    "Severe": 3
}

history_map = {
    "No": 0,
    "Yes": 1
}

df_raw["Stress_Level_Num"] = df_raw["Stress_Level"].map(stress_map)
df_raw["Mental_History_Num"] = df_raw["Mental_Health_History"].map(history_map)

FEATURES = [
    "Depression_Score",
    "Anxiety_Score",
    "Sleep_Hours",
    "Stress_Level_Num",
    "Mental_History_Num"
]

df = df_raw[FEATURES].dropna().copy()
df_full = df_raw.loc[df.index].copy()

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# KMeans
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=20
)

clusters = kmeans.fit_predict(X_pca)

df_full["Cluster"] = clusters
df_full["PCA1"] = X_pca[:, 0]
df_full["PCA2"] = X_pca[:, 1]

# Determine which cluster is low/moderate/high
cluster_summary = df_full.groupby("Cluster").agg({
    "Depression_Score": "mean",
    "Anxiety_Score": "mean",
    "Sleep_Hours": "mean"
}).round(2)

cluster_summary["Severity"] = (
    cluster_summary["Depression_Score"] * 1.5 +
    cluster_summary["Anxiety_Score"] * 1.3 -
    cluster_summary["Sleep_Hours"] * 0.5
)

sorted_clusters = cluster_summary["Severity"].sort_values().index.tolist()

cluster_label_map = {
    sorted_clusters[0]: ("🟢 Low Risk", "#68d391"),
    sorted_clusters[1]: ("🟡 Moderate Risk", "#f6ad55"),
    sorted_clusters[2]: ("🔴 High Risk", "#fc8181")
}

df_full["Risk_Label"] = df_full["Cluster"].map(
    lambda x: cluster_label_map[x][0]
)

# ───────────────────────────────────────────────────────────
# TITLE
# ───────────────────────────────────────────────────────────
st.title("🔵 K-Means Clustering with PCA")
st.markdown(
    "<span style='color:#a0aec0'>Patients are grouped using Depression, Anxiety, Sleep, Stress and History.</span>",
    unsafe_allow_html=True
)

# ───────────────────────────────────────────────────────────
# ABOUT
# ───────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>📖 How It Works</div>", unsafe_allow_html=True)

st.markdown("""
<div class='info-box'>
1. Patient features are standardized.<br>
2. PCA combines 5 features into 2 stronger dimensions.<br>
3. K-Means forms 3 groups from those dimensions.<br>
4. The groups are labelled as Low, Moderate and High Risk.
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────
# CLUSTER COUNTS
# ───────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>📊 Cluster Distribution</div>", unsafe_allow_html=True)

low_count = (df_full["Risk_Label"] == "🟢 Low Risk").sum()
mod_count = (df_full["Risk_Label"] == "🟡 Moderate Risk").sum()
high_count = (df_full["Risk_Label"] == "🔴 High Risk").sum()

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value' style='color:#68d391'>{low_count}</div>
        <div class='metric-label'>Low Risk</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value' style='color:#f6ad55'>{mod_count}</div>
        <div class='metric-label'>Moderate Risk</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value' style='color:#fc8181'>{high_count}</div>
        <div class='metric-label'>High Risk</div>
    </div>
    """, unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────
# PCA CLUSTER PLOT
# ───────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>🗺️ PCA Cluster Visualization</div>", unsafe_allow_html=True)

fig = px.scatter(
    df_full,
    x="PCA1",
    y="PCA2",
    color="Risk_Label",
    hover_data=[
        "Depression_Score",
        "Anxiety_Score",
        "Sleep_Hours",
        "Stress_Level"
    ],
    color_discrete_map={
        "🟢 Low Risk": "#68d391",
        "🟡 Moderate Risk": "#f6ad55",
        "🔴 High Risk": "#fc8181"
    },
    template="plotly_dark"
)

fig.update_layout(
    paper_bgcolor="#1a1f2e",
    plot_bgcolor="#1a1f2e",
    font_color="white"
)

st.plotly_chart(fig, use_container_width=True)

# ───────────────────────────────────────────────────────────
# SUMMARY TABLE
# ───────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>📋 Cluster Summary</div>", unsafe_allow_html=True)
st.dataframe(cluster_summary, use_container_width=True)

st.markdown("<div class='section-header'>📊 Average Values by Risk Group</div>", unsafe_allow_html=True)

summary_plot = cluster_summary.reset_index()

summary_plot["Risk_Label"] = summary_plot["Cluster"].map(
    lambda x: cluster_label_map[x][0]
)

plot_df = summary_plot.melt(
    id_vars="Risk_Label",
    value_vars=["Depression_Score", "Anxiety_Score", "Sleep_Hours"],
    var_name="Feature",
    value_name="Average"
)

fig2 = px.bar(
    plot_df,
    x="Risk_Label",
    y="Average",
    color="Feature",
    barmode="group",
    template="plotly_dark"
)

fig2.update_layout(
    paper_bgcolor="#1a1f2e",
    plot_bgcolor="#1a1f2e",
    font_color="white"
)

st.plotly_chart(fig2, use_container_width=True)

st.markdown("<div class='section-header'>😴 Sleep vs Depression</div>", unsafe_allow_html=True)

fig3 = px.scatter(
    df_full,
    x="Sleep_Hours",
    y="Depression_Score",
    color="Risk_Label",
    color_discrete_map={
        "🟢 Low Risk": "#68d391",
        "🟡 Moderate Risk": "#f6ad55",
        "🔴 High Risk": "#fc8181"
    },
    template="plotly_dark"
)

fig3.update_layout(
    paper_bgcolor="#1a1f2e",
    plot_bgcolor="#1a1f2e",
    font_color="white"
)

st.plotly_chart(fig3, use_container_width=True)

# ───────────────────────────────────────────────────────────
# PREDICTOR
# ───────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>🎯 Individual Patient Predictor</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    p_dep = st.number_input("😔 Depression Score", 0, 30, 10)
    p_sleep = st.number_input("😴 Sleep Hours", 0.0, 12.0, 7.0)

with col2:
    p_anx = st.number_input("😰 Anxiety Score", 0, 30, 8)
    p_stress = st.selectbox(
        "⚠️ Stress Level",
        ["Low", "Medium", "High", "Severe"]
    )

with col3:
    p_history = st.selectbox(
        "🧠 Mental Health History",
        ["No", "Yes"]
    )

    predict_btn = st.button("🔍 Determine Cluster", use_container_width=True)

# ───────────────────────────────────────────────────────────
# PREDICTION
# ───────────────────────────────────────────────────────────
if predict_btn:
    input_data = np.array([[
        p_dep,
        p_anx,
        p_sleep,
        stress_map[p_stress],
        history_map[p_history]
    ]])

    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)
    predicted_cluster = kmeans.predict(input_pca)[0]

    label, color = cluster_label_map[predicted_cluster]

    st.markdown(f"""
<div class='result-box'
style='background:linear-gradient(135deg,#1a1f2e,#2d3561);
border:2px solid {color};'>

<div style='font-size:3rem'>{label.split()[0]}</div>

<div style='font-size:2rem; font-weight:700; color:{color}; margin:12px 0'>
{" ".join(label.split()[1:])}
</div>

<div style='color:#cbd5e0; font-size:1rem; line-height:2'>
Depression Score: <b>{p_dep}</b> &nbsp;|&nbsp;
Anxiety Score: <b>{p_anx}</b> &nbsp;|&nbsp;
Sleep Hours: <b>{p_sleep}</b><br>
Stress Level: <b>{p_stress}</b> &nbsp;|&nbsp;
Mental Health History: <b>{p_history}</b>
</div>

</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────
# FINAL INSIGHT
# ───────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>💡 Key Insight</div>", unsafe_allow_html=True)

st.markdown("""
<div class='info-box'>
Patients in the High Risk cluster generally have:
<ul>
<li>High depression score</li>
<li>High anxiety score</li>
<li>Very low sleep hours</li>
<li>High stress level</li>
<li>Previous mental health history</li>
</ul>
</div>
""", unsafe_allow_html=True)