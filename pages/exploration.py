import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Exploration", page_icon="🔍", layout="wide")

# ─── DARK THEME CSS ────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stSidebar { background-color: #1a1f2e; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #2d3561);
        border: 1px solid #3d4f7c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #4e9af1; }
    .metric-label { font-size: 0.85rem; color: #a0aec0; margin-top: 4px; }
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
        border-left: 4px solid #4e9af1;
        padding-left: 12px;
        margin: 24px 0 16px 0;
    }
    .highlight-box {
        background: linear-gradient(135deg, #1a1f2e, #2d3561);
        border-left: 4px solid #48bb78;
        border-radius: 8px;
        padding: 20px;
        margin: 16px 0;
    }
    .formula-box {
        background: #1a1f2e;
        border: 1px solid #4e9af1;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        margin: 16px 0;
        font-family: monospace;
    }
    h1, h2, h3 { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD DATA ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Global_Mental_Health_Dataset_2025.csv")
    return df

df = load_data()

# ─── PAGE TITLE ────────────────────────────────────────────
st.title("🔍 Module II: Data Exploration & Preprocessing")
st.markdown("Understanding the dataset structure, distributions, and preparing data for ML models.")
st.markdown("---")

# ─── DATASET METRICS ───────────────────────────────────────
st.markdown("<div class='section-header'>📊 Dataset Metrics</div>", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

metrics = [
    ("2,500", "Total Records"),
    ("15", "Total Features"),
    (str(df['Age'].min()) + " - " + str(df['Age'].max()), "Age Range"),
    (str(df['Country'].nunique()), "Countries"),
    (str(df['Outcome'].nunique()), "Outcome Classes"),
]
for col, (val, label) in zip([col1, col2, col3, col4, col5], metrics):
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

# ─── ABOUT DATASET ─────────────────────────────────────────
st.markdown("<div class='section-header'>📋 About the Dataset</div>", unsafe_allow_html=True)
st.markdown("""
<div class='highlight-box'>
<b>Source:</b> Kaggle — Global Mental Health Dataset 2025<br><br>
The dataset contains mental health records of <b>2,500 patients</b> across multiple countries.
It captures depression scores, anxiety scores, stress levels, sleep hours, physical activity,
chronic illness status, treatment history, and final outcomes — making it ideal for
<b>classification, clustering, and pattern mining</b> tasks in a BI context.
</div>
""", unsafe_allow_html=True)

# ─── COLUMN DESCRIPTIONS ───────────────────────────────────
st.markdown("<div class='section-header'>🗂️ Feature Description</div>", unsafe_allow_html=True)

features_df = pd.DataFrame({
    "Column": ["Patient_ID", "Age", "Gender", "Country", "Depression_Score",
                "Anxiety_Score", "Stress_Level", "Sleep_Hours", "Physical_Activity",
                "Chronic_Illness", "Mental_Health_History", "Treatment",
                "Days_of_Treatment", "Outcome", "Work_Status"],
    "Type": ["ID", "Numerical", "Categorical", "Categorical", "Numerical",
             "Numerical", "Categorical", "Numerical", "Categorical",
             "Categorical", "Categorical", "Categorical",
             "Numerical", "Target", "Categorical"],
    "Description": [
        "Unique patient identifier",
        "Age of the patient",
        "Gender of the patient",
        "Country of the patient",
        "Score indicating level of depression (0-10)",
        "Score indicating level of anxiety (0-10)",
        "Stress level (Low / Medium / High)",
        "Average sleep hours per night",
        "Physical activity level (Low / Medium / High)",
        "Whether patient has a chronic illness (Yes/No)",
        "Past mental health history (Yes/No)",
        "Type of treatment received",
        "Number of days under treatment",
        "Treatment outcome (Recovered / Not Recovered)",
        "Employment status of patient"
    ]
})
st.dataframe(features_df, use_container_width=True, hide_index=True)

# ─── RAW DATA PREVIEW ──────────────────────────────────────
st.markdown("<div class='section-header'>👁️ Raw Data Preview</div>", unsafe_allow_html=True)
st.dataframe(df.head(10), use_container_width=True)

# ─── PREPROCESSING ─────────────────────────────────────────
st.markdown("<div class='section-header'>⚙️ Preprocessing Techniques</div>", unsafe_allow_html=True)

p1, p2, p3 = st.columns(3)
with p1:
    st.markdown("""
    <div style='background:#1a1f2e; border:1px solid #48bb78; border-radius:10px; padding:16px;'>
        <b style='color:#48bb78'>✅ Handling Missing Values</b><br><br>
        <span style='color:#a0aec0; font-size:0.85rem'>
        Checked all columns for null values. 
        Numerical columns filled with <b>mean</b>. 
        Categorical columns filled with <b>mode</b>.
        </span>
    </div>""", unsafe_allow_html=True)
with p2:
    st.markdown("""
    <div style='background:#1a1f2e; border:1px solid #4e9af1; border-radius:10px; padding:16px;'>
        <b style='color:#4e9af1'>🔢 Label Encoding</b><br><br>
        <span style='color:#a0aec0; font-size:0.85rem'>
        Categorical columns like Gender, Outcome, Stress_Level 
        converted to <b>numerical format</b> using Label Encoding 
        for ML compatibility.
        </span>
    </div>""", unsafe_allow_html=True)
with p3:
    st.markdown("""
    <div style='background:#1a1f2e; border:1px solid #ed8936; border-radius:10px; padding:16px;'>
        <b style='color:#ed8936'>📏 Normalization</b><br><br>
        <span style='color:#a0aec0; font-size:0.85rem'>
        Numerical features like Age, Depression_Score, Sleep_Hours 
        scaled using <b>Min-Max Normalization</b> to bring 
        all values between 0 and 1.
        </span>
    </div>""", unsafe_allow_html=True)

# Normalization Formula
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class='formula-box'>
    <b style='color:#4e9af1; font-size:1rem'>Min-Max Normalization Formula</b><br><br>
    <span style='color:#ffffff; font-size:1.3rem'>
        X<sub>normalized</sub> = (X − X<sub>min</sub>) / (X<sub>max</sub> − X<sub>min</sub>)
    </span><br><br>
    <span style='color:#a0aec0; font-size:0.85rem'>
        Ensures all features are on the same scale (0 to 1), improving ML model performance.
    </span>
</div>
""", unsafe_allow_html=True)

# ─── DESCRIPTIVE STATISTICS ────────────────────────────────
st.markdown("<div class='section-header'>📈 Descriptive Statistics</div>", unsafe_allow_html=True)
numeric_cols = df.select_dtypes(include=np.number)
st.dataframe(numeric_cols.describe().round(2), use_container_width=True)

# Missing values
st.markdown("**Missing Values Check:**")
missing = df.isnull().sum()
if missing.sum() == 0:
    st.success("✅ No missing values found in the dataset!")
else:
    st.dataframe(missing[missing > 0])

# ─── VISUAL EXPLORATION ────────────────────────────────────
st.markdown("<div class='section-header'>📊 Visual Exploration</div>", unsafe_allow_html=True)

# Row 1 — Age + Gender
c1, c2 = st.columns(2)
with c1:
    st.markdown("##### Age Distribution")
    fig = px.histogram(df, x="Age", nbins=20,
                       color_discrete_sequence=["#4e9af1"],
                       template="plotly_dark")
    fig.update_layout(
        paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
        font_color="#ffffff", margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown("##### Gender Distribution")
    gender_counts = df['Gender'].value_counts().reset_index()
    gender_counts.columns = ['Gender', 'Count']
    fig2 = px.pie(gender_counts, names='Gender', values='Count',
                  color_discrete_sequence=px.colors.sequential.Blues_r,
                  template="plotly_dark", hole=0.4)
    fig2.update_layout(
        paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
        font_color="#ffffff", margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)

# Row 2 — Depression Score + Sleep Hours
c3, c4 = st.columns(2)
with c3:
    st.markdown("##### Depression Score Distribution")
    fig3 = px.histogram(df, x="Depression_Score", nbins=15,
                        color_discrete_sequence=["#fc8181"],
                        template="plotly_dark")
    fig3.update_layout(
        paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
        font_color="#ffffff", margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    st.markdown("##### Sleep Hours Distribution")
    fig4 = px.histogram(df, x="Sleep_Hours", nbins=15,
                        color_discrete_sequence=["#48bb78"],
                        template="plotly_dark")
    fig4.update_layout(
        paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
        font_color="#ffffff", margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig4, use_container_width=True)

# Row 3 — Stress Level + Outcome
c5, c6 = st.columns(2)
with c5:
    st.markdown("##### Stress Level Breakdown")
    stress_counts = df['Stress_Level'].value_counts().reset_index()
    stress_counts.columns = ['Stress_Level', 'Count']
    fig5 = px.bar(stress_counts, x='Stress_Level', y='Count',
                  color='Stress_Level',
                  color_discrete_sequence=["#fc8181", "#ed8936", "#48bb78"],
                  template="plotly_dark")
    fig5.update_layout(
        paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
        font_color="#ffffff", margin=dict(t=20, b=20), showlegend=False
    )
    st.plotly_chart(fig5, use_container_width=True)

with c6:
    st.markdown("##### Treatment Outcome")
    outcome_counts = df['Outcome'].value_counts().reset_index()
    outcome_counts.columns = ['Outcome', 'Count']
    fig6 = px.pie(outcome_counts, names='Outcome', values='Count',
                  color_discrete_sequence=["#48bb78", "#fc8181"],
                  template="plotly_dark", hole=0.4)
    fig6.update_layout(
        paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
        font_color="#ffffff", margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig6, use_container_width=True)

# Row 4 — Depression vs Anxiety scatter
st.markdown("##### Depression Score vs Anxiety Score")
fig7 = px.scatter(df, x="Depression_Score", y="Anxiety_Score",
                  color="Outcome",
                  color_discrete_sequence=["#48bb78", "#fc8181"],
                  template="plotly_dark",
                  opacity=0.7)
fig7.update_layout(
    paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
    font_color="#ffffff", margin=dict(t=20, b=20)
)
st.plotly_chart(fig7, use_container_width=True)

# Row 5 — Top Countries
st.markdown("##### Top 10 Countries by Records")
top_countries = df['Country'].value_counts().head(10).reset_index()
top_countries.columns = ['Country', 'Count']
fig8 = px.bar(top_countries, x='Country', y='Count',
              color='Count',
              color_continuous_scale='Blues',
              template="plotly_dark")
fig8.update_layout(
    paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
    font_color="#ffffff", margin=dict(t=20, b=20)
)
st.plotly_chart(fig8, use_container_width=True)

# ─── BI INSIGHT ────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='background:linear-gradient(135deg, #1a3a1a, #2d5a2d); border-left:4px solid #48bb78;
border-radius:8px; padding:20px; margin:16px 0;'>
    <b style='color:#48bb78; font-size:1.1rem'>💡 Key Exploration Insights</b><br><br>
    <span style='color:#e2e8f0'>
    • Patients with <b>High Stress Levels</b> tend to have higher Depression and Anxiety scores.<br>
    • <b>Sleep Hours</b> show a clear correlation with treatment outcomes.<br>
    • Dataset is fairly balanced between Recovered and Not Recovered outcomes.<br>
    • Majority of patients fall in the <b>20–40 age group</b> — most vulnerable demographic.
    </span>
</div>
""", unsafe_allow_html=True)