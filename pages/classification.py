
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ───────────────────────────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Classification",
    page_icon="📊",
    layout="wide"
)

# ───────────────────────────────────────────────────────────
# STYLE
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

.metric-card {
    background: linear-gradient(135deg,#1a1f2e,#2d3561);
    border: 1px solid #3d4f7c;
    border-radius: 14px;
    padding: 24px;
    text-align: center;
}

.metric-value {
    font-size: 2.3rem;
    font-weight: 700;
}

.metric-label {
    color: #a0aec0;
    margin-top: 8px;
}

.result-box {
    border-radius: 14px;
    padding: 28px;
    text-align: center;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────
# LOAD DATA
# ───────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("Global_Mental_Health_Dataset_2025.csv")


df = load_data()

# Convert Stress_Level into 2 classes
# Low + Medium -> Normal
# High + Severe -> High Risk

df["Stress_Level"] = df["Stress_Level"].replace({
    "Low": "Normal",
    "Medium": "Normal",
    "High": "High Risk",
    "Severe": "High Risk"
})

# ───────────────────────────────────────────────────────────
# PREPROCESSING
# ───────────────────────────────────────────────────────────
feature_cols = [
    "Age",
    "Gender",
    "Country",
    "Depression_Score",
    "Anxiety_Score",
    "Sleep_Hours",
    "Physical_Activity",
    "Chronic_Illness",
    "Mental_Health_History",
    "Treatment",
    "Days_of_Treatment",
    "Work_Status"
]

# Fill missing values
for col in feature_cols + ["Stress_Level"]:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])
encoders = {}

for col in [
    "Gender",
    "Country",
    "Physical_Activity",
    "Chronic_Illness",
    "Mental_Health_History",
    "Treatment",
    "Work_Status",
    "Stress_Level"
]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df[feature_cols]
y = df["Stress_Level"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ───────────────────────────────────────────────────────────
# MODELS
# ───────────────────────────────────────────────────────────
knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(X_train, y_train)

knn_pred = knn_model.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# ───────────────────────────────────────────────────────────
# TITLE
# ───────────────────────────────────────────────────────────
st.title("📊 Stress Level Classification")
st.markdown(
    "Predicting whether a patient belongs to the 🟢 Normal or 🔴 High Risk category."
)

# ───────────────────────────────────────────────────────────
# ACCURACY
# ───────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>🎯 Model Accuracy</div>", unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value' style='color:#4e9af1'>{knn_acc*100:.2f}%</div>
        <div class='metric-label'>KNN Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value' style='color:#48bb78'>{rf_acc*100:.2f}%</div>
        <div class='metric-label'>Random Forest Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────
# COMPARISON CHART
# ───────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>📈 Algorithm Comparison</div>", unsafe_allow_html=True)

compare_df = pd.DataFrame({
    "Algorithm": ["KNN", "Random Forest"],
    "Accuracy": [knn_acc * 100, rf_acc * 100]
})

fig = px.bar(
    compare_df,
    x="Algorithm",
    y="Accuracy",
    color="Algorithm",
    text="Accuracy",
    template="plotly_dark"
)

fig.update_layout(
    paper_bgcolor="#1a1f2e",
    plot_bgcolor="#1a1f2e",
    font_color="white",
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# ───────────────────────────────────────────────────────────
# CONFUSION MATRIX
# ───────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>🧩 KNN Confusion Matrix</div>", unsafe_allow_html=True)

cm = confusion_matrix(y_test, knn_pred)

cm_df = pd.DataFrame(
    cm,
    columns=["Predicted High Risk", "Predicted Normal"],
    index=["Actual High Risk", "Actual Normal"]
)

st.dataframe(cm_df, use_container_width=True)

# ───────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ───────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>⭐ Most Important Features</div>", unsafe_allow_html=True)

importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False).head(5)

fig2 = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    template="plotly_dark",
    text="Importance"
)

fig2.update_layout(
    paper_bgcolor="#1a1f2e",
    plot_bgcolor="#1a1f2e",
    font_color="white"
)

st.plotly_chart(fig2, use_container_width=True)

# ───────────────────────────────────────────────────────────
# PREDICTOR
# ───────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>🔍 Patient Risk Predictor</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    p_age = st.number_input("🎂 Age", 10, 100, 30)
    p_dep = st.number_input("😔 Depression Score", 0, 30, 10)
    p_sleep = st.number_input("😴 Sleep Hours", 0.0, 12.0, 7.0)
    p_days = st.number_input("💊 Days of Treatment", 0, 365, 30)

with col2:
    p_gender = st.selectbox("👤 Gender", ["Male", "Female"])
    p_country = st.selectbox("🌍 Country", sorted(load_data()["Country"].dropna().unique()))
    p_anx = st.number_input("😰 Anxiety Score", 0, 30, 8)
    p_activity = st.selectbox("🏃 Physical Activity", ["Low", "Moderate", "High"])

with col3:
    p_chronic = st.selectbox("🩺 Chronic Illness", ["No", "Yes"])
    p_history = st.selectbox("🧠 Mental Health History", ["No", "Yes"])
    p_treatment = st.selectbox("💉 Treatment", sorted(load_data()["Treatment"].dropna().unique()))
    p_work = st.selectbox("💼 Work Status", sorted(load_data()["Work_Status"].dropna().unique()))
    predict_btn = st.button("Predict Risk", use_container_width=True)

if predict_btn:
    input_row = pd.DataFrame([[
        p_age,
        encoders["Gender"].transform([p_gender])[0],
        encoders["Country"].transform([p_country])[0],
        p_dep,
        p_anx,
        p_sleep,
        encoders["Physical_Activity"].transform([p_activity])[0],
        encoders["Chronic_Illness"].transform([p_chronic])[0],
        encoders["Mental_Health_History"].transform([p_history])[0],
        encoders["Treatment"].transform([p_treatment])[0],
        p_days,
        encoders["Work_Status"].transform([p_work])[0]
    ]], columns=feature_cols)

    # Convert values into scaled form used during training
    input_scaled = scaler.transform(input_row)

    # Predict using Random Forest
    pred = rf_model.predict(input_scaled)[0]
    probs = rf_model.predict_proba(input_scaled)[0]

    # Extra rule for clearly severe cases
    if (
        p_dep >= 20 or
        p_anx >= 18 or
        p_sleep <= 4 or
        p_days >= 180
    ):
        label = "High Risk"
    else:
        label = encoders["Stress_Level"].inverse_transform([pred])[0]

    # Color and icon
    if label == "High Risk":
        color = "#fc8181"
        emoji = "🔴"
    else:
        color = "#68d391"
        emoji = "🟢"

    confidence = max(probs) * 100

    st.markdown(f"""
<div class='result-box'
style='background:linear-gradient(135deg,#1a1f2e,#2d3561);
border:2px solid {color};'>

<div style='font-size:3rem'>{emoji}</div>

<div style='font-size:2rem; font-weight:700; color:{color}; margin-top:10px;'>
{label}
</div>

<div style='margin-top:14px; color:#cbd5e0;'>
Confidence: <b>{confidence:.2f}%</b><br><br>
Most influential factors in this dataset:<br>
<b>Days of Treatment, Sleep Hours, Age, Depression Score and Anxiety Score</b>
</div>

</div>
""", unsafe_allow_html=True)

