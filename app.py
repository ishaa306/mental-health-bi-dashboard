import streamlit as st

st.set_page_config(
    page_title="Mental Health BI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme like reference
st.markdown("""
<style>
    /* Dark background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stSidebar {
        background-color: #1a1f2e;
    }
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #2d3561);
        border: 1px solid #3d4f7c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4e9af1;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0;
        margin-top: 4px;
    }
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        border-left: 4px solid #4e9af1;
        padding-left: 12px;
        margin: 24px 0 16px 0;
    }
    /* Star schema box */
    .schema-box {
        background: #1a1f2e;
        border: 1px solid #3d4f7c;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        margin: 8px;
    }
    .fact-table {
        background: linear-gradient(135deg, #1e3a5f, #2d5a8e);
        border: 2px solid #4e9af1;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .dim-table {
        background: linear-gradient(135deg, #1a2f1a, #2d5a2d);
        border: 2px solid #48bb78;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    /* Highlight box */
    .highlight-box {
        background: linear-gradient(135deg, #1a1f2e, #2d3561);
        border-left: 4px solid #4e9af1;
        border-radius: 8px;
        padding: 20px;
        margin: 16px 0;
    }
    h1, h2, h3 {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── MAIN PAGE ─────────────────────────────────────────────
st.title(" Global Mental Health" )

st.markdown("---")

# Project description
st.markdown("""
<div class='highlight-box'>
<b> Project Overview</b><br><br>
This application is a comprehensive <b>Business Intelligence tool</b> for analyzing the 
<b>Global Mental Health Dataset 2025</b>. It covers the end-to-end <b>KDD (Knowledge Discovery in Databases)</b> 
process — from raw data exploration to machine learning predictions and actionable BI decisions.
</div>
""", unsafe_allow_html=True)

# ─── KEY STATS ─────────────────────────────────────────────
st.markdown("<div class='section-header'> Dataset at a Glance</div>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class='metric-card'>
        <div class='metric-value'>5,000+</div>
        <div class='metric-label'>Total Records</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class='metric-card'>
        <div class='metric-value'>20+</div>
        <div class='metric-label'>Features</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class='metric-card'>
        <div class='metric-value'>3</div>
        <div class='metric-label'>ML Algorithms</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class='metric-card'>
        <div class='metric-value'>2025</div>
        <div class='metric-label'>Dataset Year</div>
    </div>""", unsafe_allow_html=True)

# ─── KDD PROCESS ───────────────────────────────────────────
st.markdown("<div class='section-header'>🔄 KDD Process Followed</div>", unsafe_allow_html=True)

kdd_cols = st.columns(5)
steps = [
    ("1️⃣", "Data\nSelection", "#4e9af1"),
    ("2️⃣", "Data\nPreprocessing", "#48bb78"),
    ("3️⃣", "Data\nTransformation", "#ed8936"),
    ("4️⃣", "Data\nMining", "#9f7aea"),
    ("5️⃣", "Knowledge\nPresentation", "#fc8181"),
]
for col, (icon, label, color) in zip(kdd_cols, steps):
    with col:
        st.markdown(f"""
        <div style='background:#1a1f2e; border:1px solid {color}; border-radius:10px; 
        padding:16px; text-align:center; margin:4px;'>
            <div style='font-size:1.5rem'>{icon}</div>
            <div style='color:{color}; font-weight:600; font-size:0.85rem; margin-top:8px'>{label}</div>
        </div>""", unsafe_allow_html=True)

# ─── STAR SCHEMA ───────────────────────────────────────────
st.markdown("<div class='section-header'>🗄️ Data Warehouse — Star Schema</div>", unsafe_allow_html=True)
st.markdown("A Star Schema has been designed for analytical processing. It separates business data into facts and dimensions.")

# Visual Star Schema
col_left, col_center, col_right = st.columns([1, 1.5, 1])

with col_left:
    st.markdown("""
    <div class='dim-table'>
        <b style='color:#48bb78'>Dim_Student</b><br>
        <small style='color:#a0aec0'>
         student_id<br>age<br>gender<br>country<br>academic_level
        </small>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='dim-table'>
        <b style='color:#48bb78'>Dim_Platform</b><br>
        <small style='color:#a0aec0'>
        🔑 platform_id<br>platform_name<br>platform_type
        </small>
    </div>""", unsafe_allow_html=True)

with col_center:
    st.markdown("""
    <div class='fact-table'>
        <b style='color:#4e9af1; font-size:1.1rem'>⭐ Fact_MentalHealth</b><br><br>
        <small style='color:#a0aec0'>
        🔑 record_id<br>
        🔗 student_id (FK)<br>
        🔗 platform_id (FK)<br>
        🔗 behavior_id (FK)<br>
        🔗 health_id (FK)<br><br>
        daily_usage_time<br>
        sleep_hours<br>
        mental_health_score<br>
        addiction_score<br>
        conflicts_over_social_media
        </small>
    </div>""", unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div class='dim-table'>
        <b style='color:#48bb78'>Dim_Behavior</b><br>
        <small style='color:#a0aec0'>
        🔑 behavior_id<br>usage_pattern<br>addiction_level<br>avg_daily_usage
        </small>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='dim-table'>
        <b style='color:#48bb78'>Dim_Health</b><br>
        <small style='color:#a0aec0'>
        🔑 health_id<br>mental_health_score<br>sleep_quality<br>stress_level
        </small>
    </div>""", unsafe_allow_html=True)

# ─── ALGORITHMS USED ───────────────────────────────────────
st.markdown("<div class='section-header'>🤖 Algorithms Used</div>", unsafe_allow_html=True)

acol1, acol2, acol3, acol4 = st.columns(4)
algos = [
    ("Logistic Regression", "Supervised", "Predicts addiction risk (High/Low) based on usage and sleep patterns.", "#4e9af1"),
    ("Decision Tree", "Supervised", "Creates visual decision paths to classify student behavior.", "#48bb78"),
    ("K-Means Clustering", "Unsupervised", "Groups students into behavioral cohorts based on usage patterns.", "#ed8936"),
    ("Apriori Rules", "Pattern Mining", "Discovers hidden behavioral associations in the dataset.", "#9f7aea"),
]
for col, ( name, atype, desc, color) in zip([acol1, acol2, acol3, acol4], algos):
    with col:
        st.markdown(f"""
        <div style='background:#1a1f2e; border:1px solid {color}; border-radius:10px; 
        padding:16px; height:200px;'>
            <div style='font-size:1.8rem'>{icon}</div>
            <div style='color:{color}; font-weight:700; margin:8px 0 4px 0'>{name}</div>
            <div style='background:{color}22; color:{color}; font-size:0.7rem; 
            padding:2px 8px; border-radius:20px; display:inline-block; margin-bottom:8px'>{atype}</div>
            <div style='color:#a0aec0; font-size:0.8rem'>{desc}</div>
        </div>""", unsafe_allow_html=True)

# ─── FOOTER ────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#4a5568; padding:20px; border-top:1px solid #2d3748'>
     
</div>
""", unsafe_allow_html=True)
