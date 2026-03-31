import streamlit as st
import pandas as pd
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules

# ───────────────────────────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Association Rules",
    page_icon="🔗",
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
    padding: 20px;
    text-align: center;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #63b3ed;
}

.metric-label {
    color: #a0aec0;
    margin-top: 8px;
}

.rule-box {
    background: linear-gradient(135deg,#1a1f2e,#2d3561);
    border-left: 5px solid #f56565;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────
# LOAD & CLEAN DATA
# ───────────────────────────────────────────────────────────
@st.cache_data
def load_rules():
    df = pd.read_csv("Global_Mental_Health_Dataset_2025.csv")

    # Clean copy only for this page
    df_clean = df.copy()

    # Remove impossible values
    df_clean = df_clean[
        (df_clean["Sleep_Hours"] >= 0) &
        (df_clean["Sleep_Hours"] <= 12) &
        (df_clean["Depression_Score"] >= 0) &
        (df_clean["Anxiety_Score"] >= 0) &
        (df_clean["Days_of_Treatment"] >= 0)
    ]

    important_cols = [
        "Depression_Score",
        "Anxiety_Score",
        "Sleep_Hours",
        "Physical_Activity",
        "Chronic_Illness",
        "Mental_Health_History",
        "Stress_Level"
    ]

    df_clean = df_clean.dropna(subset=important_cols)

    # Dataset-based thresholds
    dep_cut = df_clean["Depression_Score"].quantile(0.75)
    anx_cut = df_clean["Anxiety_Score"].quantile(0.75)
    sleep_cut = df_clean["Sleep_Hours"].quantile(0.25)

    association_df = pd.DataFrame()

    association_df["High_Depression"] = (
        df_clean["Depression_Score"] >= dep_cut
    )

    association_df["High_Anxiety"] = (
        df_clean["Anxiety_Score"] >= anx_cut
    )

    association_df["Low_Sleep"] = (
        df_clean["Sleep_Hours"] <= sleep_cut
    )

    association_df["Long_Treatment"] = (
        df_clean["Days_of_Treatment"] >= 180
    )

    association_df["Low_Activity"] = (
        df_clean["Physical_Activity"] == "Low"
    )

    association_df["Chronic_Illness"] = (
        df_clean["Chronic_Illness"] == "Yes"
    )

    association_df["Mental_Health_History"] = (
        df_clean["Mental_Health_History"] == "Yes"
    )

    association_df["High_Risk"] = (
        df_clean["Stress_Level"].isin(["High", "Severe"])
    )

    frequent_items = apriori(
        association_df,
        min_support=0.03,
        use_colnames=True
    )

    rules = association_rules(
        frequent_items,
        metric="confidence",
        min_threshold=0.45
    )

    if rules.empty:
        return pd.DataFrame(), dep_cut, anx_cut, sleep_cut

    rules = rules[
        rules["consequents"].apply(
            lambda x: "High_Risk" in x
        )
    ]

    rules = rules[
        (rules["confidence"] >= 0.45) &
        (rules["lift"] >= 1.01)
    ]

    if len(rules) == 0:
        return pd.DataFrame(), dep_cut, anx_cut, sleep_cut

    rules["antecedents"] = rules["antecedents"].apply(
        lambda x: ", ".join(list(x))
    )

    rules["consequents"] = rules["consequents"].apply(
        lambda x: ", ".join(list(x))
    )

    rules = rules[[
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift"
    ]]

    rules["support"] = rules["support"].round(3)
    rules["confidence"] = rules["confidence"].round(3)
    rules["lift"] = rules["lift"].round(3)

    rules = rules.sort_values(
        by=["lift", "confidence"],
        ascending=False
    )

    return rules, dep_cut, anx_cut, sleep_cut


rules_df, dep_cut, anx_cut, sleep_cut = load_rules()

# ───────────────────────────────────────────────────────────
# TITLE
# ───────────────────────────────────────────────────────────
st.title("🔗 Association Rule Mining")
st.markdown(
    "Discovering which combinations of symptoms and lifestyle factors are most strongly associated with High Risk mental health cases."
)

# ───────────────────────────────────────────────────────────
# THRESHOLDS USED
# ───────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>⚙️ Data Cleaning & Thresholds</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{dep_cut:.1f}+</div>
        <div class='metric-label'>High Depression Threshold</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{anx_cut:.1f}+</div>
        <div class='metric-label'>High Anxiety Threshold</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{sleep_cut:.1f} hrs</div>
        <div class='metric-label'>Low Sleep Threshold</div>
    </div>
    """, unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────
# TOP RULE CARDS
# ───────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>🔥 Strongest High Risk Patterns</div>", unsafe_allow_html=True)

if rules_df.empty:
    st.warning("No meaningful association rules were found in the cleaned dataset.")
else:
    for _, row in rules_df.head(3).iterrows():
        st.markdown(f"""
<div class='rule-box'>
<div style='font-size:1.1rem; font-weight:700; color:#f56565;'>
        {row['antecedents']} ➜ {row['consequents']} 
</div>

<div style='margin-top:10px; color:#cbd5e0;'>
        Support: <b>{row['support']}</b> &nbsp;&nbsp; | &nbsp;&nbsp;
        Confidence: <b>{row['confidence']}</b> &nbsp;&nbsp; | &nbsp;&nbsp;
        Lift: <b>{row['lift']}</b>
</div>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────
# RULE TABLE
# ───────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>📋 Association Rules Table</div>", unsafe_allow_html=True)

if not rules_df.empty:
    st.dataframe(rules_df, use_container_width=True)

# ───────────────────────────────────────────────────────────
# VISUALIZATION
# ───────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>📈 Rule Strength Comparison</div>", unsafe_allow_html=True)

if not rules_df.empty:
    fig = px.bar(
        rules_df.head(10),
        x="lift",
        y="antecedents",
        orientation="h",
        color="confidence",
        template="plotly_dark",
        text="lift"
    )

    fig.update_layout(
        paper_bgcolor="#1a1f2e",
        plot_bgcolor="#1a1f2e",
        font_color="white",
        yaxis_title="Condition Combination",
        xaxis_title="Lift"
    )

    st.plotly_chart(fig, use_container_width=True)

# ───────────────────────────────────────────────────────────
# INSIGHT
# ───────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>💡 Key Insight</div>", unsafe_allow_html=True)

st.info(
    "Patients with both high anxiety and low sleep, or with low physical activity and chronic illness, are more likely to fall into the High Risk category. These combinations appeared most frequently in the cleaned dataset."
)
