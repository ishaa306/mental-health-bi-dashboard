import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv("Global_Mental_Health_Dataset_2025.csv")

print("=" * 70)
print("ASSOCIATION RULES TEST — CLEANED DATA")
print("=" * 70)

# ============================================================
# CREATE CLEAN COPY
# ============================================================
df_clean = df.copy()

# Remove impossible / invalid values
df_clean = df_clean[
    (df_clean["Sleep_Hours"] >= 0) &
    (df_clean["Sleep_Hours"] <= 12) &
    (df_clean["Depression_Score"] >= 0) &
    (df_clean["Anxiety_Score"] >= 0) &
    (df_clean["Days_of_Treatment"] >= 0)
]

# Remove rows with missing important values
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

print(f"Original Rows : {len(df)}")
print(f"Cleaned Rows  : {len(df_clean)}")

# ============================================================
# DATA-BASED THRESHOLDS
# ============================================================
dep_cut = df_clean["Depression_Score"].quantile(0.75)
anx_cut = df_clean["Anxiety_Score"].quantile(0.75)
sleep_cut = df_clean["Sleep_Hours"].quantile(0.25)

print("\nThresholds chosen from dataset:")
print(f"High Depression >= {dep_cut}")
print(f"High Anxiety    >= {anx_cut}")
print(f"Low Sleep       <= {sleep_cut}")

# ============================================================
# CREATE ASSOCIATION DATASET
# ============================================================
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

# Strong combined condition
association_df["Severe_Condition"] = (
    association_df["High_Depression"] &
    association_df["Low_Sleep"]
)

print("\nColumn Counts:")
for col in association_df.columns:
    print(f"{col:25} -> {association_df[col].sum()}")

# ============================================================
# APRIORI
# ============================================================
frequent_items = apriori(
    association_df,
    min_support=0.03,
    use_colnames=True
)

print(f"\nFrequent Itemsets Found: {len(frequent_items)}")

# ============================================================
# GENERATE RULES
# ============================================================
rules = association_rules(
    frequent_items,
    metric="confidence",
    min_threshold=0.45
)

if rules.empty:
    print("\nNo rules were generated.")
else:
    # Keep only rules leading to High Risk
    rules = rules[
        rules["consequents"].apply(
            lambda x: "High_Risk" in x
        )
    ]

    # Keep moderately useful rules
    rules = rules[
        (rules["confidence"] >= 0.45) &
        (rules["lift"] >= 1.01)
    ]

    if len(rules) == 0:
        print("\nNo High Risk rules passed the filters.")
    else:
        # Convert frozensets to readable text
        rules["antecedents"] = rules["antecedents"].apply(
            lambda x: ", ".join(list(x))
        )

        rules["consequents"] = rules["consequents"].apply(
            lambda x: ", ".join(list(x))
        )

        # Keep only needed columns
        rules = rules[[
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift"
        ]]

        # Round values
        rules["support"] = rules["support"].round(3)
        rules["confidence"] = rules["confidence"].round(3)
        rules["lift"] = rules["lift"].round(3)

        # Sort strongest rules first
        rules = rules.sort_values(
            by=["lift", "confidence"],
            ascending=False
        )

        print("\n" + "=" * 70)
        print("TOP HIGH RISK RULES")
        print("=" * 70)

        print(rules.head(10).to_string(index=False))

print("\nDone.")