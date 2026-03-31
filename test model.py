import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────
print("=" * 60)
print("  MENTAL HEALTH CLASSIFICATION — BINARY BENCHMARK")
print("=" * 60)

df = pd.read_csv("Global_Mental_Health_Dataset_2025.csv")
print(f"\n✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ──────────────────────────────────────────────
# 2. MERGE 4 CLASSES → 2 CLASSES
# ──────────────────────────────────────────────
#   Poor + Fair       →  "At Risk"
#   Good + Excellent  →  "Stable"
# ──────────────────────────────────────────────
merge_map = {
    "Poor"      : "At Risk",
    "Fair"      : "At Risk",
    "Good"      : "Stable",
    "Excellent" : "Stable"
}

df["Outcome_Binary"] = df["Outcome"].map(merge_map)

print(f"\n✅ Outcome classes merged:")
print(f"   Poor + Fair       → At Risk  ({(df['Outcome_Binary'] == 'At Risk').sum()} patients)")
print(f"   Good + Excellent  → Stable   ({(df['Outcome_Binary'] == 'Stable').sum()} patients)")

TARGET = "Outcome_Binary"

# ──────────────────────────────────────────────
# 3. PREPROCESSING
# ──────────────────────────────────────────────
df_model = df.copy()

# Drop original Outcome and ID-like columns
drop_cols = [c for c in df_model.columns if "id" in c.lower() or "patient" in c.lower()]
drop_cols.append("Outcome")
df_model.drop(columns=drop_cols, inplace=True, errors="ignore")

# Fill missing values
for col in df_model.columns:
    if df_model[col].dtype == "object":
        df_model[col].fillna(df_model[col].mode()[0], inplace=True)
    else:
        df_model[col].fillna(df_model[col].mean(), inplace=True)

# Label encode all categorical columns
le = LabelEncoder()
for col in df_model.select_dtypes(include="object").columns:
    if col != TARGET:
        df_model[col] = le.fit_transform(df_model[col])

# Encode target
df_model[TARGET] = df_model[TARGET].map({"At Risk": 0, "Stable": 1})

features = [c for c in df_model.columns if c != TARGET]
print(f"\n✅ Preprocessing done.")
print(f"   Features used : {features}")

# ──────────────────────────────────────────────
# 4. SPLIT
# ──────────────────────────────────────────────
X = df_model[features]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n✅ Train/Test split: {len(X_train)} train | {len(X_test)} test")
print(f"   Class balance  : At Risk={y_train.value_counts()[0]} | Stable={y_train.value_counts()[1]}")

# ──────────────────────────────────────────────
# 5. MODELS
# ──────────────────────────────────────────────
models = {
    "Logistic Regression" : (LogisticRegression(max_iter=1000, random_state=42), True),
    "Decision Tree"       : (DecisionTreeClassifier(random_state=42),             False),
    "Random Forest"       : (RandomForestClassifier(n_estimators=100, random_state=42), False),
    "KNN"                 : (KNeighborsClassifier(n_neighbors=5),                  True),
}

# ──────────────────────────────────────────────
# 6. TRAIN & EVALUATE
# ──────────────────────────────────────────────
results = []

print("\n" + "=" * 60)
print("  RESULTS PER ALGORITHM")
print("=" * 60)

for name, (model, use_scaled) in models.items():
    Xtr = X_train_sc if use_scaled else X_train
    Xte = X_test_sc  if use_scaled else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    results.append({
        "Algorithm" : name,
        "Accuracy"  : round(acc  * 100, 2),
        "Precision" : round(prec * 100, 2),
        "Recall"    : round(rec  * 100, 2),
        "F1-Score"  : round(f1   * 100, 2),
    })

    print(f"\n🔹 {name}")
    print(f"   Accuracy  : {acc*100:.2f}%")
    print(f"   Precision : {prec*100:.2f}%")
    print(f"   Recall    : {rec*100:.2f}%")
    print(f"   F1-Score  : {f1*100:.2f}%")
    print(f"   Confusion Matrix:")
    print(f"   (Rows=Actual, Cols=Predicted)  [At Risk | Stable]")
    print(f"   {cm}")
    print(f"\n   Full Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["At Risk", "Stable"],
                                zero_division=0))

# ──────────────────────────────────────────────
# 7. SUMMARY TABLE
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL COMPARISON TABLE")
print("=" * 60)

results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
results_df.index = range(1, len(results_df) + 1)
print(results_df.to_string())

# ──────────────────────────────────────────────
# 8. WINNER
# ──────────────────────────────────────────────
best = results_df.iloc[0]
second = results_df.iloc[1]

print(f"\n🏆 BEST ALGORITHM    : {best['Algorithm']}")
print(f"   Accuracy          : {best['Accuracy']}%")
print(f"   F1-Score          : {best['F1-Score']}%")
print(f"\n🥈 2ND BEST          : {second['Algorithm']}")
print(f"   Accuracy          : {second['Accuracy']}%")
print(f"   F1-Score          : {second['F1-Score']}%")
print(f"\n✅ Use these 2 for classification.py")

# ──────────────────────────────────────────────
# 9. FEATURE IMPORTANCE (Random Forest)
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FEATURE IMPORTANCE — Random Forest")
print("=" * 60)
rf_model = models["Random Forest"][0]
feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_imp = feat_imp.sort_values(ascending=False)
for feat, score in feat_imp.items():
    bar = "█" * int(score * 100)
    print(f"   {feat:<30} {score:.4f}  {bar}")

print("\n✅ Done! Share the accuracy scores above.")
print("=" * 60)