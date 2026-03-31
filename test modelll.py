import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 70)
print("MENTAL HEALTH DATASET — STRESS LEVEL CLASSIFICATION TEST")
print("=" * 70)

df = pd.read_csv("Global_Mental_Health_Dataset_2025.csv")

print(f"\nDataset Loaded Successfully")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# ============================================================
# 2. TARGET COLUMN
# ============================================================
TARGET = "Stress_Level"

print(f"\nTarget Column: {TARGET}")
print("Unique Values:", df[TARGET].unique())

# ============================================================
# 3. PREPROCESSING
# ============================================================
df_model = df.copy()

# Drop columns not useful for prediction
drop_cols = []

for col in df_model.columns:
    if "id" in col.lower() or "patient" in col.lower():
        drop_cols.append(col)

# Remove original target leakage columns if needed
if "Outcome" in df_model.columns:
    drop_cols.append("Outcome")

df_model.drop(columns=drop_cols, inplace=True, errors="ignore")

# Fill missing values
for col in df_model.columns:
    if df_model[col].dtype == "object":
        df_model[col].fillna(df_model[col].mode()[0], inplace=True)
    else:
        df_model[col].fillna(df_model[col].mean(), inplace=True)

# Encode categorical feature columns
encoders = {}

for col in df_model.select_dtypes(include="object").columns:
    if col != TARGET:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        encoders[col] = le

# Encode target column manually
df_model[TARGET] = df_model[TARGET].replace({
    "Low": "Normal",
    "Medium": "Normal",
    "High": "High Risk",
    "Severe": "High Risk"
})

target_encoder = LabelEncoder()
df_model[TARGET] = target_encoder.fit_transform(df_model[TARGET])
# Remove rows that could not be mapped
df_model = df_model.dropna(subset=[TARGET])

# Convert to integer
df_model[TARGET] = df_model[TARGET].astype(int)

print("\nStress Level Distribution:")
print(df_model[TARGET].value_counts().sort_index())

# ============================================================
# 4. FEATURES & TARGET
# ============================================================
features = [col for col in df_model.columns if col != TARGET]

X = df_model[features]
y = df_model[TARGET]

print("\nFeatures Used:")
for f in features:
    print("-", f)

# ============================================================
# 5. TRAIN TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scale only for LR and KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain Size : {len(X_train)}")
print(f"Test Size  : {len(X_test)}")

print("\nClass Distribution in Training Set:")
for label, name in [
    (0, "High Risk"),
    (1, "Normal")

]:
    print(f"{name:8} : {(y_train == label).sum()}")

# ============================================================
# 6. DEFINE MODELS
# ============================================================
models = {
    "Logistic Regression": (
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        True
    ),

    "Decision Tree": (
        DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42
        ),
        False
    ),

    "Random Forest": (
        RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42
        ),
        False
    ),

    "KNN": (
        KNeighborsClassifier(
    n_neighbors=11,
    weights="distance"
),
        True
    )
}

# ============================================================
# 7. TRAIN & EVALUATE
# ============================================================
results = []

label_names = ["High Risk", "Normal"]

print("\n" + "=" * 70)
print("MODEL RESULTS")
print("=" * 70)

for name, (model, needs_scaling) in models.items():

    if needs_scaling:
        Xtr = X_train_scaled
        Xte = X_test_scaled
    else:
        Xtr = X_train
        Xte = X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    results.append({
        "Algorithm": name,
        "Accuracy": round(acc * 100, 2),
        "Precision": round(prec * 100, 2),
        "Recall": round(rec * 100, 2),
        "F1 Score": round(f1 * 100, 2)
    })

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_names,
            zero_division=0
        )
    )

# ============================================================
# 8. FINAL COMPARISON
# ============================================================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Accuracy", ascending=False)

print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)
print(results_df.to_string(index=False))

# ============================================================
# 9. BEST MODEL
# ============================================================
best_model = results_df.iloc[0]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)
print(f"Algorithm : {best_model['Algorithm']}")
print(f"Accuracy  : {best_model['Accuracy']}%")
print(f"Precision : {best_model['Precision']}%")
print(f"Recall    : {best_model['Recall']}%")
print(f"F1 Score  : {best_model['F1 Score']}%")

# ============================================================
# 10. FEATURE IMPORTANCE FOR RANDOM FOREST
# ============================================================
rf_model = models["Random Forest"][0]

# Refit to make sure feature_importances_ exists
rf_model.fit(X_train, y_train)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

for _, row in importance_df.iterrows():
    print(f"{row['Feature']:<30} {row['Importance']:.4f}")

print("\nDone.")