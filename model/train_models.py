# =========================================================
# ML Assignment 2 - Model Training Script
# Dataset: Heart Disease (heart.csv)
# =========================================================

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("data/heart.csv")

# -----------------------------
# 2. Convert target to Binary
# num = 0  -> No disease
# num > 0  -> Disease
# -----------------------------
df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

# -----------------------------
# 3. Handle Missing Values
# -----------------------------
imputer = SimpleImputer(strategy="most_frequent")
df[:] = imputer.fit_transform(df)

# -----------------------------
# 4. Encode Categorical Columns
# -----------------------------
label_encoder = LabelEncoder()

for col in df.columns:
    if df[col].dtype == object:
        df[col] = label_encoder.fit_transform(df[col])

# -----------------------------
# 5. Feature & Target Split
# -----------------------------
X = df.drop(columns=["num", "id"])
y = df["num"]

# -----------------------------
# 6. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 7. Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 8. Define Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
}

# -----------------------------
# 9. Evaluation Function
# -----------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

# -----------------------------
# 10. Train, Evaluate & Save
# -----------------------------
results = []
os.makedirs("model/saved_models", exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    metrics["Model"] = name
    results.append(metrics)

    joblib.dump(model, f"model/saved_models/{name}.pkl")

# -----------------------------
# 11. Save Results Table
# -----------------------------
results_df = pd.DataFrame(results)
results_df = results_df[
    ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
]

results_df.to_csv("model/model_comparison.csv", index=False)

print("\nModel Training Completed Successfully!\n")
print(results_df)
