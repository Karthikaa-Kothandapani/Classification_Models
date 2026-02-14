# =========================================================
# ML Assignment 2 - Streamlit Application
# Dataset: Heart Disease Classification
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Classification",
    layout="wide"
)

st.title("❤️ Heart Disease Classification – ML Models")
st.markdown(
    "This app demonstrates multiple **classification models** trained on the Heart Disease dataset."
)

# ---------------------------------------------------------
# Load Model Comparison Metrics
# ---------------------------------------------------------
metrics_df = pd.read_csv("model/model_comparison.csv")

# ---------------------------------------------------------
# Sidebar – Model Selection
# ---------------------------------------------------------
st.sidebar.header("Model Selection")

model_name = st.sidebar.selectbox(
    "Choose a Classification Model",
    metrics_df["Model"].tolist()
)

model_path = f"model/saved_models/{model_name}.pkl"
model = joblib.load(model_path)

# ---------------------------------------------------------
# Display Metrics
# ---------------------------------------------------------
st.subheader("📊 Model Evaluation Metrics")

selected_metrics = metrics_df[metrics_df["Model"] == model_name]
st.dataframe(selected_metrics, use_container_width=True)

# ---------------------------------------------------------
# CSV Upload Section
# ---------------------------------------------------------
st.subheader("📂 Upload Test Dataset (CSV)")

uploaded_file = st.file_uploader(
    "Upload test data CSV (same format as heart.csv) to view confusion matrix and classification report.",
    type=["csv"]
)

# ---------------------------------------------------------
# Prediction & Evaluation
# ---------------------------------------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("### Preview of Uploaded Data")
    st.dataframe(data.head())

    # -----------------------------
    # Preprocessing (same as training)
    # -----------------------------
    data["num"] = data["num"].apply(lambda x: 1 if x > 0 else 0)

    imputer = SimpleImputer(strategy="most_frequent")
    data[:] = imputer.fit_transform(data)

    label_encoder = LabelEncoder()
    for col in data.columns:
        if data[col].dtype == object:
            data[col] = label_encoder.fit_transform(data[col])

    X = data.drop(columns=["num", "id"])
    y_true = data["num"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # Predictions
    # -----------------------------
    y_pred = model.predict(X_scaled)

    # -----------------------------------------------------
    # Confusion Matrix
    # -----------------------------------------------------
    st.subheader("🧩 Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # -----------------------------------------------------
    # Classification Report
    # -----------------------------------------------------
# -----------------------------------------------------
# Classification Report (Table Format)
# -----------------------------------------------------
    st.subheader("📄 Classification Report")

    report_dict = classification_report(
        y_true,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_df = report_df.round(3)

    st.dataframe(report_df, use_container_width=True)


else:
    st.info("⬆️ Upload a CSV file to run predictions and view results.")
