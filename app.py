import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="🫀",
    layout="wide"
)

# -------------------------------
# Title & Description
# -------------------------------
st.title("🫀 Heart Disease Risk Prediction")
st.caption("ML Assignment 2 | Developed by Karthikaa Kothandapani")

st.markdown("""
This interactive web application allows users to **compare multiple machine learning
classification models** for predicting the **presence of heart disease**
based on clinical attributes.

📌 *Only test datasets should be uploaded, as per Streamlit free-tier limitations.*
""")

# -------------------------------
# Sidebar Content
# -------------------------------
st.sidebar.header("🔧 Application Controls")

st.sidebar.markdown("""
### About This App
- Multiple trained ML models are available
- Upload a CSV test dataset
- Select a model to evaluate predictions
- View metrics, confusion matrix & classification report
""")

# -------------------------------
# Load Available Models
# -------------------------------
MODEL_DIR = "model/saved_models"

available_models = {
    "Logistic Regression": "Logistic Regression.pkl",
    "Decision Tree": "Decision Tree.pkl",
    "KNN": "KNN.pkl",
    "Naive Bayes": "Naive Bayes.pkl",
    "Random Forest": "Random Forest.pkl",
    "XGBoost": "XGBoost.pkl"
}

selected_model_name = st.sidebar.selectbox(
    "Select Classification Model",
    list(available_models.keys())
)

# Load selected model
model_path = os.path.join(MODEL_DIR, available_models[selected_model_name])
trained_model = joblib.load(model_path)

# -------------------------------
# Dataset Upload
# -------------------------------
uploaded_csv = st.file_uploader(
    "📂 Upload Test Dataset (CSV format only)",
    type=["csv"]
)

if uploaded_csv is not None:
    # Read dataset
    heart_df = pd.read_csv(uploaded_csv)

    st.subheader("📊 Uploaded Dataset Preview")
    st.dataframe(heart_df.head())

    # -------------------------------
    # Data Preparation
    # -------------------------------
    if "num" not in heart_df.columns:
        st.error("❌ Target column `num` not found in dataset.")
    else:
        features = heart_df.drop(columns=["num"])
        target = heart_df["num"]

        # Convert categorical columns if any
        features = pd.get_dummies(features, drop_first=True)

        # Align columns if model was trained with fixed feature set
        try:
            predictions = trained_model.predict(features)
            probabilities = (
                trained_model.predict_proba(features)[:, 1]
                if hasattr(trained_model, "predict_proba")
                else None
            )

            # -------------------------------
            # Evaluation Metrics
            # -------------------------------
            st.subheader("📈 Model Evaluation Metrics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Accuracy", f"{accuracy_score(target, predictions):.3f}")
                st.metric("Precision", f"{precision_score(target, predictions):.3f}")

            with col2:
                st.metric("Recall", f"{recall_score(target, predictions):.3f}")
                st.metric("F1 Score", f"{f1_score(target, predictions):.3f}")

            with col3:
                if probabilities is not None:
                    st.metric("ROC-AUC", f"{roc_auc_score(target, probabilities):.3f}")
                else:
                    st.metric("ROC-AUC", "N/A")

            # -------------------------------
            # Confusion Matrix
            # -------------------------------
            st.subheader("🧩 Confusion Matrix")

            cm = confusion_matrix(target, predictions)
            fig, ax = plt.subplots()
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax
            )
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("Actual Label")
            st.pyplot(fig)

            # -------------------------------
            # Classification Report
            # -------------------------------
            st.subheader("📄 Classification Report")
            report = classification_report(target, predictions, output_dict=False)
            st.text(report)

        except Exception as e:
            st.error("⚠️ Error during prediction. Please ensure feature compatibility.")
            st.exception(e)

else:
    st.info("⬆️ Upload a CSV test dataset to begin model evaluation.")


