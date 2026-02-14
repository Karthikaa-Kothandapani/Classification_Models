# Heart Disease Classification – ML Assignment 2

##  Assignment Overview

This project demonstrates a **complete end-to-end Machine Learning workflow**, including:

* Data preprocessing
* Training and evaluating multiple **classification models**
* Comparing model performance using standard metrics
* Building an **interactive Streamlit web application**
* Deploying the application on **Streamlit Community Cloud (Free Tier)**

The objective is to predict the **presence of heart disease** using clinical and physiological attributes.

---

## 📂 Project Structure

```
Classification_Models/
│
├── app.py                         # Streamlit web application
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
│
├── data/
│   └── heart.csv                  # Dataset used for training & testing
│
├── model/
│   ├── train_models.py            # Model training & evaluation script
│   ├── model_comparison.csv       # Performance comparison of all models
│   └── saved_models/              # Trained ML models (.pkl files)
│       ├── Logistic Regression.pkl
│       ├── Decision Tree.pkl
│       ├── KNN.pkl
│       ├── Naive Bayes.pkl
│       ├── Random Forest.pkl
│       └── XGBoost.pkl
```

---

## 📊 Dataset Description

* **Dataset Name:** Heart Disease Dataset (`heart.csv`)
* **Target Variable:** `num`

  * `0` → No heart disease
  * `1` → Presence of heart disease
* **Features:**
  Includes age, sex, chest pain type, resting blood pressure, cholesterol, ECG results, maximum heart rate, exercise-induced angina, and other clinical indicators.

Only **small test datasets** are uploaded through the Streamlit interface, as required by the **Streamlit Community Cloud free-tier limitations**.

---

## 🤖 Machine Learning Models Implemented

The following classification models were implemented, trained, and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Random Forest Classifier
6. XGBoost Classifier

All models follow a consistent preprocessing pipeline and are saved for reuse within the Streamlit application.

---

## 📈 Model Evaluation Metrics

Each model is evaluated using multiple performance metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Matthews Correlation Coefficient (MCC)

A consolidated comparison of all models is available in:

```
model/model_comparison.csv
```

---

## 🖥️ Streamlit Web Application Features

The Streamlit application fulfills **all required assignment criteria**.

### ✅ Implemented Features

* CSV dataset upload option (test data only)
* Model selection dropdown (multiple trained models)
* Prediction on uploaded dataset
* Display of evaluation metrics
* Confusion matrix visualization
* Classification report output

The application enables interactive testing and comparison of different classification models.

---

## 🌐 Live Application (Deployed)

🔗 **Streamlit App URL:**
👉 [https://classificationmodels-karthikaa-assignment.streamlit.app/](https://classificationmodels-karthikaa-assignment.streamlit.app/)

This link can be used directly for assignment evaluation.

---

## ▶️ How to Run Locally (Optional)

To run the application locally:

```bash
# Clone the repository
git clone https://github.com/Karthikaa-Kothandapani/Classification_Models.git
cd Classification_Models

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Linux / Mac
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## 🛠️ Technologies Used

* Python 3
* Scikit-learn
* XGBoost
* Pandas, NumPy
* Matplotlib / Seaborn
* Streamlit
* Git & GitHub

---

## 🎯 Learning Outcomes

This assignment demonstrates:

* End-to-end machine learning pipeline development
* Model evaluation and comparison
* Handling categorical and missing data
* Building interactive ML dashboards
* Deploying ML applications using cloud platforms

---

## ✅ Assignment Compliance Checklist

✔ Multiple classification models implemented
✔ Model evaluation and comparison
✔ Interactive Streamlit web application
✔ Dataset upload functionality
✔ Model selection dropdown
✔ Confusion matrix and classification report
✔ Deployed on Streamlit Community Cloud (Free Tier)
✔ Shareable public application link