# ML Assignment 2 – Classification Models for Heart Disease Prediction

---

## a. Problem Statement

The objective of this assignment is to build and evaluate multiple **machine learning classification models** to predict the **presence of heart disease** using clinical and physiological parameters.
The project also aims to demonstrate an **end-to-end ML workflow**, including data preprocessing, model training, evaluation, comparison, and deployment through an interactive web application.

---

## b. Dataset Description  **[1 Mark]**

* **Dataset Name:** Heart Disease Dataset (`heart.csv`)
* **Target Variable:** `num`

  * `0` → No heart disease
  * `1` → Presence of heart disease
* **Number of Records:** 920
* **Number of Features:** 15 input features + 1 target column
* **Feature Types:**

  * Numerical (age, cholesterol, resting blood pressure, etc.)
  * Categorical (sex, chest pain type, ECG results, etc.)

The dataset contains clinical attributes commonly used in cardiovascular risk assessment.
Only **test datasets** are uploaded in the Streamlit app, in compliance with Streamlit Community Cloud free-tier limitations.

---

## c. Models Used and Evaluation  **[6 Marks]**

The following **six classification models** were implemented, trained, and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (kNN)
4. Naive Bayes
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

### 🔹 Model Comparison Table

| ML Model Name            | Accuracy | AUC   | Precision | Recall | F1 Score | MCC   |
| ------------------------ | -------- | ----- | --------- | ------ | -------- | ----- |
| Logistic Regression      | 0.815    | 0.894 | 0.827     | 0.843  | 0.835    | 0.625 |
| Decision Tree            | 0.739    | 0.731 | 0.745     | 0.804  | 0.774    | 0.469 |
| kNN                      | 0.848    | 0.876 | 0.885     | 0.833  | 0.859    | 0.696 |
| Naive Bayes              | 0.826    | 0.884 | 0.830     | 0.863  | 0.846    | 0.647 |
| Random Forest (Ensemble) | 0.826    | 0.919 | 0.850     | 0.833  | 0.842    | 0.649 |
| XGBoost (Ensemble)       | 0.826    | 0.904 | 0.837     | 0.853  | 0.845    | 0.647 |

---

## d. Model Performance Observations  **[3 Marks]**

| ML Model Name            | Observation about Model Performance                                                                                              |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Provided stable and interpretable results with good overall balance, but slightly lower performance compared to ensemble models. |
| Decision Tree            | Showed lower accuracy and MCC, indicating overfitting and sensitivity to data variations.                                        |
| kNN                      | Achieved the **highest accuracy and MCC**, performing well due to effective neighborhood-based classification.                   |
| Naive Bayes              | Performed consistently with strong recall, indicating good sensitivity in detecting heart disease cases.                         |
| Random Forest (Ensemble) | Delivered strong AUC and balanced performance, benefiting from ensemble averaging and reduced overfitting.                       |
| XGBoost (Ensemble)       | Achieved high AUC and recall, demonstrating strong predictive power and robustness on the dataset.                               |

---

## Step 6: Deployment on Streamlit Community Cloud

The trained models were deployed using an **interactive Streamlit web application** on **Streamlit Community Cloud (Free Tier)**.

### 🔹 Application Features:

* CSV dataset upload (test data only)
* Model selection dropdown
* Evaluation metrics display
* Confusion matrix visualization
* Classification report output

### 🔗 Live Application Link:

👉 **[https://classificationmodels-karthikaa-assignment.streamlit.app/](https://classificationmodels-karthikaa-assignment.streamlit.app/)**


