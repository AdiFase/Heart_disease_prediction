# ❤️ Heart Disease Prediction System

An end-to-end Machine Learning project that predicts whether a person is at risk of heart disease based on clinical and health-related attributes.

---

## 📌 Project Overview

Heart Disease Prediction System is a machine learning application developed using Python and Scikit-learn. The project analyzes patient health information and predicts the likelihood of heart disease.

The project follows a complete machine learning pipeline including:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Feature Selection
- Data Scaling
- Model Training
- Hyperparameter Tuning
- Model Evaluation
- Model Deployment using Streamlit

The trained Logistic Regression model is deployed through a Streamlit web application where users can enter medical information and receive instant predictions.

---

## 🎯 Problem Statement

Heart disease is one of the leading causes of death worldwide. Early prediction can help doctors and patients make informed decisions regarding diagnosis and treatment.

This project aims to build an accurate machine learning model capable of predicting heart disease from patient medical records.

---

## 🚀 Features

- User-friendly Streamlit interface
- Data preprocessing pipeline
- Feature engineering
- StandardScaler integration
- Logistic Regression model
- Saved model using Joblib/Pickle
- Fast prediction

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

---

## 📂 Project Structure

```text
Heart-Disease-Prediction/
│
├── app.py
├── logistic.heart_disease_prediction_model.pkl
├── scaler.pkl
├── columns.pkl
├── requirements.txt
├── README.md
```

---

## ⚙️ Machine Learning Workflow

```text
Dataset
    ↓
Data Cleaning
    ↓
EDA
    ↓
Feature Engineering
    ↓
Feature Selection
    ↓
Train-Test Split
    ↓
Feature Scaling
    ↓
Model Training
    ↓
Model Evaluation
    ↓
Hyperparameter Tuning
    ↓
Save Model
    ↓
Streamlit Deployment
```

---

## 🤖 Machine Learning Models Evaluated

The following algorithms were trained and evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Gaussian Naive Bayes

The models were compared using Accuracy Score, Confusion Matrix, Classification Report, and F1 Score.

After evaluation, **Logistic Regression** was selected as the final model because it achieved the best balance between accuracy and generalization.

---

## 📊 Model Evaluation Metrics

- Accuracy Score
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 💻 Installation

Clone the repository

```bash
git clone https://github.com/your-username/Heart-Disease-Prediction.git
```

Move into the project folder

```bash
cd Heart-Disease-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📈 How to Use

1. Launch the Streamlit application.
2. Enter the patient's medical information.
3. Click the **Predict** button.
4. The model predicts whether the patient is likely to have heart disease.

---

## 📌 Future Improvements

- Deploy on Streamlit Cloud
- Add XGBoost and CatBoost models
- Improve UI/UX
- Explain predictions using SHAP
- Docker deployment
- FastAPI integration
- Model monitoring

---

## 📚 Learning Outcomes

This project demonstrates:

- End-to-End Machine Learning Pipeline
- Feature Engineering
- Feature Selection
- Data Scaling
- Model Comparison
- Hyperparameter Tuning
- Model Serialization
- Streamlit Deployment

---

## ⚠️ Disclaimer

This project is developed for educational and learning purposes only. It should not be used as a substitute for professional medical diagnosis or treatment.

