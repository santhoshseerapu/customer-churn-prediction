# Customer Churn Prediction and Explainable AI System

## 📌 Project Overview

This project is a Machine Learning-based **Customer Churn Prediction System** designed to predict whether a customer is likely to leave a service.

The project uses multiple machine learning models to analyze customer data and identify potential churn patterns. The system is designed to support better business decision-making by providing predictions and insights from customer data.

---

## 🎯 Objectives

- Predict whether a customer is likely to churn.
- Analyze customer-related features affecting churn.
- Compare multiple Machine Learning models.
- Provide an easy-to-use prediction interface.
- Support explainable and interpretable Machine Learning analysis.

---

## 🤖 Machine Learning Models

The project includes multiple trained Machine Learning models:

- Logistic Regression
- Random Forest
- XGBoost

The trained models are stored as `.pkl` files and can be used for customer churn prediction.

---

## 📂 Project Structure

```text
customer-churn-prediction/
│
├── Data/
│   └── raw/
│       └── Customer dataset
│
├── notebooks/
│   └── Jupyter notebooks for data analysis and model development
│
├── app.py
│
├── feature_columns.pkl
├── logistic_regression_model.pkl
├── rf_churn_model.pkl
├── xgboost_churn_model.pkl
├── ltv_model.pkl
│
├── .gitignore
│
└── README.md