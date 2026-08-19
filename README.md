# Telco Customer Churn Prediction — End-to-End Machine Learning & Deployment

An end-to-end machine learning project that predicts whether a telecommunications customer is likely to churn. This project covers the complete ML workflow, from data preprocessing and model comparison to hyperparameter tuning, model evaluation, API development, and Streamlit deployment.
https://telco-customer-churn-prediction-end-to-end-machine-learning-de.streamlit.app/
##  Project Overview

Customer churn is a major business challenge for telecommunications companies. The goal of this project is to build a machine learning system that identifies customers who are likely to leave the company.

The project focuses not only on achieving high predictive performance, but also on building a practical and deployable machine learning solution.

## Objectives

- Predict customer churn using customer and service information.
- Perform data cleaning and preprocessing.
- Handle categorical variables using one-hot encoding.
- Apply feature scaling where appropriate.
- Compare multiple machine learning algorithms.
- Evaluate models using multiple classification metrics.
- Handle class imbalance using SMOTE and class weights.
- Perform hyperparameter tuning using GridSearchCV.
- Analyze ROC-AUC and Precision-Recall curves.
- Save the trained model using Joblib.
- Build a REST API using FastAPI.
- Build an interactive prediction interface using Streamlit.

##  Dataset

The project uses the **Telco Customer Churn dataset**, containing information about customer demographics, account information, services, contract details, billing, and churn status.

### Target Variable

`Churn`

- `0` → Customer did not churn
- `1` → Customer churned

##  Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Imbalanced-learn
- Matplotlib
- Joblib
- FastAPI
- Uvicorn
- Streamlit
- Requests

##  Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature / Target Separation
   ↓
Categorical Encoding
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Comparison
   ↓
Class Imbalance Handling
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
FastAPI
   ↓
Streamlit
   ↓
Deployment


**Author**
**Uroosa Khan**
