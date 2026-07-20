# 🏥 Hospital Length of Stay Prediction

## Overview

The **Hospital Length of Stay Prediction System** is an end-to-end Machine Learning application that predicts the expected number of days a patient will remain hospitalized based on demographic information, admission details, clinical conditions, lifestyle factors, and hospital-related attributes.

The application consists of:

- FastAPI REST API
- Streamlit Web Application
- Machine Learning Training Pipeline
- Model Comparison Framework
- MLflow Experiment Tracking
- Modular Project Architecture

The primary objective is to assist healthcare providers in:

- Resource Planning
- Bed Management
- Capacity Forecasting
- Clinical Decision Support
- Hospital Operations Optimization

---

# Project Architecture

```
                    +--------------------+
                    | Streamlit UI       |
                    +---------+----------+
                              |
                              |
                              ▼
                    +--------------------+
                    | FastAPI REST API   |
                    +---------+----------+
                              |
              --------------------------------
              |              |               |
              ▼              ▼               ▼
        Router Layer    Service Layer   Implementation
                              |
                              ▼
                    Preprocessing Pipeline
                              |
                              ▼
                     Trained ML Model
                              |
                              ▼
                    Prediction Response
```


# Features

- Hospital Length of Stay Prediction
- FastAPI REST API
- Streamlit Dashboard
- Automated Data Preprocessing
- Feature Engineering
- Model Comparison
- Best Model Selection
- MLflow Tracking
- Modular Codebase
- Production Ready Architecture

---

# Machine Learning Workflow

```
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Encoding
      │
      ▼
Scaling
      │
      ▼
Train/Test Split
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Model Comparison
      │
      ▼
Best Model Selection
      │
      ▼
Model Deployment
      │
      ▼
Prediction API
```

---

# Machine Learning Models

The project evaluates numerous regression algorithms.

### Baseline Models

- Dummy Regressor

### Linear Models

- Multiple Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Bayesian Ridge
- Lars
- Lasso Lars
- Orthogonal Matching Pursuit
- Passive Aggressive Regressor
- Huber Regressor
- RANSAC
- TheilSen

### Tree Models

- Decision Tree
- Extra Trees
- Random Forest

### Ensemble Models

- Bagging Regressor
- AdaBoost
- Gradient Boosting
- Histogram Gradient Boosting
- Voting Regressor
- Hard Voting
- Weighted Voting
- Weighted Ensemble
- Blending
- Stacking

### Support Vector Models

- Support Vector Regression

### Gradient Boosting Libraries

- XGBoost
- LightGBM
- CatBoost

---

# Input Features

The prediction model accepts patient and hospital-related information including:

| Category | Examples |
|-----------|----------|
| Demographics | Age, Gender, Marital Status |
| Admission | Admission Type, Admission Source |
| Diagnosis | Primary Diagnosis |
| Clinical | Severity Score, Comorbidity Count |
| Lifestyle | Smoking, Alcohol |
| Insurance | Insurance Type |
| Vitals | BMI |
| History | Previous Admissions |

---

# Prediction Output

The API predicts:

- Estimated Length of Stay (Days)

Example

```json
{
    "success": true,
    "predicted_length_of_stay": 6.84,
    "execution_time": 0.018
}
```

---

# REST API

## Health Check

```
GET /
```

Returns

```json
{
    "application":"Clinical Decision Support System",
    "version":"1.0.0",
    "status":"Running"
}
```

---

## Health Endpoint

```
GET /health
```

Returns

```json
{
    "status":"healthy"
}
```

---

## Prediction Endpoint

```
POST /los/predict
```

Example Request

```json
{
  "admission_source": "Referral",
  "admission_type": "Emergency",
  "age": 45,
  "bmi": 27.4,
  "comorbidity_count": 2,
  "discharge_disposition": "Home",
  "gender": "Male",
  "hospital_department": "Cardiology",
  "insurance_type": "Private",
  "previous_admissions": 1,
  "primary_diagnosis": "Pneumonia",
  "severity_score": 7.5,
  "smoker": "No",
  "glucose": 190,
  "marital_status": "Married",
  "oxygen_saturation": 96,
  "hemoglobin": 12.7,
  "creatinine": 2.86,
  "blood_pressure": 160,
  "temperature": 98,
  "heart_rate": 99
}
```

---

# Streamlit Dashboard

The Streamlit application provides:

- User-friendly Prediction Form
- Interactive Input Controls
- Real-Time Prediction
- REST API Integration
- Clean Visualization
- Prediction Summary

---

# MLflow Integration

The project supports experiment tracking using MLflow.

Tracked information includes:

- Parameters
- Metrics
- Artifacts
- Models
- Feature Importance
- Model Versions
- Prediction Inference

---

# Model Evaluation Metrics

The regression models are evaluated using:

- MAE
- MSE
- RMSE
- R² Score
- Adjusted R²
- MAPE (if applicable)

---

# Technology Stack

| Category | Technology |
|------------|------------|
| Language | Python |
| Backend | FastAPI |
| Frontend | Streamlit |
| Machine Learning | Scikit-Learn |
| Boosting | XGBoost |
| Boosting | LightGBM |
| Boosting | CatBoost |
| Tracking | MLflow |
| Data | Pandas |
| Numerical Computing | NumPy |
| Visualization | Matplotlib |
| Visualization | Seaborn |
| Model Serialization | Joblib |

---

# Installation

Clone the repository

```bash
git clone <repository-url>
```

Create virtual environment

```bash
python -m venv venv
```

Activate environment

Windows

```bash
venv\Scripts\activate
```

Linux

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running FastAPI

```bash
cd app

uvicorn main:app --reload
```

API

```
http://localhost:8000
```

Swagger

```
http://localhost:8000/docs
```

---

# Running Streamlit

```bash
streamlit run ui/app.py
```

---

# Model Training

Execute individual training scripts from the **app/ml** directory.

Example

```bash
python train_randomforest_regressor.py
```

or

```bash
python train_xgb_regressor.py
```

---

# Model Comparison

Run

```bash
python app/ml_comparison/models_comparison.py
```

This compares all regression algorithms and identifies the best-performing model.

---

# Best Model Selection

The framework automatically selects the best model based on evaluation metrics and stores it for inference.

---

# Future Enhancements

- SHAP Explainability
- Batch Prediction
- Docker Deployment
- Kubernetes Deployment
- Azure Deployment
- CI/CD Pipeline
- Model Monitoring
- Data Drift Detection
- Feature Store Integration
- Authentication & Authorization

---

# Applications

- Hospital Resource Planning
- Bed Occupancy Management
- Patient Flow Optimization
- Clinical Decision Support
- Hospital Analytics
- Healthcare Operations
- Insurance Risk Assessment

---

# Author

Developed as a modular Machine Learning solution for predicting hospital length of stay using multiple regression algorithms, FastAPI, Streamlit, and MLflow following production-ready software engineering practices.