# 🏥 Hospital Length of Stay Prediction System

An end-to-end AI-powered healthcare application that predicts a patient's **Hospital Length of Stay (LOS)** using Machine Learning.

The project consists of:

- 🧠 FastAPI Backend
- ⚛️ React TypeScript Frontend
- 📊 Streamlit Dashboard
- 🤖 Multiple Machine Learning Regression Models
- 📈 Model Comparison & Best Model Selection
- 📦 Pre-trained Model Artifacts

---

# Project Architecture

```
Hospital_Length_Of_Stay
│
├── backend
│   ├── hospital-length-of-stay-python-app
│   ├── data
│   └── README.md
│
├── ui
│   ├── reactjs
│   │     └── hospital-length-of-stay-react-app
│   │
│   └── streamlit
│         └── hospital-length-of-stay-streamlit-app
│
└── README.md
```

---

# Features

- Predict Hospital Length of Stay
- REST API using FastAPI
- Modern React Dashboard
- Streamlit User Interface
- Automatic Data Preprocessing
- Model Comparison
- Best Model Selection
- Saved Model Loading
- MLflow-ready Tracking
- JSON-based Prediction API
- Modular Project Structure

---

# Technology Stack

## Backend

- Python
- FastAPI
- Scikit-Learn
- XGBoost
- LightGBM
- CatBoost
- Pandas
- NumPy
- Joblib
- Uvicorn

---

## Frontend

- React 19
- TypeScript
- Vite
- Material UI
- Axios
- React Router

---

## Machine Learning

The project evaluates multiple regression algorithms including:

- Dummy Regressor
- Linear Regression
- Ridge Regression
- Bayesian Ridge
- Lasso
- ElasticNet
- Lars
- Lasso Lars
- Orthogonal Matching Pursuit
- Passive Aggressive Regressor
- Huber Regressor
- RANSAC Regressor
- TheilSen Regressor
- Decision Tree
- Random Forest
- Extra Trees
- Bagging
- AdaBoost
- Gradient Boosting
- HistGradientBoosting
- Support Vector Regressor
- KNN Regressor
- XGBoost
- LightGBM
- CatBoost
- Voting Regressor
- Weighted Voting Regressor
- Hard Voting Regressor
- Stacking Regressor
- Blending Regressor

---

# Backend Structure

```
backend
│
├── data
│
├── hospital-length-of-stay-python-app
│
├── implementations
├── ml
├── ml_comparison
├── models
├── routers
├── services
├── utils
│
├── best_saved_artifacts
│
├── main.py
└── requirements.txt
```

---

# Frontend Structure

```
hospital-length-of-stay-react-app

src
│
├── api
├── components
├── config
├── models
├── pages
├── styles
│
├── App.tsx
└── main.tsx
```

---

# Streamlit Structure

```
hospital-length-of-stay-streamlit-app

├── api
├── components
├── pages
├── utils
├── app.py
└── requirements.txt
```

---

# Installation

## Clone Repository

```bash
git clone <repository-url>

cd Hospital_Length_Of_Stay
```

---

# Backend Setup

Navigate to backend

```bash
cd backend/hospital-length-of-stay-python-app
```

Create virtual environment

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux/macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

Install dependencies

```bash
pip install -r ../requirements.txt
```

Run FastAPI

```bash
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

ReDoc

```
http://localhost:8000/redoc
```

---

# React Frontend Setup

Navigate

```bash
cd ui/reactjs/hospital-length-of-stay-react-app
```

Install

```bash
npm install
```

Run

```bash
npm run dev
```

Application

```
http://localhost:5173
```

Build

```bash
npm run build
```

---

# Streamlit Setup

Navigate

```bash
cd ui/streamlit/hospital-length-of-stay-streamlit-app
```

Install

```bash
pip install -r requirements.txt
```

Run

```bash
streamlit run app.py
```

---

# API Endpoint

Example

```
POST /predict
```

Content-Type

```
application/json
```

Example Request

```json
{
  "age": 65,
  "gender": "Male",
  "admission_type": "Emergency",
  "diagnosis": "Pneumonia"
}
```

Example Response

```json
{
  "predicted_length_of_stay": 5.84
}
```

---

# Model Training

Training scripts are located under

```
ml/
```

Each algorithm has an individual training script.

Examples

```
train_randomforest_regressor.py

train_xgb_regressor.py

train_lgbm_regressor.py

train_catboost_regressor.py

train_bayesian_ridge_regressor.py
```

---

# Best Model Selection

The project automatically compares regression models using evaluation metrics and identifies the best-performing model.

Saved artifacts include:

```
best_saved_artifacts/

preprocessor.pkl

BayesianRidge.pkl
```

---

# Evaluation Metrics

Typical regression metrics include:

- MAE
- MSE
- RMSE
- R² Score
- Adjusted R²
- MAPE
- SMAPE

---

# Workflow

```
Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Feature Engineering
     │
     ▼
Preprocessing
     │
     ▼
Train Multiple Models
     │
     ▼
Model Evaluation
     │
     ▼
Best Model Selection
     │
     ▼
Save Model
     │
     ▼
FastAPI Prediction API
     │
     ▼
React / Streamlit UI
```

---

# Future Enhancements

- MLflow Model Registry
- Docker Support
- Kubernetes Deployment
- CI/CD Pipeline
- Authentication & Authorization
- Explainable AI (SHAP/LIME)
- Batch Prediction
- Monitoring Dashboard
- Drift Detection
- Cloud Deployment (Azure/AWS/GCP)

---

# License

This project is intended for educational, research, and healthcare analytics purposes.

---

# Author

Hospital Length of Stay Prediction System

Built using

- FastAPI
- React
- TypeScript
- Streamlit
- Scikit-Learn
- Machine Learning
- Healthcare Analytics