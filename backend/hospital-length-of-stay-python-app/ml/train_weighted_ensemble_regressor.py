# ============================================================
# Weighted Ensemble Regressor
# ============================================================

import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


from utils.preprocessing import DataPreprocessor
preprocessor = DataPreprocessor()

from sklearn.base import (
    BaseEstimator,
    RegressorMixin
)

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

import xgboost as xgb
import lightgbm as lgb



# ============================================================
# Custom Weighted Ensemble
# ============================================================

class WeightedEnsembleRegressor(
    BaseEstimator,
    RegressorMixin
):

    def __init__(
        self,
        models,
        weights
    ):
        self.models = models
        self.weights = weights

    def fit(
        self,
        X,
        y
    ):

        for model in self.models:
            model.fit(X, y)

        return self

    def predict(
        self,
        X
    ):

        predictions = np.column_stack([

            model.predict(X)

            for model in self.models

        ])

        return np.average(
            predictions,
            axis=1,
            weights=self.weights
        )

# ============================================================
# Load Dataset
# ============================================================

df = preprocessor.load_data(
    "data/hospital_length_of_stay_dataset.csv"
)

# Summary
preprocessor.dataset_summary(df)

# Validation
df = preprocessor.validate_data(df)

# Missing Values
print("\nMissing Values Report: ")
print(
    preprocessor.missing_value_report(df)
)

# Remove Duplicates
df = preprocessor.remove_duplicates(df)

# Feature Engineering
df = preprocessor.feature_engineering(df)

# Outlier Processing
numeric_cols = df.select_dtypes(
    include="number"
).columns.tolist()

numeric_cols.remove(
    "length_of_stay_days"
)

print("\nOutlier: ")
print(
    preprocessor.detect_outliers(
        df,
        numeric_cols
    )
)

df = preprocessor.cap_outliers(
    df,
    numeric_cols
)

print("\ndf shape: ",df.shape)


# Features & Target
X = df.drop(
    columns=["length_of_stay_days"]
)

y = df["length_of_stay_days"]

print("X Shape: ",X.shape)
print("y Shape: ",y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
print("\nBefore preprocessing:")
print("X_train Shape: ",X_train.shape)
print("X_test Shape: ",X_test.shape)

X_train = preprocessor.fit_transform(
    X_train
)

X_test = preprocessor.transform(
    X_test
)

feature_names = (
    preprocessor.preprocessor
    .get_feature_names_out()
)

#print("\nfeature_names: ")
#print(feature_names)

print("\nAfter preprocessing:")
print("X_train Shape: ",X_train.shape)
print("X_test Shape: ",X_test.shape)




X_train = pd.DataFrame(
    X_train,
    columns=feature_names
)

X_test = pd.DataFrame(
    X_test,
    columns=feature_names
)



preprocessor.save_preprocessor(
    "saved_artifacts/preprocessor.pkl"
)

# ============================================================
# Base Models
# ============================================================

rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

et_model = ExtraTreesRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=500,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)

lgbm_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    random_state=42,
    verbosity=-1
)

# ============================================================
# Weights
# ============================================================

weights = [
    0.10,   # RF
    0.15,   # ET
    0.35,   # XGB
    0.40    # LGBM
]

# ============================================================
# Ensemble
# ============================================================

ensemble = WeightedEnsembleRegressor(
    models=[
        rf_model,
        et_model,
        xgb_model,
        lgbm_model
    ],
    weights=weights
)

# ============================================================
# MLflow
# ============================================================

mlflow.set_experiment(
    "WeightedEnsembleRegressor"
)

with mlflow.start_run():

    ensemble.fit(
        X_train,
        y_train
    )

    predictions = ensemble.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    mape = mean_absolute_percentage_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    mlflow.log_metrics({
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R2_SCORE": r2
    })

    mlflow.log_param(
        "weights",
        str(weights)
    )

    print("\n=========================")
    print("Weighted Ensemble")
    print("=========================")

    print(f"MAE      : {mae:.4f}")
    print(f"MSE      : {mse:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"MAPE     : {mape:.4f}")
    print(f"R2 Score : {r2:.4f}")