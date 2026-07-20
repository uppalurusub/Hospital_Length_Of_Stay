# ============================================================
# Hard Voting Regressor (Median Ensemble)
# ============================================================
from utils.preprocessing import DataPreprocessor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

preprocessor = DataPreprocessor()
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.base import (
    BaseEstimator,
    RegressorMixin
)

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor
)

import xgboost as xgb
import lightgbm as lgb


from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# Custom Median Voting Regressor
# ============================================================

class HardVotingRegressor(
    BaseEstimator,
    RegressorMixin
):

    def __init__(
        self,
        models
    ):
        self.models = models

    def fit(
        self,
        X,
        y
    ):

        for model in self.models:
            model.fit(
                X,
                y
            )

        return self

    def predict(
        self,
        X
    ):

        predictions = np.column_stack([

            model.predict(X)

            for model in self.models

        ])

        return np.median(
            predictions,
            axis=1
        )

# ============================================================
# Load Data
# ============================================================

# Load Data
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

print("\nAfter preprocessing:")
print("X_train Shape: ",X_train.shape)
print("X_test Shape: ",X_test.shape)

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
    max_depth=6,
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
# Ensemble
# ============================================================

model = HardVotingRegressor(

    models=[

        rf_model,
        et_model,
        xgb_model,
        lgbm_model

    ]

)

# ============================================================
# Train
# ============================================================

model.fit(
    X_train,
    y_train
)

# ============================================================
# Predict
# ============================================================

predictions = model.predict(
    X_test
)



# ============================================================
# Metrics
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    predictions
)

print("\n==============================")
print("Hard Voting Regressor")
print("==============================")

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")


# ============================================================
# Metrics With Weights
# ============================================================

weighted_predictions = (

    0.10 * rf_model.predict(X_test)

    +

    0.20 * et_model.predict(X_test)

    +

    0.35 * xgb_model.predict(X_test)

    +

    0.35 * lgbm_model.predict(X_test)

)

mae = mean_absolute_error(
    y_test,
    weighted_predictions
)

mse = mean_squared_error(
    y_test,
    weighted_predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    weighted_predictions
)

print("\n==============================")
print("Hard Voting Regressor With Weights")
print("==============================")

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")