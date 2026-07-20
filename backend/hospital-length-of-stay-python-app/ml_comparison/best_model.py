import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from utils.preprocessing import DataPreprocessor
import joblib
import os
import joblib
import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("mlflow").setLevel(logging.CRITICAL)

os.makedirs(
    "best_saved_artifacts",
    exist_ok=True
)


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

# Linear Models
from sklearn.linear_model import (
    
    BayesianRidge,
    
)


from utils.preprocessing import DataPreprocessor

# =====================================================
# MLFLOW
# =====================================================

mlflow.set_tracking_uri(
    #"http://127.0.0.1/5000"
    "http://localhost:5000"
)

EXPERIMENT_NAME = "Hospital_LOS_Regression"

mlflow.set_experiment(
    EXPERIMENT_NAME
)

# =====================================================
# METRICS
# =====================================================

def evaluate_model(
    y_true,
    y_pred
):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_true,
        y_pred
    )

    mape = mean_absolute_percentage_error(
        y_true,
        y_pred
    )

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }


# =====================================================
# DATA
# =====================================================

preprocessor = DataPreprocessor()

df = preprocessor.load_data(
    "../data/hospital_length_of_stay_dataset.csv"
)

df = preprocessor.validate_data(df)
df = preprocessor.remove_duplicates(df)
df = preprocessor.feature_engineering(df)

numeric_cols = df.select_dtypes(
    include="number"
).columns.tolist()

numeric_cols.remove(
    "length_of_stay_days"
)

df = preprocessor.cap_outliers(
    df,
    numeric_cols
)

X = df.drop(
    columns=["length_of_stay_days"]
)

y = df["length_of_stay_days"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

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

X_train = pd.DataFrame(
    X_train,
    columns=feature_names
)

X_test = pd.DataFrame(
    X_test,
    columns=feature_names
)

# =====================================================
# MODELS
# =====================================================

models = {

    "BayesianRidge":
        BayesianRidge(),
}

# =====================================================
# TRAIN ALL MODELS
# =====================================================

results = []

best_r2 = -9999
best_run_id = None
best_model_name = None
print("ML Flow Started")

for model_name, model in models.items():

    with mlflow.start_run(
        run_name=model_name
    ) as run:
        
        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(
            X_test
        )

        metrics = evaluate_model(
            y_test,
            y_pred
        )

        mlflow.log_params(
            model.get_params()
        )

        mlflow.log_metrics(
            metrics
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        results.append({
            "Model": model_name,
            **metrics
        })

        print(
            f"{model_name} R2={metrics['R2']:.4f}"
        )

        run_id = run.info.run_id

        

            
            

print("ML Flow Completed")



print("\nModel:", model_name)
print("R2:", metrics["R2"])
print("Run ID:", run_id)

# =====================================================
# REGISTER BEST MODEL
# =====================================================

model_uri = (
    f"runs:/{run_id}/model"
)

registered_model_name = (
    "HospitalLengthOfStayRegressor"
)

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=registered_model_name
)

print(
    f"Registered Model: "
    f"{registered_model_name}"
)

print(
    f"Version: "
    f"{registered_model.version}"
)

joblib.dump(
    model,
    f"best_saved_artifacts/{model_name}.pkl"
)


joblib.dump(
    preprocessor,
    f"best_saved_artifacts/preprocessor.pkl"
)

print(
    f"Best model saved as "
    f"best_saved_artifacts/{model_name}.pkl"
)

