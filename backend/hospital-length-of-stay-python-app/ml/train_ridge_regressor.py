from utils.preprocessing import DataPreprocessor
from sklearn.model_selection import train_test_split

preprocessor = DataPreprocessor()

import os
import joblib

os.makedirs(
    "saved_artifacts",
    exist_ok=True
)

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

from sklearn.linear_model import Ridge
from matplotlib import pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
import seaborn as sns
import numpy as np

class RidgeRegressorTrainer:

    def __init__(self):

        self.model = Ridge()           
        
        self.model_name = "RidgeRegressor"

    ##################################################
    # SMAPE
    ##################################################

    @staticmethod
    def smape(
        y_true,
        y_pred
    ):

        denominator = (
            np.abs(y_true)
            +
            np.abs(y_pred)
        ) / 2

        denominator = np.where(
            denominator == 0,
            1e-10,
            denominator
        )

        smape_value = np.mean(
            np.abs(
                y_true - y_pred
            ) / denominator
        ) * 100

        return smape_value

    ##################################################
    # RMSLE
    ##################################################

    @staticmethod
    def rmsle(
        y_true,
        y_pred
    ):

        y_true = np.maximum(
            y_true,
            0
        )

        y_pred = np.maximum(
            y_pred,
            0
        )

        return np.sqrt(

            np.mean(

                (
                    np.log1p(y_pred)
                    -
                    np.log1p(y_true)
                ) ** 2

            )

        )

    ##################################################
    # Train
    ##################################################
    def train(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        print("X_train Shape: ",X_train.shape)
        print("X_test Shape: ",X_test.shape)

        self.model.fit(
            X_train,
            y_train
        )

        y_pred = self.model.predict(
            X_test
        )

        results = self.evaluate(
            y_test,
            y_pred,
            X_test.shape[1]
        )

        return results

        
        
    

    

    ##################################################
    # Evaluation
    ##################################################

    def evaluate(
        self,
        y_true,
        y_pred,
        n_features
    ):
        print("\nNo. of features: ",n_features)

        mae = mean_absolute_error(
            y_true,
            y_pred
        )

        mse = mean_squared_error(
            y_true,
            y_pred
        )

        rmse = np.sqrt(mse)

        rmsle = self.rmsle(
            y_true,
            y_pred
        )

        mape = (
            mean_absolute_percentage_error(
                y_true,
                y_pred
            )
            * 100
        )

        smape = self.smape(
            y_true,
            y_pred
        )

        r2 = r2_score(
            y_true,
            y_pred
        )

        n = len(y_true)

        print("n: ",n)

        
        if n_features < n-1:

            adjusted_r2 = (

                1
                -
                (
                    (1-r2)
                    *
                    (n-1)
                )
                /
                (
                    n
                    -
                    n_features
                    -
                    1
                )

            )
        
        else:
            adjusted_r2 = None
            print(
                "Adjusted R² is not meaningful "
                "because n_features >= n - 1"
            )
        

        goodness_of_fit = max(
            0,
            r2 * 100
        )

        """
        print("\n")
        print("="*60)
        print("DUMMY REGRESSOR RESULTS")
        print("="*60)

        print(f"MAE               : {mae:.4f}")
        print(f"MSE               : {mse:.4f}")
        print(f"RMSE              : {rmse:.4f}")
        print(f"RMSLE             : {rmsle:.4f}")
        print(f"MAPE              : {mape:.2f}%")
        print(f"SMAPE             : {smape:.2f}%")

        print(f"R2 Score          : {r2:.4f}")
        print(f"Adjusted R2       : {adjusted_r2:.4f}")

        print(
            f"Goodness Of Fit   : "
            f"{goodness_of_fit:.2f}%"
        )

        print("="*60)
        """

        self.visualize(
            y_true,
            y_pred
        )

        metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "rmsle": rmsle,
            "mape": mape,
            "smape": smape,
            "r2": r2,
            "adjusted_r2": adjusted_r2,
            "goodness_of_fit": goodness_of_fit
        }

        return {
            "model_name": self.model_name,
            "metrics": metrics
        }

    ##################################################
    # Visualizations
    ##################################################

    def visualize(
        self,
        y_true,
        y_pred
    ):

        residuals = (
            y_true
            -
            y_pred
        )

        ##########################################
        # Actual vs Predicted
        ##########################################

        plt.figure(
            figsize=(8,6)
        )

        plt.scatter(
            y_true,
            y_pred,
            alpha=0.5
        )

        plt.plot(

            [
                y_true.min(),
                y_true.max()
            ],

            [
                y_true.min(),
                y_true.max()
            ],

            'r--'

        )

        plt.title(
            "Actual vs Predicted"
        )

        plt.xlabel(
            "Actual"
        )

        plt.ylabel(
            "Predicted"
        )

        #plt.show()
        plt.savefig(
            "saved_artifacts/ridge_actual_vs_predicted.png"
        )
        plt.close()

        ##########################################
        # Residual Plot
        ##########################################

        plt.figure(
            figsize=(8,6)
        )

        plt.scatter(
            y_pred,
            residuals,
            alpha=0.5
        )

        plt.axhline(
            y=0,
            color="red",
            linestyle="--"
        )

        plt.title(
            "Residual Plot"
        )

        plt.xlabel(
            "Predicted"
        )

        plt.ylabel(
            "Residuals"
        )

        #plt.show()
        plt.savefig(
            "saved_artifacts/ridge_residual_plot.png"
        )
        plt.close()

        ##########################################
        # Error Distribution
        ##########################################

        plt.figure(
            figsize=(8,6)
        )

        sns.histplot(
            residuals,
            kde=True
        )

        plt.title(
            "Error Distribution"
        )

        plt.xlabel(
            "Residual Error"
        )

        plt.ylabel(
            "Frequency"
        )

        #plt.show()
        plt.savefig(
            "saved_artifacts/ridge_error_distribution.png"
        )
        plt.close()

dr = RidgeRegressorTrainer()

print(
    "\nRidge REGRESSOR RESULTS"
)
results = dr.train(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test
)

print("\nModel Results")

print(
    f"Model: {results['model_name']}"
)

for metric, value in results["metrics"].items():

    if isinstance(value, (int, float, np.floating)):
        print(
            f"{metric.upper():<20}: {value:.4f}"
        )
    else:
        print(
            f"{metric.upper():<20}: {value}"
        )



joblib.dump(
    dr.model,
    "saved_artifacts/ridge_regressor.pkl"
)

