import os
import joblib
from typing import Any


class ModelLoader:

    def __init__(
        self,
        model_path: str = "best_saved_artifacts/BayesianRidge.pkl",
        preprocessor_path: str = "best_saved_artifacts/preprocessor.pkl"
    ):

        self.model_path = model_path
        self.preprocessor_path = preprocessor_path

        self.model = None
        self.preprocessor = None

        self.load_artifacts()

    ##################################################
    # Load Artifacts
    ##################################################

    def load_artifacts(
        self
    ) -> None:

        if not os.path.exists(
            self.model_path
        ):
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        if not os.path.exists(
            self.preprocessor_path
        ):
            raise FileNotFoundError(
                f"Preprocessor not found: "
                f"{self.preprocessor_path}"
            )

        self.model = joblib.load(
            self.model_path
        )

        self.preprocessor = joblib.load(
            self.preprocessor_path
        )

        print(
            f"Model loaded from "
            f"{self.model_path}"
        )

        print(
            f"Preprocessor loaded from "
            f"{self.preprocessor_path}"
        )

    ##################################################
    # Get Model
    ##################################################

    def get_model(
        self
    ) -> Any:

        return self.model

    ##################################################
    # Get Preprocessor
    ##################################################

    def get_preprocessor(
        self
    ) -> Any:

        return self.preprocessor

    ##################################################
    # Predict
    ##################################################

    def predict(
        self,
        data
    ):

        transformed_data = (
            self.preprocessor.transform(
                data
            )
        )

        prediction = self.model.predict(
            transformed_data
        )

        return prediction