import time
import pandas as pd

from typing import Dict

from utils.model_loader import ModelLoader


class LengthOfStayImplementation:

    def __init__(self):

        self.loader = ModelLoader()

        self.model = (
            self.loader.get_model()
        )

        self.preprocessor = (
            self.loader.get_preprocessor()
        )

    ##################################################
    # Predict LOS
    ##################################################

    def predict(
        self,
        patient_data: Dict
    ) -> Dict:

        start_time = time.time()

        try:

            input_df = pd.DataFrame(
                [patient_data]
            )

            input_df = (
                self.preprocessor.feature_engineering(
                    input_df
                )
            )

            

            transformed_data = (
                self.preprocessor.transform(
                    input_df
                )
            )

            feature_names = (
                self.preprocessor.preprocessor
                .get_feature_names_out()
            )

            transformed_data = pd.DataFrame(
                transformed_data,
                columns=feature_names
            )

            """
            expected_columns = set(
                self.preprocessor.feature_names_in_
            )

            received_columns = set(
                input_df.columns
            )

            missing_columns = (
                expected_columns
                -
                received_columns
            )

            print(
                "Missing Columns:",
                missing_columns
            )"""

            prediction = (
                self.model.predict(
                    transformed_data
                )
            )

            execution_time = (
                time.time() - start_time
            )

            return {

                "success": True,

                "predicted_length_of_stay_days":
                    round(
                        float(prediction[0]),
                        2
                    ),

                "model_name":
                    type(
                        self.model
                    ).__name__,

                "execution_time_seconds":
                    round(
                        execution_time,
                        4
                    ),

                "message":
                    "Prediction generated successfully"
            }

        except Exception as e:

            return {

                "success": False,

                "predicted_length_of_stay_days":
                    0.0,

                "model_name":
                    type(
                        self.model
                    ).__name__,

                "execution_time_seconds":
                    0,

                "message":
                    str(e)
            }