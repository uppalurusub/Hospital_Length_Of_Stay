import mlflow
import json
import traceback
from datetime import datetime


# =====================================================
# MLFLOW CONFIG
# =====================================================

mlflow.set_tracking_uri(
    "http://localhost:5000"
)

mlflow.set_experiment(
    "Hospital_LOS_Inference"
)


# =====================================================
# TRACK PREDICTION REQUEST
# =====================================================

def track_prediction_request(
    request_data: dict,
    prediction_result: dict,
    execution_time: float
):

    try:

        with mlflow.start_run(
            nested=True
        ):

            # ======================================
            # Params
            # ======================================

            for key, value in request_data.items():

                mlflow.log_param(
                    key,
                    str(value)
                )

            # ======================================
            # Metrics
            # ======================================

            mlflow.log_metric(
                "execution_time_seconds",
                float(execution_time)
            )

            if (
                "predicted_length_of_stay_days"
                in prediction_result
            ):

                mlflow.log_metric(
                    "predicted_length_of_stay_days",
                    float(
                        prediction_result[
                            "predicted_length_of_stay_days"
                        ]
                    )
                )

            # ======================================
            # Tags
            # ======================================

            mlflow.set_tag(
                "model_name",
                prediction_result.get(
                    "model_name",
                    "Unknown"
                )
            )

            mlflow.set_tag(
                "prediction_timestamp",
                datetime.utcnow().isoformat()
            )

            mlflow.set_tag(
                "prediction_status",
                "SUCCESS"
            )

            # ======================================
            # Artifacts
            # ======================================

            with open(
                "prediction_request.json",
                "w"
            ) as f:

                json.dump(
                    request_data,
                    f,
                    indent=4
                )

            mlflow.log_artifact(
                "prediction_request.json"
            )

    except Exception as e:

        print(
            f"Inference tracking failed: {e}"
        )


def track_prediction_error(
    request_data: dict,
    error_message: str
):

    try:

        with mlflow.start_run(
            run_name="LOS_Prediction_Error",
            nested=True
        ):

            # =====================================
            # Request Parameters
            # =====================================

            for key, value in request_data.items():

                try:

                    mlflow.log_param(
                        key,
                        str(value)
                    )

                except Exception:
                    pass

            # =====================================
            # Error Metrics
            # =====================================

            mlflow.log_metric(
                "prediction_success",
                0
            )

            # =====================================
            # Tags
            # =====================================

            mlflow.set_tag(
                "status",
                "FAILED"
            )

            mlflow.set_tag(
                "error_message",
                error_message[:500]
            )

            mlflow.set_tag(
                "timestamp",
                datetime.utcnow().isoformat()
            )

            # =====================================
            # Save Error Artifact
            # =====================================

            error_payload = {

                "request_data":
                    request_data,

                "error_message":
                    error_message,

                "timestamp":
                    datetime.utcnow().isoformat(),

                "traceback":
                    traceback.format_exc()

            }

            error_file = (
                "prediction_error.json"
            )

            with open(
                error_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    error_payload,
                    f,
                    indent=4,
                    default=str
                )

            mlflow.log_artifact(
                error_file
            )

            print(
                "Prediction error logged to MLflow."
            )

    except Exception as tracking_error:

        print(
            f"MLflow error tracking failed: "
            f"{tracking_error}"
        )