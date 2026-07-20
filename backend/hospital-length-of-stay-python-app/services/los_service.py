import time
import logging

from datetime import datetime

from implementations.los_implementation import (
    LengthOfStayImplementation
)

from models.request_schema import (
    LengthOfStayPredictionRequest
)

from models.response_schema import (
    PredictionResponse
)

from implementations.mlflow_inference_tracker import (
    track_prediction_request,
    track_prediction_error
)

logger = logging.getLogger(__name__)


class LengthOfStayService:

    def __init__(self):

        self.impl = LengthOfStayImplementation()

    ##################################################
    # Predict Length Of Stay
    ##################################################

    def predict_length_of_stay(
        self,
        request: LengthOfStayPredictionRequest
    ) -> PredictionResponse:

        start_time = time.time()

        request_data = request.model_dump()

        try:

            result = self.impl.predict(
                request_data
            )

            execution_time = (
                time.time() - start_time
            )

            # =====================================
            # MLflow Inference Tracking
            # =====================================

            track_prediction_request(
                request_data=request_data,
                prediction_result=result,
                execution_time=execution_time
            )

            logger.info(
                f"LOS Prediction completed in "
                f"{execution_time:.4f}s"
            )

            return PredictionResponse(
                success=result["success"],
                predicted_length_of_stay_days=result[
                    "predicted_length_of_stay_days"
                ],
                model_name=result["model_name"],
                prediction_timestamp=datetime.utcnow(),
                message=result["message"]
            )

        except Exception as e:

            execution_time = (
                time.time() - start_time
            )

            logger.exception(
                "LOS Prediction Failed"
            )

            
            # =====================================
            # MLflow Error Tracking
            # =====================================

            track_prediction_error(
                request_data=request_data,
                error_message=str(e)
            )

            raise Exception(
                f"Prediction failed: {str(e)}"
            )