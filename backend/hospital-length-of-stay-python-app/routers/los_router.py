from fastapi import (
    APIRouter,
    HTTPException,
    status
)

from services.los_service import (
    LengthOfStayService
)

from models.request_schema import (
    LengthOfStayPredictionRequest
)

from models.response_schema import (
    PredictionResponse
)

router = APIRouter(
    prefix="/los",
    tags=["Length Of Stay Prediction"]
)

service = LengthOfStayService()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Hospital Length Of Stay"
)
async def predict_length_of_stay(
    request: LengthOfStayPredictionRequest
):

    try:

        response = (
            service.predict_length_of_stay(
                request
            )
        )

        return response

    except Exception as e:

        raise HTTPException(
            status_code=
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )