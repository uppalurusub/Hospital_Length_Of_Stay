from pydantic import BaseModel, Field
from typing import Optional


class LengthOfStayPredictionRequest(
    BaseModel
):
    age: int = Field(
        ...,
        gt=0,
        example=45
    )

    gender: str = Field(
        ...,
        example="Male"
    )

    admission_type: str = Field(
        ...,
        example="Emergency"
    )

    primary_diagnosis: str = Field(
        ...,
        example="Pneumonia"
    )

    comorbidity_count: int = Field(
        ...,
        ge=0,
        example=2
    )

    previous_admissions: int = Field(
        ...,
        ge=0,
        example=1
    )

    severity_score: float = Field(
        ...,
        ge=0,
        example=7.5
    )

    insurance_type: str = Field(
        ...,
        example="Private"
    )

    smoker: str = Field(
        ...,
        example="No"
    )

    bmi: float = Field(
        ...,
        gt=0,
        example=27.4
    )

    hospital_department: Optional[str] = Field(
        ...,
        example="Cardiology"
    )

    admission_source: Optional[str] = Field(
        ...,
        example="Referral"
    )

    discharge_disposition: Optional[str] = Field(
        default="Unknown"
    )

    marital_status: str

    heart_rate: float

    blood_pressure: float

    temperature: float

    oxygen_saturation: float

    glucose: float

    creatinine: float

    hemoglobin: float

    class Config:
        json_schema_extra = {
            "example": {
                "age": 45,
                "gender": "Male",
                "admission_type": "Emergency",
                "primary_diagnosis": "Pneumonia",
                "comorbidity_count": 2,
                "previous_admissions": 1,
                "severity_score": 7.5,
                "insurance_type": "Private",
                "smoker": "No",
                "bmi": 27.4,
                "hospital_department": "Cardiology",
                "admission_source": "Referral",
                "discharge_disposition": "Home"
            }
        }