"""Strict public API schemas."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from common.schema import ALLOWED_VALUES


class CustomerFeatures(BaseModel):
    """Exactly the 19 raw customer features accepted by prediction."""
    model_config = ConfigDict(extra="forbid")
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: Optional[float] = Field(default=None, ge=0)

    @field_validator("gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod")
    @classmethod
    def validate_category(cls, value: str, info: object) -> str:
        """Keep API category domains identical to raw schema domains."""
        field_name = getattr(info, "field_name")
        stripped = value.strip()
        if stripped not in ALLOWED_VALUES[field_name]:
            raise ValueError(f"invalid value for {field_name}")
        return stripped

    @field_validator("SeniorCitizen")
    @classmethod
    def validate_senior_citizen(cls, value: int) -> int:
        """Allow only source-domain binary senior indicator values."""
        if value not in ALLOWED_VALUES["SeniorCitizen"]:
            raise ValueError("SeniorCitizen must be 0 or 1")
        return value
