"""Pydantic v2 API schemas."""
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class PredictFeaturesRequest(BaseModel):
    features: list[float] = Field(..., min_length=19, max_length=19,
        description="19 colour features: r_mean g_mean b_mean r_std g_std b_std "
                    "optical_density r_norm g_norm b_norm g_over_r b_over_r g_over_rb "
                    "hue saturation value l_star a_star b_star")
    sex: str = "female"
    pregnant: bool = False
    trimester: Optional[int] = Field(None, ge=1, le=3)

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v: str) -> str:
        if v not in ("male","female"): raise ValueError("sex must be 'male' or 'female'")
        return v


class PredictResponse(BaseModel):
    hemoglobin_gdl: float
    severity: str
    who_threshold_gdl: float
    is_anemic: bool
    sex: str; pregnant: bool; trimester: Optional[int]
    recommendation: str; referral_required: bool
    treatment: list[str]; monitoring: str; color_code: str
    diagnostic_features: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str; model_loaded: bool; version: str; who_guideline: str
