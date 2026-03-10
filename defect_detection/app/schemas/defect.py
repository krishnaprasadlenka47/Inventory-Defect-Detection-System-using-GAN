from datetime import datetime
from pydantic import BaseModel, Field


class InferenceResult(BaseModel):
    filename: str
    predicted_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_defective: bool
    model_version: str

    model_config = {"from_attributes": True}


class InferenceLogOut(InferenceResult):
    id: int
    created_at: datetime


class GenerateRequest(BaseModel):
    label: int = Field(..., ge=0, le=1, description="0 = normal, 1 = defective")
    num_images: int = Field(default=5, ge=1, le=50)


class GenerateResponse(BaseModel):
    label: int
    label_name: str
    num_generated: int
    saved_paths: list[str]


class TrainingRunOut(BaseModel):
    id: int
    run_type: str
    epochs_completed: int
    final_g_loss: float | None
    final_d_loss: float | None
    final_accuracy: float | None
    checkpoint_path: str | None
    notes: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class StatsResponse(BaseModel):
    total_inferences: int
    total_defective: int
    total_normal: int
    defect_rate_percent: float
    recent_logs: list[InferenceLogOut]
