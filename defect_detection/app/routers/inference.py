from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.schemas.defect import InferenceResult, StatsResponse
from app.services.classifier_service import run_inference, get_stats

router = APIRouter(prefix="/inference", tags=["Inference"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}


@router.post("/predict", response_model=InferenceResult)
async def predict(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG.",
        )

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image exceeds 10 MB limit.",
        )

    return await run_inference(image_bytes, file.filename, db)


@router.get("/stats", response_model=StatsResponse)
async def stats(
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
):
    return await get_stats(db, limit)
