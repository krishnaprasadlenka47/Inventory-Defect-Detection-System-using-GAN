import io
import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.ml.classifier.model import DefectCNN
from app.models.defect import InferenceLog
from app.schemas.defect import InferenceResult
from app.config import settings

LABEL_MAP = {0: "normal", 1: "defective"}

_classifier: DefectCNN | None = None

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(settings.image_size),
    transforms.CenterCrop(settings.image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def load_classifier() -> DefectCNN:
    global _classifier
    if _classifier is not None:
        return _classifier

    ckpt = os.path.join(settings.checkpoint_dir, "classifier.pt")
    if not os.path.exists(ckpt):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classifier checkpoint not found. Train first: python -m app.ml.classifier.train",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DefectCNN(num_classes=settings.num_classes)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    _classifier = model
    return _classifier


async def run_inference(image_bytes: bytes, filename: str, db: AsyncSession) -> InferenceResult:
    model = load_classifier()
    device = next(model.parameters()).device

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        confidence, predicted = probs.max(1)

    pred_idx = predicted.item()
    pred_class = LABEL_MAP[pred_idx]
    conf_score = round(confidence.item(), 4)

    log = InferenceLog(
        filename=filename,
        predicted_class=pred_class,
        confidence=conf_score,
        is_defective=(pred_idx == 1),
        model_version="v1",
    )
    db.add(log)
    await db.commit()
    await db.refresh(log)

    return InferenceResult(
        filename=filename,
        predicted_class=pred_class,
        confidence=conf_score,
        is_defective=(pred_idx == 1),
        model_version="v1",
    )


async def get_stats(db: AsyncSession, limit: int = 10):
    total = await db.scalar(select(func.count()).select_from(InferenceLog))
    defective = await db.scalar(
        select(func.count()).select_from(InferenceLog).where(InferenceLog.is_defective == True)
    )
    normal = total - defective
    defect_rate = round((defective / total * 100), 2) if total > 0 else 0.0

    recent = await db.execute(
        select(InferenceLog).order_by(InferenceLog.created_at.desc()).limit(limit)
    )
    recent_logs = recent.scalars().all()

    return {
        "total_inferences": total,
        "total_defective": defective,
        "total_normal": normal,
        "defect_rate_percent": defect_rate,
        "recent_logs": recent_logs,
    }
