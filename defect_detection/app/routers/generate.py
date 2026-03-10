from fastapi import APIRouter
from app.schemas.defect import GenerateRequest, GenerateResponse
from app.services.gan_service import generate_images, LABEL_MAP

router = APIRouter(prefix="/generate", tags=["GAN Generation"])


@router.post("/", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    paths = generate_images(request.label, request.num_images)
    return GenerateResponse(
        label=request.label,
        label_name=LABEL_MAP.get(request.label, str(request.label)),
        num_generated=len(paths),
        saved_paths=paths,
    )
