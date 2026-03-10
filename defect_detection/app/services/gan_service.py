import os
import uuid
import torch
from fastapi import HTTPException, status
from torchvision.utils import save_image

from app.ml.gan.cgan import Generator
from app.config import settings

LABEL_MAP = {0: "normal", 1: "defective"}

_generator: Generator | None = None


def load_generator() -> Generator:
    global _generator
    if _generator is not None:
        return _generator

    ckpt = os.path.join(settings.checkpoint_dir, "generator.pt")
    if not os.path.exists(ckpt):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Generator checkpoint not found. Train the cGAN first: python -m app.ml.gan.train",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator(settings.latent_dim, settings.num_classes, settings.image_size)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    _generator = model
    return _generator


def generate_images(label: int, num_images: int) -> list[str]:
    G = load_generator()
    device = next(G.parameters()).device

    os.makedirs(settings.synthetic_dir, exist_ok=True)
    label_name = LABEL_MAP.get(label, str(label))
    save_dir = os.path.join(settings.synthetic_dir, label_name)
    os.makedirs(save_dir, exist_ok=True)

    noise = torch.randn(num_images, settings.latent_dim, device=device)
    labels_tensor = torch.full((num_images,), label, dtype=torch.long, device=device)

    with torch.no_grad():
        fake_imgs = G(noise, labels_tensor)

    saved = []
    for i, img in enumerate(fake_imgs):
        filename = f"{label_name}_{uuid.uuid4().hex[:8]}.png"
        path = os.path.join(save_dir, filename)
        save_image(img, path, normalize=True)
        saved.append(path)

    return saved
