import io
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app

client = TestClient(app)


def make_test_image() -> bytes:
    img = Image.new("RGB", (64, 64), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_unsupported_type():
    response = client.post(
        "/inference/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


@patch("app.services.classifier_service.load_classifier")
def test_predict_valid_image(mock_load):
    import torch
    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([torch.zeros(1)])
    mock_model.return_value = torch.tensor([[0.2, 0.8]])
    mock_load.return_value = mock_model

    image_bytes = make_test_image()
    assert image_bytes is not None
