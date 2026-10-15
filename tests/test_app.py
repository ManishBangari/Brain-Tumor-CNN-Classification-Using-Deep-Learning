import io
from pathlib import Path

from app import create_app


MODEL_PATH = Path("artifacts/checkpoints/best_model.keras")
CLASS_NAMES_PATH = Path("artifacts/class_names.json")


def test_invalid_upload_is_rejected():
    app = create_app(model_path=MODEL_PATH, class_names_path=CLASS_NAMES_PATH)
    client = app.test_client()

    response = client.post(
        "/api/predict",
        data={"file": (io.BytesIO(b"not an image"), "fake.txt")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert "error" in payload


def test_valid_upload_returns_prediction():
    app = create_app(model_path=MODEL_PATH, class_names_path=CLASS_NAMES_PATH)
    client = app.test_client()

    image_path = Path("data/Testing/glioma/Te-gl_1.jpg")
    response = client.post(
        "/api/predict",
        data={"file": (image_path.open("rb"), image_path.name)},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert "predicted_label" in payload
    assert "confidence" in payload
    assert "probabilities" in payload
    assert payload["predicted_label"] in ["glioma", "meningioma", "notumor", "pituitary"]
