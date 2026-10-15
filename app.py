from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from src.BrainTumorCNNClassifier.pipeline.brain_tumor_pipeline import load_model_metadata, predict_single_image

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
MAX_IMAGE_BYTES = 16 * 1024 * 1024


def _resolve_model_path() -> Path:
    candidates = sorted(
        Path("artifacts").glob("run-*/checkpoints/best_model.keras"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    legacy = Path("artifacts/checkpoints/best_model.keras")
    if legacy.exists():
        return legacy
    raise FileNotFoundError("No trained model artifact was found. Train one with `python main.py train --output-dir artifacts/run-001`.")


def _resolve_class_names_path(model_path: Path | None = None) -> Path:
    model_path = model_path or _resolve_model_path()
    candidates = [
        Path("artifacts/class_names.json"),
        *(sorted(Path("artifacts").glob("run-*/class_names.json"), key=lambda item: item.stat().st_mtime, reverse=True)),
        model_path.with_suffix(".json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No class mapping found near {model_path}")


def _load_class_names(model_path: Path | None = None) -> list[str]:
    model_path = model_path or _resolve_model_path()
    class_names_path = _resolve_class_names_path(model_path)
    if class_names_path.exists():
        try:
            return json.loads(class_names_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    metadata = load_model_metadata(model_path)
    return metadata.get("class_names", [])


def create_app(model_path: str | Path | None = None, class_names_path: str | Path | None = None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES
    resolved_model_path = Path(model_path) if model_path else _resolve_model_path()
    resolved_class_names_path = Path(class_names_path) if class_names_path else _resolve_class_names_path(resolved_model_path)
    app.config["MODEL_PATH"] = str(resolved_model_path)
    app.config["CLASS_NAMES_PATH"] = str(resolved_class_names_path)

    @app.get("/")
    def index():
        return render_template("index.html", result=None, error=None)

    @app.get("/health")
    def health_check():
        return jsonify({"status": "ok", "model_path": app.config["MODEL_PATH"]})

    def handle_uploaded_image(file_storage):
        if file_storage is None or file_storage.filename == "":
            return None, "Please upload an MRI image before submitting."

        suffix = Path(file_storage.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            return None, "Unsupported image type. Use JPG, JPEG, PNG, BMP, or GIF."

        file_bytes = file_storage.read()
        if len(file_bytes) > MAX_IMAGE_BYTES:
            return None, "Uploaded file is too large. Please keep uploads under 16MB."

        try:
            with Image.open(io.BytesIO(file_bytes)) as image:
                image.verify()
            with Image.open(io.BytesIO(file_bytes)) as image:
                image = image.convert("RGB")
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                    image.save(temp_file.name)
                    temp_path = temp_file.name
        except UnidentifiedImageError:
            return None, "The uploaded file is not a valid image."
        except OSError:
            return None, "The uploaded image could not be processed."

        class_names = _load_class_names(resolved_model_path)
        try:
            result = predict_single_image(temp_path, resolved_model_path, class_names, image_size=(224, 224))
        except Exception as exc:  # pragma: no cover - error path handled by route
            return None, f"Prediction failed: {exc}"
        finally:
            Path(temp_path).unlink(missing_ok=True)

        return result, None

    @app.post("/predict")
    def predict_html():
        result, error = handle_uploaded_image(request.files.get("file"))
        return render_template("index.html", result=result, error=error)

    @app.post("/api/predict")
    def predict_api():
        result, error = handle_uploaded_image(request.files.get("file"))
        if error:
            return jsonify({"error": error}), 400
        return jsonify(result)

    return app


try:
    app = create_app()
except FileNotFoundError:
    app = Flask(__name__)

    @app.get("/")
    def index_fallback():
        return "No trained model artifact is available yet. Run `python main.py train --output-dir artifacts/run-001` first."

    @app.get("/health")
    def health_fallback():
        return jsonify({"status": "missing_model"})
