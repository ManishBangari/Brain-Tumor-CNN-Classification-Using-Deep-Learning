from __future__ import annotations

import json
from pathlib import Path

from src.BrainTumorCNNClassifier import logger
from src.BrainTumorCNNClassifier.pipeline.brain_tumor_pipeline import get_class_names


def validate_dataset(train_dir: str | Path = "data/Training", test_dir: str | Path = "data/Testing") -> dict:
    train_path = Path(train_dir)
    test_path = Path(test_dir)
    if not train_path.exists():
        raise FileNotFoundError(f"Training directory not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Testing directory not found: {test_path}")

    class_names = get_class_names(train_path)
    report = {"classes": class_names, "training": {}, "testing": {}}
    for class_name in class_names:
        train_class_dir = train_path / class_name
        test_class_dir = test_path / class_name
        t_files = sorted(p for p in train_class_dir.rglob("*") if p.is_file())
        te_files = sorted(p for p in test_class_dir.rglob("*") if p.is_file())
        report["training"][class_name] = len(t_files)
        report["testing"][class_name] = len(te_files)

    if set(report["training"]) != set(report["testing"]):
        raise ValueError("Training and testing class names differ.")

    for split_name, split_dir in (("training", train_path), ("testing", test_path)):
        for class_name in class_names:
            class_dir = split_dir / class_name
            for file_path in sorted(class_dir.iterdir()):
                if file_path.is_file() and file_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".gif"}:
                    try:
                        from PIL import Image

                        with Image.open(file_path) as img:
                            img.verify()
                    except Exception as exc:  # pragma: no cover - validation failure path
                        raise ValueError(f"Unreadable image at {file_path}: {exc}") from exc

    output_path = Path("data/validation_report.json")
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Validated dataset and wrote report to %s", output_path)
    return report


if __name__ == "__main__":
    validate_dataset()
