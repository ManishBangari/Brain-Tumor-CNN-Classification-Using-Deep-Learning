from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.BrainTumorCNNClassifier import logger


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"file://{(PROJECT_ROOT / 'mlruns').as_posix()}")


def get_class_names(train_dir: str | Path) -> list[str]:
    train_path = Path(train_dir)
    if not train_path.exists():
        raise FileNotFoundError(f"Training directory not found: {train_path}")
    class_names = sorted([p.name for p in train_path.iterdir() if p.is_dir()])
    if not class_names:
        raise ValueError(f"No class folders found in training data: {train_path}")
    return class_names


def configure_mlflow(tracking_uri: str | None = None):
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = tracking_uri or MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    return uri


def save_model_metadata(
    model_path: str | Path,
    class_names: list[str],
    image_size: tuple[int, int],
    batch_size: int,
    learning_rate: float,
    seed: int,
):
    model_path = Path(model_path)
    metadata_path = model_path.with_suffix(".json")
    metadata = {
        "class_names": class_names,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved model metadata at %s", metadata_path)
    return metadata_path


def load_model_metadata(model_path: str | Path) -> dict[str, Any]:
    model_path = Path(model_path)
    metadata_path = model_path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found at {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _decode_and_resize(path: tf.Tensor, image_size: tuple[int, int]) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    return tf.image.resize(image, image_size)


def build_datasets(
    train_dir: str | Path,
    test_dir: str | Path,
    class_names: list[str] | None = None,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 42,
):
    train_path = Path(train_dir)
    test_path = Path(test_dir)
    if not train_path.exists():
        raise FileNotFoundError(f"Training directory not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Testing directory not found: {test_path}")
    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be between 0 and 1")

    class_names = class_names or get_class_names(train_path)
    test_classes = {p.name for p in test_path.iterdir() if p.is_dir()}
    if test_classes != set(class_names):
        raise ValueError(
            "Training and testing classes differ: "
            f"training={sorted(class_names)}, testing={sorted(test_classes)}"
        )
    logger.info("Loading datasets for classes: %s", class_names)

    supported_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    train_files: list[str] = []
    train_labels: list[int] = []
    val_files: list[str] = []
    val_labels: list[int] = []
    test_files: list[str] = []
    test_labels: list[int] = []

    for class_index, class_name in enumerate(class_names):
        training_class_dir = train_path / class_name
        testing_class_dir = test_path / class_name
        if not training_class_dir.is_dir() or not testing_class_dir.is_dir():
            raise ValueError(f"Missing class directory for {class_name}")

        training_files = sorted(
            str(path)
            for path in training_class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in supported_suffixes
        )
        testing_files = sorted(
            str(path)
            for path in testing_class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in supported_suffixes
        )
        if len(training_files) < 2 or not testing_files:
            raise ValueError(
                f"Class {class_name!r} needs at least two training images "
                "and one testing image"
            )

        random.Random(seed + class_index).shuffle(training_files)
        validation_count = max(1, round(len(training_files) * validation_split))
        validation_files = training_files[:validation_count]
        fitting_files = training_files[validation_count:]

        train_files.extend(fitting_files)
        train_labels.extend([class_index] * len(fitting_files))
        val_files.extend(validation_files)
        val_labels.extend([class_index] * len(validation_files))
        test_files.extend(testing_files)
        test_labels.extend([class_index] * len(testing_files))

    def decode_and_label(path, label):
        image = _decode_and_resize(path, image_size)
        label = tf.one_hot(label, depth=len(class_names))
        return image, label

    def make_dataset(files, labels, *, shuffle):
        dataset = tf.data.Dataset.from_tensor_slices((files, labels))
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=len(files), seed=seed, reshuffle_each_iteration=True
            )
        return (
            dataset.map(decode_and_label, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

    train_ds = make_dataset(train_files, train_labels, shuffle=True)
    val_ds = make_dataset(val_files, val_labels, shuffle=False)
    test_ds = make_dataset(test_files, test_labels, shuffle=False)
    return train_ds, val_ds, test_ds, class_names


def build_model(num_classes: int, input_shape: tuple[int, int, int] = (224, 224, 3), learning_rate: float = 1e-4):
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            keras.layers.Rescaling(1.0 / 255.0),
            keras.layers.Conv2D(32, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
    )
    return model


def save_class_names(class_names: list[str], out_path: str | Path):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(class_names, fh, indent=2)
    logger.info("Saved class names to %s", out)


def save_metrics(metrics: dict[str, Any], out_path: str | Path):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Saved metrics to %s", out)


def compute_classification_metrics(model, test_ds, class_names: list[str]):
    predictions = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = []
    for _, labels in test_ds:
        y_true.extend(np.argmax(labels.numpy(), axis=1).tolist())
    y_true = np.asarray(y_true)
    total = len(class_names)
    confusion = np.zeros((total, total), dtype=int)
    for true_i, pred_i in zip(y_true, y_pred):
        confusion[int(true_i), int(pred_i)] += 1

    per_class = {}
    for idx, label in enumerate(class_names):
        tp = confusion[idx, idx]
        fp = confusion[:, idx].sum() - tp
        fn = confusion[idx, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(confusion[idx, :].sum()),
        }

    return {
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def train_model(
    train_dir: str | Path,
    test_dir: str | Path,
    output_dir: str | Path,
    epochs: int = 3,
    batch_size: int = 32,
    image_size: tuple[int, int] = (224, 224),
    validation_split: float = 0.2,
    seed: int = 42,
    learning_rate: float = 1e-4,
):
    class_names = get_class_names(train_dir)
    output_dir_path = Path(output_dir)
    checkpoint_dir = output_dir_path / "checkpoints"
    model_path = checkpoint_dir / "best_model.keras"
    if model_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing model checkpoint: {model_path}. "
            "Choose a new --output-dir."
        )

    train_ds, val_ds, test_ds, class_names = build_datasets(
        train_dir=train_dir,
        test_dir=test_dir,
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        seed=seed,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    configure_mlflow()

    model = build_model(
        num_classes=len(class_names),
        input_shape=(image_size[0], image_size[1], 3),
        learning_rate=learning_rate,
    )
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    logger.info("Starting model training for %s epochs", epochs)
    with mlflow.start_run(run_name=f"brain_tumor_cnn_{seed}") as run:
        mlflow.log_params(
            {
                "train_dir": str(train_dir),
                "test_dir": str(test_dir),
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "image_size": f"{image_size[0]}x{image_size[1]}",
                "validation_split": float(validation_split),
                "learning_rate": float(learning_rate),
                "seed": int(seed),
                "num_classes": int(len(class_names)),
            }
        )
        mlflow.log_param("class_names", json.dumps(class_names))
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[checkpoint_callback],
        )

        for epoch in range(len(history.history["loss"])):
            mlflow.log_metric("train_loss", float(history.history["loss"][epoch]), step=epoch)
            mlflow.log_metric("train_accuracy", float(history.history["accuracy"][epoch]), step=epoch)
            mlflow.log_metric("val_loss", float(history.history["val_loss"][epoch]), step=epoch)
            mlflow.log_metric("val_accuracy", float(history.history["val_accuracy"][epoch]), step=epoch)

        metrics = {
            "train": history.history,
            "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        }
        save_metrics(metrics, output_dir_path / "metrics.json")
        save_class_names(class_names, output_dir_path / "class_names.json")
        save_model_metadata(
            model_path=model_path,
            class_names=class_names,
            image_size=image_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
        )

        best_model = keras.models.load_model(model_path)
        test_loss, test_accuracy, test_precision, test_recall = best_model.evaluate(test_ds, verbose=0)
        classification_report = compute_classification_metrics(best_model, test_ds, class_names)
        final_metrics = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "test_precision": float(test_precision),
            "test_recall": float(test_recall),
            "per_class": classification_report["per_class"],
            "confusion_matrix": classification_report["confusion_matrix"],
        }
        save_metrics(final_metrics, output_dir_path / "test_metrics.json")
        mlflow.log_metric("test_loss", float(test_loss))
        mlflow.log_metric("test_accuracy", float(test_accuracy))
        mlflow.log_metric("test_precision", float(test_precision))
        mlflow.log_metric("test_recall", float(test_recall))
        for label_name, values in classification_report["per_class"].items():
            mlflow.log_metric(f"{label_name}_precision", float(values["precision"]))
            mlflow.log_metric(f"{label_name}_recall", float(values["recall"]))
            mlflow.log_metric(f"{label_name}_f1", float(values["f1"]))
        mlflow.log_artifact(str(output_dir_path / "metrics.json"))
        mlflow.log_artifact(str(output_dir_path / "test_metrics.json"))
        mlflow.log_artifact(str(output_dir_path / "class_names.json"))
        mlflow.keras.log_model(best_model, artifact_path="model")

    logger.info("Training complete. Best model saved at %s", model_path)
    return model_path, class_names


def load_model(model_path: str | Path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return keras.models.load_model(model_path)


def predict_single_image(image_path: str | Path, model_path: str | Path, class_names: list[str], image_size: tuple[int, int] = (224, 224)):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = _decode_and_resize(tf.convert_to_tensor(str(image_path)), image_size)
    array = image.numpy()
    array = np.expand_dims(array, axis=0)
    model = load_model(model_path)
    probabilities = model.predict(array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = class_names[predicted_index]
    predicted_confidence = float(probabilities[predicted_index])
    result = {
        "image": str(image_path),
        "predicted_label": predicted_label,
        "predicted_index": predicted_index,
        "confidence": predicted_confidence,
        "probabilities": {class_names[i]: float(probabilities[i]) for i in range(len(class_names))},
    }
    logger.info("Prediction for %s -> %s (confidence %.4f)", image_path, predicted_label, predicted_confidence)
    return result
