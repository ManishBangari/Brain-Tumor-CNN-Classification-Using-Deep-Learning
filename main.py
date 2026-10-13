from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.BrainTumorCNNClassifier import logger
from src.BrainTumorCNNClassifier.pipeline.brain_tumor_pipeline import (
    get_class_names,
    predict_single_image,
    train_model,
)
from src.BrainTumorCNNClassifier.utils.commonFile import read_yaml


def load_config():
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return read_yaml(config_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Brain tumor MRI classifier workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the CNN on the Training split")
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--image-size", type=int, default=None)
    train_parser.add_argument("--learning-rate", type=float, default=None)
    train_parser.add_argument("--seed", type=int, default=None)
    train_parser.add_argument("--train-dir", type=str, default=None)
    train_parser.add_argument("--test-dir", type=str, default=None)
    train_parser.add_argument("--output-dir", type=str, default="artifacts")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the saved model on the Testing split")
    eval_parser.add_argument("--model-path", type=str, default="artifacts/checkpoints/best_model.keras")
    eval_parser.add_argument("--test-dir", type=str, default="data/Testing")
    eval_parser.add_argument("--train-dir", type=str, default="data/Training")

    predict_parser = subparsers.add_parser("predict", help="Predict a single MRI image")
    predict_parser.add_argument("--image", type=str, required=True)
    predict_parser.add_argument("--model-path", type=str, default="artifacts/checkpoints/best_model.keras")
    predict_parser.add_argument("--train-dir", type=str, default="data/Training")

    return parser


def run_training(args: argparse.Namespace):
    config = load_config()
    train_dir = args.train_dir or config.paths.train_dir
    test_dir = args.test_dir or config.paths.test_dir
    epochs = args.epochs if args.epochs is not None else config.model.epochs
    batch_size = args.batch_size if args.batch_size is not None else config.model.batch_size
    image_size = (args.image_size, args.image_size) if args.image_size is not None else tuple(config.model.image_size)
    learning_rate = args.learning_rate if args.learning_rate is not None else config.model.learning_rate
    seed = args.seed if args.seed is not None else config.model.seed

    model_path, class_names = train_model(
        train_dir=train_dir,
        test_dir=test_dir,
        output_dir=args.output_dir,
        epochs=epochs,
        batch_size=batch_size,
        image_size=image_size,
        validation_split=config.model.validation_split,
        seed=seed,
        learning_rate=learning_rate,
    )
    logger.info("Training complete; best checkpoint saved at %s", model_path)
    print(json.dumps({"model_path": str(model_path), "class_names": class_names}, indent=2))


def run_evaluation(args: argparse.Namespace):
    config = load_config()
    class_names = get_class_names(args.train_dir or config.paths.train_dir)
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    from src.BrainTumorCNNClassifier.pipeline.brain_tumor_pipeline import build_datasets, load_model

    _, _, test_ds, _ = build_datasets(
        train_dir=args.train_dir or config.paths.train_dir,
        test_dir=args.test_dir or config.paths.test_dir,
        class_names=class_names,
        image_size=tuple(config.model.image_size),
        batch_size=config.model.batch_size,
        validation_split=config.model.validation_split,
        seed=config.model.seed,
    )
    model = load_model(model_path)
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_ds, verbose=0)
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
    }
    print(json.dumps(metrics, indent=2))


def run_prediction(args: argparse.Namespace):
    config = load_config()
    class_names = get_class_names(args.train_dir or config.paths.train_dir)
    result = predict_single_image(
        image_path=args.image,
        model_path=args.model_path,
        class_names=class_names,
        image_size=tuple(config.model.image_size),
    )
    print(json.dumps(result, indent=2))


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_training(args)
    elif args.command == "evaluate":
        run_evaluation(args)
    elif args.command == "predict":
        run_prediction(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
