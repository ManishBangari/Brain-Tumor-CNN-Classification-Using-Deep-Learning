from pathlib import Path

import pytest

from src.BrainTumorCNNClassifier.pipeline.brain_tumor_pipeline import build_datasets, build_model, get_class_names


def test_dataset_shapes_are_valid():
    train_dir = Path("data/Training")
    class_names = get_class_names(train_dir)

    train_ds, val_ds, test_ds, loaded_names = build_datasets(
        train_dir=train_dir,
        test_dir=Path("data/Testing"),
        class_names=class_names,
        image_size=(224, 224),
        batch_size=16,
        validation_split=0.2,
        seed=42,
    )

    images, labels = next(iter(train_ds))
    assert images.shape[0] <= 16
    assert images.shape[1:] == (224, 224, 3)
    assert labels.shape[1] == len(loaded_names)
    assert len(val_ds.element_spec[0].shape) == 4
    assert len(test_ds.element_spec[0].shape) == 4


def test_model_outputs_expected_class_count():
    model = build_model(num_classes=4, input_shape=(224, 224, 3), learning_rate=1e-4)
    assert model.output_shape[-1] == 4
    assert model.optimizer.learning_rate.numpy() == pytest.approx(1e-4)
