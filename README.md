# Brain Tumor MRI Image Classifier

An educational/research project for classifying brain MRI images into four
dataset labels with a locally trained TensorFlow CNN. It is **not a
clinically validated diagnostic system** and must not be used to make
medical decisions.

## Dataset

Use the required [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
(`masoudnickparvar/brain-tumor-mri-dataset`). Download/extract it locally so
the repository has this layout:

```text
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

The loader uses the class-directory names as labels and decodes image content
as RGB, then resizes images to 224 × 224. The current local copy was validated
with 5,600 Training and 1,600 Testing images. Counts can vary by dataset copy.
The dataset and its source files are local inputs; they are ignored by Git.

To download with Kaggle CLI, first install and configure Kaggle access locally
using Kaggle's instructions. Do not put credentials in this repository or
paste them into chat. Then run from the repository root:

```bash
python -m pip install kaggle
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data --unzip
```

If Kaggle access is unavailable, download the dataset manually from the linked
Kaggle page and extract it to the same `data/` layout.

## Environment setup

Python 3.12.3 and TensorFlow 2.16.2 were verified together in the project
virtual environment. Python 3.10.12 is also the version previously listed for
this project; other Python/library combinations have not been verified here.

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

The verified environment ran TensorFlow on CPU; no CUDA GPU was available.
No API credentials, external inference service, or `.env` file is needed.

## Configuration and workflow

Default paths and training parameters are in
[`config/config.yaml`](config/config.yaml): dataset directories, 224 × 224
input images, batch size 32, 20% validation split, seed 42, learning rate
0.0001, and 3 epochs. `params.yaml` contains corresponding basic run
parameters. Command-line options can override supported training values.

The training loader makes a deterministic, per-class validation split from
`Training`; only the remaining Training images are used for fitting. The
separate `Testing` split is reserved for evaluation. The CNN has three
convolution/pooling blocks and a four-unit softmax output in sorted class
order: `glioma`, `meningioma`, `notumor`, `pituitary`. Images are resized and
converted to RGB; the model rescales pixel values internally.

### Train

Choose a new output directory for each run. Training refuses to overwrite an
existing `checkpoints/best_model.keras` checkpoint.

```bash
python main.py train --epochs 3 --output-dir artifacts/run-001
```

Optional supported training arguments: `--batch-size`, `--image-size`
(square integer), `--learning-rate`, `--seed`, `--train-dir`, and `--test-dir`.
The default output directory is `artifacts/`.

Each run writes:

```text
artifacts/run-001/
├── checkpoints/best_model.keras
├── class_names.json
├── metrics.json
└── test_metrics.json
```

`metrics.json` stores epoch history and final validation accuracy.
`test_metrics.json` stores test loss, accuracy, precision, and recall. Testing
metrics are calculated after training; do not use them to tune the model.
Metrics/checkpoints are local artifacts and ignored by Git.

### Evaluate

Evaluation loads the checkpoint and evaluates the Testing split. Class names
are inferred from the Training directory, so keep the same dataset and class
folder names used for training.

```bash
python main.py evaluate \
  --model-path artifacts/run-001/checkpoints/best_model.keras \
  --train-dir data/Training \
  --test-dir data/Testing
```

The command prints test loss, accuracy, precision, and recall. It does not
generate a confusion matrix or per-class report.

### Predict one image

```bash
python main.py predict \
  --image data/Testing/glioma/Te-gl_114.jpg \
  --model-path artifacts/run-001/checkpoints/best_model.keras \
  --train-dir data/Training
```

Replace the example image with a path to an image you want to classify.
Prediction uses the same RGB decoding and resizing as dataset loading, loads
the class order from the Training folder names, and prints the predicted
label, confidence, and class probabilities.

## Scope and limitations

- There is no implemented Flask app, route, or UI launch command; the checked-in
  HTML template is not connected to a service.
- No automated test files are currently included.
- The existing checkpoint in `artifacts/checkpoints/` predates the corrected
  stratified validation split and should not be treated as a final model.
  Use a new output directory to train with the corrected loader.
- A one-epoch CPU smoke test completed, but no full training run has been
  performed. Smoke-test results are not a model benchmark.
- Dataset image counts, split balance, and quality should be rechecked for
  any different local dataset copy. No perceptual near-duplicate audit has
  been performed.
- This classifier has not been clinically validated and is not suitable for
  diagnosis or treatment decisions.
