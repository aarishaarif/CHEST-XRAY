# Chest X-Ray Classification

This project trains and serves a two-stage DenseNet121 chest X-ray classifier.
The first (gatekeeper) model separates X-ray images from natural/non-X-ray images.
Only accepted images are passed to the disease classifier, which predicts
`COVID`, `NORMAL`, or `PNEUMONIA`.

The repository includes the original training notebook, exported Keras
checkpoints, and a Streamlit application for inference. The app follows the
notebook preprocessing exactly: RGB conversion, resize to 224×224, and DenseNet
`preprocess_input`.

## Repository contents

- `chest-xray-classification-ipynb.ipynb` — dataset preparation, training, evaluation, Grad-CAM, and export workflow.
- `model/gatekeeper_best.zip` — trained X-ray-vs-not-X-ray checkpoint.
- `model/disease_best.zip` — fine-tuned COVID/NORMAL/PNEUMONIA checkpoint.
- `model/model_config.json` — class-index, input-size, preprocessing, and threshold metadata.
- `app.py` — Streamlit inference UI.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

The supplied checkpoints are ZIP-form Keras archives. The app safely creates
temporary `.keras` copies under `/tmp` when it starts, so the original files are
never modified. Models are loaded lazily after an image is uploaded.

## Notebook workflow

The notebook builds the gatekeeper dataset from chest X-rays and sampled CIFAR-10
natural images. It trains a frozen-base gatekeeper, trains the disease model in
two phases (frozen base followed by partial fine-tuning), selects the stronger
disease checkpoint on validation accuracy, and exports `model_config.json`.
Its final prediction pipeline mirrors the Streamlit app: gatekeeper thresholding,
disease classification, and optional Grad-CAM.

The notebook’s Kaggle dataset path is environment-specific. To reproduce
training elsewhere, update `xray_dir` and provide `COVID`, `NORMAL`, and
`PNEUMONIA` folders before running the data-loading cells. Training requires a
TensorFlow environment and is not needed to run the supplied UI checkpoints.

