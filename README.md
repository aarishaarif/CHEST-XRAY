# Chest X-Ray Classification
A deep learning based Chest X-Ray Classification system using DenseNet121 and Grad-CAM.

The system uses a two-stage classification pipeline:

## Gatekeeper Model — determines whether the uploaded image is a valid chest X-ray.

## Disease Classification Model — classifies valid chest X-rays into:
COVID
NORMAL
PNEUMONIA

## Features
DenseNet121-based image classification
Two-stage prediction pipeline
FastAPI REST API
Dockerized application
Web-based frontend
Image validation
Confidence scores and class probabilities
Grad-CAM visual explanations
Automated API tests
Health-check endpoint

## Model Architecture
## Stage 1 — X-Ray Gatekeeper
The first model checks whether the uploaded image is a valid chest X-ray.

If the confidence is below the configured threshold, the image is rejected and is not passed to the disease classifier.

Uploaded Image
      ↓
Gatekeeper Model
      ↓
Valid Chest X-Ray?
   ↙          ↘
 No            Yes
 ↓              ↓
Reject       Disease Model
                  ↓
          COVID / NORMAL / PNEUMONIA
The gatekeeper threshold is:

0.5

## Stage 2 — Disease Classification
If the image passes the gatekeeper, the disease classification model predicts one of three classes:

COVID
NORMAL
PNEUMONIA

This project trains and serves a two-stage DenseNet121 chest X-ray classifier.
The first (gatekeeper) model separates X-ray images from natural/non-X-ray images.
Only accepted images are passed to the disease classifier, which predicts
`COVID`, `NORMAL`, or `PNEUMONIA`.

The repository includes the original training notebook, exported Keras
checkpoints, and a Streamlit application for inference. The app follows the
notebook preprocessing exactly: RGB conversion, resize to 224×224, and DenseNet
`preprocess_input`.

## Repository contents

- `chest-xray-classification.ipynb` — dataset preparation, training, evaluation, Grad-CAM, and export workflow.
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
