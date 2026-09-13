"""Streamlit inference app for the chest X-ray classification models."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
CONFIG_PATH = MODEL_DIR / "model_config.json"

st.set_page_config(page_title="Chest X-Ray Classifier", page_icon="🩻", layout="wide")

def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        :root { --ink:#142b3a; --muted:#607482; --teal:#0d8f8b; --teal-dark:#086a6b; --wash:#f4f8f9; --line:#dce8eb; }
        .stApp { background: linear-gradient(145deg,#f7fbfc 0%,#eef6f7 100%); color:var(--ink); font-family:'DM Sans',sans-serif; }
        .block-container { max-width:1180px; padding:2.5rem 2rem 4rem; }
        h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; color:var(--ink) !important; letter-spacing:-.025em; }
        h1 { font-size:2.5rem !important; margin-bottom:.25rem !important; }
        h2 { font-size:1.35rem !important; margin-top:1.4rem !important; }
        [data-testid='stSidebar'] { background:#102f3d; }
        [data-testid='stSidebar'] * { color:#eaf6f7 !important; }
        [data-testid='stSidebar'] hr { border-color:rgba(255,255,255,.18); }
        [data-testid='stSidebar'] [data-testid='stAlert'] { background:rgba(255,198,92,.14); border:1px solid rgba(255,198,92,.35); }
        [data-testid='stFileUploader'] { background:white; border:1.5px dashed #9dc4c8; border-radius:16px; padding:.4rem; box-shadow:0 8px 24px rgba(20,55,68,.06); }
        [data-testid='stFileUploaderDropzone'] { background:#f7fbfb; border-radius:12px; }
        [data-testid='stMetric'] { background:white; border:1px solid var(--line); border-radius:14px; padding:1rem 1.1rem; box-shadow:0 8px 24px rgba(20,55,68,.06); }
        [data-testid='stMetricLabel'] { color:var(--muted) !important; font-weight:600; }
        [data-testid='stMetricValue'] { color:var(--teal-dark) !important; font-family:'Space Grotesk',sans-serif; }
        [data-testid='stImage'] img { border-radius:14px; border:1px solid var(--line); box-shadow:0 8px 24px rgba(20,55,68,.06); }
        .eyebrow { color:var(--teal); text-transform:uppercase; letter-spacing:.16em; font-weight:700; font-size:.72rem; margin-bottom:.35rem; }
        .subtitle { color:var(--muted); font-size:1.04rem; margin-bottom:1.6rem; }
        .brand-mark { display:inline-flex; align-items:center; justify-content:center; width:42px; height:42px; border-radius:12px; background:#24b8ae; font-size:1.45rem; margin-bottom:.8rem; }
        .result-card { background:white; border:1px solid var(--line); border-radius:16px; padding:1.05rem 1.2rem; margin:.4rem 0 1.1rem; box-shadow:0 8px 24px rgba(20,55,68,.06); }
        .result-label { color:var(--muted); font-size:.78rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em; }
        .result-value { color:var(--teal-dark); font-family:'Space Grotesk',sans-serif; font-size:1.7rem; font-weight:700; margin-top:.2rem; }
        .pill { display:inline-block; color:#08716e; background:#e2f5f2; border:1px solid #bde5e1; border-radius:999px; padding:.28rem .65rem; font-size:.78rem; font-weight:700; }
        footer { visibility:hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def prepare_model_archive(filename: str) -> Path:
    """Keras checkpoints were supplied as ZIPs; expose a .keras copy to Keras."""
    source = MODEL_DIR / filename
    if not source.exists():
        raise FileNotFoundError(f"Model file not found: {source}")
    cache_dir = Path("/tmp/chest_xray_classifier_models")
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"{source.stem}.keras"
    if not destination.exists() or source.stat().st_mtime > destination.stat().st_mtime:
        shutil.copy2(source, destination)
    return destination

@st.cache_resource(show_spinner="Loading trained models…")
def load_models():
    import tensorflow as tf
    gatekeeper = tf.keras.models.load_model(prepare_model_archive("gatekeeper_best.zip"))
    classifier = tf.keras.models.load_model(prepare_model_archive("disease_best.zip"))
    return tf, gatekeeper, classifier

@st.cache_data
def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)

def make_input(image: Image.Image, image_size: tuple[int, int], tf):
    """Match the notebook's RGB resize and DenseNet preprocessing exactly."""
    rgb_image = image.convert("RGB")
    resized = rgb_image.resize(image_size)
    image_array = np.asarray(resized, dtype="float32")
    batch = np.expand_dims(image_array, axis=0)
    processed = tf.keras.applications.densenet.preprocess_input(batch.copy())
    return rgb_image, image_array, processed

def build_gradcam(tf, model, model_input: np.ndarray) -> np.ndarray:
    """Generate Grad-CAM for the final DenseNet convolutional layer."""
    layer_name = "conv5_block16_2_conv"
    grad_model = tf.keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        convolution, predictions = grad_model(model_input)
        class_index = tf.argmax(predictions[0])
        score = predictions[:, class_index]
    gradients = tape.gradient(score, convolution)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = convolution[0] @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def heatmap_overlay(image_array: np.ndarray, heatmap: np.ndarray, tf, alpha: float = 0.42):
    heatmap = tf.image.resize(heatmap[..., np.newaxis], image_array.shape[:2]).numpy()[..., 0]
    colors = np.array([[20, 35, 130], [0, 180, 220], [255, 220, 0], [225, 35, 25]])
    positions = np.linspace(0, 1, len(colors))
    colored = np.stack([np.interp(heatmap, positions, colors[:, channel]) for channel in range(3)], axis=-1)
    return np.uint8(np.clip((1 - alpha) * image_array + alpha * colored, 0, 255))

def main() -> None:
    inject_styles()
    config = load_config()
    image_size = tuple(config["img_size"])
    labels = {int(index): name for index, name in config["disease_idx_to_class"].items()}
    with st.sidebar:
        st.markdown('<div class="brand-mark">🩻</div>', unsafe_allow_html=True)
        st.markdown("### Chest X-Ray AI")
        st.caption("Clinical imaging screening workspace")
        st.divider()
        st.write("**How it works**")
        st.write("1. Checks whether the upload resembles a chest X-ray.")
        st.write("2. Classifies accepted images as COVID, NORMAL, or PNEUMONIA.")
        st.divider()
        st.warning("For education and research only — not medical advice or a diagnostic device.")
    st.markdown('<div class="eyebrow">AI-assisted radiology screening</div>', unsafe_allow_html=True)
    st.title("Chest X-Ray Classification")
    st.markdown('<div class="subtitle">A fast, two-stage assessment for chest radiographs — with transparent confidence and attention insights.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    show_gradcam = st.checkbox("Show Grad-CAM attention map", value=True)
    if uploaded_file is None:
        st.markdown('<div class="result-card"><span class="result-label">Ready for analysis</span><div style="margin-top:.45rem;color:#607482">Upload a PNG or JPEG chest radiograph above to begin screening.</div></div>', unsafe_allow_html=True)
        return
    try:
        original = Image.open(uploaded_file)
        original.load()
    except Exception:
        st.error("This file could not be read as an image. Please choose a valid PNG or JPEG.")
        return
    left, right = st.columns([1, 1.35], gap="large")
    with left:
        st.image(original, caption="Uploaded image", use_container_width=True)
    try:
        tf, gatekeeper, classifier = load_models()
    except ModuleNotFoundError:
        st.error("TensorFlow is not installed. Run `pip install -r requirements.txt`, then restart Streamlit.")
        return
    except Exception as error:
        st.error(f"The trained models could not be loaded: {error}")
        return
    _, resized_array, model_input = make_input(original, image_size, tf)
    with st.spinner("Analyzing image…"):
        xray_probability = float(gatekeeper.predict(model_input, verbose=0)[0][0])
    with right:
        st.subheader("Screening result")
        st.metric("Chest X-ray likelihood", f"{xray_probability:.1%}")
        st.progress(min(max(xray_probability, 0.0), 1.0))
    if xray_probability < float(config["gatekeeper_threshold"]):
        with right:
            st.warning("This image was not accepted as a chest X-ray, so no disease prediction was made.")
        return
    with st.spinner("Classifying X-ray…"):
        probabilities = classifier.predict(model_input, verbose=0)[0]
    prediction_index = int(np.argmax(probabilities))
    prediction = labels[prediction_index]
    confidence = float(probabilities[prediction_index])
    with right:
        st.success("Image accepted as a chest X-ray")
        st.markdown(f'<div class="result-card"><span class="result-label">Predicted condition</span><div class="result-value">{prediction}</div><span class="pill">{confidence:.1%} model confidence</span></div>', unsafe_allow_html=True)
    st.subheader("Class probabilities")
    st.bar_chart({labels[index]: float(probabilities[index]) for index in range(len(probabilities))}, horizontal=True)
    if show_gradcam:
        st.subheader("Model attention (Grad-CAM)")
        try:
            heatmap = build_gradcam(tf, classifier, model_input)
            overlay = heatmap_overlay(resized_array, heatmap, tf)
            attention, explanation = st.columns([1, 1.35], gap="large")
            with attention:
                st.image(overlay, caption=f"Grad-CAM for {prediction}", use_container_width=True)
            with explanation:
                st.caption("Warmer regions show areas that most influenced this prediction. They do not identify disease boundaries and are not for clinical interpretation.")
        except Exception as error:
            st.info(f"The attention map is unavailable for this model: {error}")

if __name__ == "__main__":
    main()
