import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import gradio as gr
from PIL import Image

from inference.inference_engine import BoneMDInferenceEngine
from utils.xray_loader import load_xray
from utils.xray_preprocess import preprocess_xray
from utils.ct_loader import load_ct_volume
from utils.ct_preprocess import preprocess_ct


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

TITLE = "BoneMD-Net"

TEST_CACHE = PROJECT_ROOT / "cache" / "test"

CLASS_NAMES = {
    0: "Normal",
    1: "Osteopenia",
    2: "Osteoporosis",
}


# ------------------------------------------------------------
# Load model once
# ------------------------------------------------------------

print("Loading BoneMD-Net inference engine...")

ENGINE = BoneMDInferenceEngine()

print(
    f"Model loaded on {ENGINE.device}, "
    f"checkpoint epoch {ENGINE.checkpoint_epoch}"
)


# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------

def load_preview(zip_path, dicom_path):
    image, _ = load_xray(
        zip_path,
        dicom_path,
    )

    image = preprocess_xray(image)

    image = (
        np.clip(image, 0.0, 1.0) * 255
    ).astype(np.uint8)

    return Image.fromarray(
        image,
        mode="L",
    )


def load_uploaded_xray_preview(dicom_path):
    if dicom_path is None:
        return None

    import pydicom

    ds = pydicom.dcmread(dicom_path)
    image = ds.pixel_array.astype(np.float32)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        bits_stored = int(ds.BitsStored)
        max_value = float((1 << bits_stored) - 1)
        image = max_value - image

    image = preprocess_xray(image)

    image = (
        np.clip(image, 0.0, 1.0) * 255
    ).astype(np.uint8)

    return Image.fromarray(image, mode="L")


def predict_uploaded_images(
    ap_path,
    lateral_path,
    ct_zip_path,
    ct_patient_id,
):
    if ap_path is None or lateral_path is None or ct_zip_path is None:
        raise ValueError(
            "Please upload AP X-ray, Lateral X-ray, and CT ZIP."
        )

    import pydicom
    import torch
    import zipfile

    def prepare_xray(dicom_path):
        ds = pydicom.dcmread(dicom_path)
        image = ds.pixel_array.astype(np.float32)

        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            bits_stored = int(ds.BitsStored)
            max_value = float((1 << bits_stored) - 1)
            image = max_value - image

        image = preprocess_xray(image)
        return torch.from_numpy(image).float().unsqueeze(0)

    if ct_patient_id is None:
        raise ValueError(
            "Please enter the CT Patient ID."
        )

    ct_patient_id = int(ct_patient_id)

    ct_folder = f"lumos_ct_{ct_patient_id:03d}"

    with zipfile.ZipFile(ct_zip_path, "r") as archive:

        prefix = ct_folder + "/"

        dicom_files = [
            name
            for name in archive.namelist()
            if name.startswith(prefix)
            and name.lower().endswith(".dcm")
            and "__macosx" not in name.lower()
        ]

    if not dicom_files:
        raise ValueError(
            f"No CT DICOM files found for Patient ID "
            f"{ct_patient_id} ({ct_folder}) in the uploaded ZIP."
        )

    ct_volume, ct_metadata = load_ct_volume(
        ct_zip_path,
        ct_folder,
    )
    ct_volume = preprocess_ct(
        ct_volume,
        ct_metadata,
    )

    ap = prepare_xray(ap_path).unsqueeze(0)
    lateral = prepare_xray(lateral_path).unsqueeze(0)
    ct = torch.from_numpy(ct_volume).float().unsqueeze(0).unsqueeze(0)

    ap = ap.to(ENGINE.device)
    lateral = lateral.to(ENGINE.device)
    ct = ct.to(ENGINE.device)

    with torch.no_grad():
        outputs = ENGINE.model(
            ap,
            lateral,
            ct,
        )

        probabilities = torch.softmax(
            outputs["logits"],
            dim=1,
        )[0]

        predicted_class = torch.argmax(
            probabilities
        ).item()

    return {
        "predicted_class": predicted_class,
        "predicted_name": CLASS_NAMES[predicted_class],
        "probabilities": {
            CLASS_NAMES[i]: float(
                probabilities[i].item()
            )
            for i in range(3)
        },
        "feature_dimensions": {
            "AP X-ray": int(outputs["ap_features"].shape[-1]),
            "Lateral X-ray": int(outputs["lateral_features"].shape[-1]),
            "CT": int(outputs["ct_features"].shape[-1]),
            "Fused representation": int(outputs["fused_features"].shape[-1]),
        },
    }


# ------------------------------------------------------------
# Visual result formatting
# ------------------------------------------------------------

def format_probability_cards(probabilities):
    if not probabilities:
        return """
        <div class="bm-empty-result">
            Run an analysis to view class probabilities.
        </div>
        """

    order = [
        ("Normal", "bm-normal"),
        ("Osteopenia", "bm-osteopenia"),
        ("Osteoporosis", "bm-osteoporosis"),
    ]

    html = """
    <div class="bm-metric-list">
    """

    for name, css_class in order:
        value = float(probabilities.get(name, 0.0))
        percent = value * 100.0

        html += f"""
        <div class="bm-metric">
            <div class="bm-metric-header">
                <span>{name}</span>
                <strong>{percent:.1f}%</strong>
            </div>

            <div class="bm-progress-track">
                <div
                    class="bm-progress-fill {css_class}"
                    style="width:{percent:.1f}%"
                ></div>
            </div>
        </div>
        """

    html += "</div>"

    return html


def format_feature_cards(feature_dimensions):
    if not feature_dimensions:
        return """
        <div class="bm-empty-result">
            Run an analysis to view feature representations.
        </div>
        """

    html = """
    <div class="bm-metric-list">
    """

    for label, dimension in feature_dimensions.items():
        html += f"""
        <div class="bm-metric">
            <div class="bm-metric-header">
                <span>{label}</span>
                <strong>{int(dimension)}-D</strong>
            </div>
        </div>
        """

    html += "</div>"

    return html


def analyze_uploaded_images(
    ap_path,
    lateral_path,
    ct_zip_path,
    ct_patient_id,
):
    result = predict_uploaded_images(
        ap_path,
        lateral_path,
        ct_zip_path,
        ct_patient_id,
    )

    prediction_text = (
        f"{result['predicted_name']} "
        f"(Class {result['predicted_class']})"
    )

    return (
        prediction_text,
        format_probability_cards(
            result["probabilities"]
        ),
        format_feature_cards(
            result["feature_dimensions"]
        ),
    )


def ct_slice_to_image(volume, slice_index):
    slice_index = int(
        np.clip(
            slice_index,
            0,
            volume.shape[0] - 1,
        )
    )

    slice_image = volume[slice_index]

    # Visualization only.
    low = -1000.0
    high = 1000.0

    slice_image = np.clip(
        slice_image,
        low,
        high,
    )

    slice_image = (
        (slice_image - low)
        / (high - low)
        * 255.0
    ).astype(np.uint8)

    return Image.fromarray(
        slice_image,
        mode="L",
    )


def load_ct_volume_for_preview(zip_path, patient_folder):
    volume, metadata = load_ct_volume(
        zip_path,
        patient_folder,
    )

    return preprocess_ct(
        volume,
        metadata,
    )


def load_uploaded_ct_preview(zip_path, patient_id):
    if zip_path is None:
        return None, None

    if patient_id is None:
        raise ValueError(
            "Please enter the CT Patient ID."
        )

    import zipfile

    patient_id = int(patient_id)
    patient_folder = f"lumos_ct_{patient_id:03d}"

    with zipfile.ZipFile(zip_path, "r") as archive:
        prefix = patient_folder + "/"

        dicom_files = [
            name
            for name in archive.namelist()
            if name.startswith(prefix)
            and name.lower().endswith(".dcm")
            and "__macosx" not in name.lower()
        ]

    if not dicom_files:
        raise ValueError(
            f"No CT DICOM files found for Patient ID {patient_id} "
            f"({patient_folder}) in the uploaded ZIP."
        )

    volume = load_ct_volume_for_preview(
        zip_path,
        patient_folder,
    )

    return (
        ct_slice_to_image(volume, 96),
        volume,
    )


def update_ct_slice(volume, slice_index):
    if volume is None:
        return None

    return ct_slice_to_image(
        volume,
        slice_index,
    )


def analyze_patient(patient_id):

    empty = (
        None,
        None,
        None,
        None,
        None,
        "Enter a patient ID.",
        format_probability_cards({}),
        format_feature_cards({}),
        "No patient selected.",
    )

    if patient_id is None:
        return empty

    try:
        patient_id = int(patient_id)
    except (TypeError, ValueError):
        return (
            None,
            None,
            None,
            None,
            None,
            "Invalid patient ID.",
            format_probability_cards({}),
            format_feature_cards({}),
            "Please enter a numeric patient ID.",
        )

    try:
        # Use the cached multimodal test patient.
        sample = ENGINE.find_patient(patient_id)

        # Generate display images from the cached tensors.
        ap_array = (
            sample["ap"]
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )

        lateral_array = (
            sample["lateral"]
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )

        ap_image = Image.fromarray(
            np.uint8(
                np.clip(ap_array, 0.0, 1.0) * 255.0
            )
        )

        lateral_image = Image.fromarray(
            np.uint8(
                np.clip(lateral_array, 0.0, 1.0) * 255.0
            )
        )

        # Cached CT tensor is [1, D, H, W].
        ct_volume = (
            sample["ct"]
            .detach()
            .cpu()
            .numpy()
            .squeeze(0)
        )

        slice_index = min(
            96,
            ct_volume.shape[0] - 1,
        )

        ct_image = ct_slice_to_image(
            ct_volume,
            slice_index,
        )

        # Run the final BoneMDTeacherNorm model.
        result = ENGINE.predict_patient(
            patient_id
        )

    except Exception as exc:

        return (
            None,
            None,
            None,
            None,
            None,
            "Inference error",
            format_probability_cards({}),
            format_feature_cards({}),
            f"{type(exc).__name__}: {exc}",
        )

    prediction = (
        f"{result['predicted_name']} "
        f"(Class {result['predicted_class']})"
    )

    true_name = result.get(
        "true_name",
        "Unknown",
    )

    status = (
        f"Patient {patient_id} analyzed successfully. "
        f"True label: {true_name}. "
        f"Final teacher checkpoint: epoch "
        f"{result['checkpoint_epoch']}."
    )

    return (
        ap_image,
        lateral_image,
        ct_image,
        ct_volume,
        f"Slice **{slice_index + 1} / {ct_volume.shape[0]}**",
        prediction,
        format_probability_cards(
            result["probabilities"]
        ),
        format_feature_cards({
            "AP X-ray": result["ap_features"][-1],
            "Lateral X-ray": result["lateral_features"][-1],
            "CT": result["ct_features"][-1],
            "Fused representation": result["fused_features"][-1],
        }),
        status,
    )


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------

CUSTOM_CSS = """
/* ============================================================
   BoneMD-Net — Research Dashboard
   ============================================================ */

:root {
    --bm-navy: #0f172a;
    --bm-blue: #2563eb;
    --bm-blue-soft: #eff6ff;
    --bm-border: #e2e8f0;
    --bm-muted: #64748b;
    --bm-bg: #f8fafc;
    --bm-card: #ffffff;
}

body {
    background: var(--bm-bg) !important;
}

.gradio-container {
    max-width: 1500px !important;
    margin: auto !important;
    padding: 24px 28px 48px !important;
}

/* ------------------------------------------------------------
   Header
   ------------------------------------------------------------ */

.bm-header {
    background:
        linear-gradient(
            135deg,
            #0f172a 0%,
            #172554 55%,
            #1d4ed8 100%
        );
    border-radius: 22px;
    padding: 30px 34px;
    margin-bottom: 22px;
    color: white;
    box-shadow: 0 12px 35px rgba(15, 23, 42, 0.16);
}

.bm-header-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 20px;
}

.bm-brand {
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -0.8px;
    margin: 0;
    color: #ffffff !important;
}

.bm-header .bm-brand,
.bm-header .bm-brand * {
    color: #ffffff !important;
}

.bm-tagline {
    margin-top: 7px;
    font-size: 15px;
    color: rgba(255,255,255,0.86) !important;
    line-height: 1.5;
}

.bm-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    white-space: nowrap;
}

.bm-status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #4ade80;
}

/* ------------------------------------------------------------
   Section cards
   ------------------------------------------------------------ */

.bm-card {
    background: var(--bm-card);
    border: 1px solid var(--bm-border);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 5px 18px rgba(15, 23, 42, 0.045);
    margin-bottom: 18px;
}

.bm-card-title {
    color: var(--bm-navy);
    font-size: 17px;
    font-weight: 750;
    margin-bottom: 3px;
}

.bm-card-subtitle {
    color: var(--bm-muted);
    font-size: 12px;
    margin-bottom: 17px;
}

/* ------------------------------------------------------------
   Main workspace alignment
   ------------------------------------------------------------ */

.bm-workspace {
    align-items: flex-start !important;
}

.bm-workspace > div {
    align-self: flex-start !important;
}

.bm-result-panel {
    align-self: flex-start !important;
    background:
        linear-gradient(
            180deg,
            #ffffff 0%,
            #fbfdff 100%
        ) !important;
}

.bm-result-panel .gr-label {
    min-height: 0 !important;
}

.bm-result-panel .block {
    margin-bottom: 12px !important;
}

.bm-result-panel h4 {
    margin-top: 12px !important;
    margin-bottom: 8px !important;
}

/* ------------------------------------------------------------
   Mode labels
   ------------------------------------------------------------ */

.bm-mode {
    background: #f8fafc;
    border: 1px solid var(--bm-border);
    border-radius: 12px;
    padding: 11px 13px;
    margin-bottom: 12px;
}

.bm-mode-title {
    font-size: 12px;
    font-weight: 750;
    color: var(--bm-navy);
}

.bm-mode-text {
    font-size: 11px;
    color: var(--bm-muted);
    margin-top: 3px;
}

.bm-workspace .bm-mode {
    margin-bottom: 9px !important;
}

.bm-workspace .gr-number {
    margin-bottom: 8px !important;
}

.bm-workspace button {
    margin-top: 4px !important;
}

/* ------------------------------------------------------------
   Research result metrics
   ------------------------------------------------------------ */

.bm-result-html {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}

.bm-metric-list {
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 4px 0 8px;
}

.bm-metric {
    width: 100%;
}

.bm-metric-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 7px;
    font-size: 13px;
    color: #334155;
}

.bm-metric-header strong {
    font-size: 14px;
    color: #0f172a;
}

.bm-progress-track {
    width: 100%;
    height: 9px;
    background: #e8edf5;
    border-radius: 999px;
    overflow: hidden;
}

.bm-progress-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.35s ease;
}

.bm-normal {
    background: #64748b;
}

.bm-osteopenia {
    background: #f59e0b;
}

.bm-osteoporosis {
    background: #dc2626;
}

.bm-fusion {
    background: #2563eb;
}

.bm-empty-result {
    padding: 14px 0;
    color: #94a3b8;
    font-size: 13px;
}

/* ------------------------------------------------------------
   Prediction
   ------------------------------------------------------------ */

.bm-prediction {
    background:
        linear-gradient(
            145deg,
            #f8fbff,
            #eef4ff
        );
    border: 1px solid #dbe7ff;
    border-radius: 16px;
    padding: 17px;
}

.bm-prediction-label {
    font-size: 11px;
    color: var(--bm-muted);
    text-transform: uppercase;
    letter-spacing: 0.7px;
    font-weight: 700;
}

.bm-prediction-value {
    font-size: 28px;
    font-weight: 800;
    color: var(--bm-navy);
    margin-top: 6px;
}

.bm-prediction-box textarea,
.bm-prediction-box input {
    font-size: 22px !important;
    font-weight: 800 !important;
    color: #0f172a !important;
    min-height: 62px !important;
}

.bm-prediction-box label {
    font-weight: 750 !important;
    color: #334155 !important;
}

/* ------------------------------------------------------------
   Image evidence
   ------------------------------------------------------------ */

.bm-evidence-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 24px 2px 12px;
    padding: 0 4px;
}

.bm-evidence-title {
    color: #0f172a;
    font-size: 21px;
    font-weight: 800;
    letter-spacing: -0.3px;
}

.bm-evidence-subtitle {
    color: #64748b;
    font-size: 12px;
    margin-top: 3px;
}

.bm-evidence-badge {
    padding: 6px 10px;
    border-radius: 8px;
    background: #eff6ff;
    border: 1px solid #dbeafe;
    color: #2563eb;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.7px;
}

/* Image cards */

.bm-image-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 10px;
    overflow: hidden;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
}

.bm-image-card .label-wrap {
    padding: 4px 6px 8px !important;
}

.bm-image-card label {
    font-size: 12px !important;
    font-weight: 750 !important;
    color: #334155 !important;
}

.bm-image {
    background: #020617 !important;
    border: 1px solid #1e293b !important;
    border-radius: 11px !important;
    overflow: hidden !important;
}

.bm-image img {
    object-fit: contain !important;
}

/* CT controls */

.bm-ct-controls {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 12px 16px;
    margin-top: 10px;
}

.bm-ct-info {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 54px;
    color: #334155;
    font-size: 13px;
    font-weight: 650;
}

/* ------------------------------------------------------------
   Result cards
   ------------------------------------------------------------ */

.bm-result-card {
    background: white;
    border: 1px solid var(--bm-border);
    border-radius: 16px;
    padding: 18px;
}

.bm-result-title {
    color: var(--bm-navy);
    font-size: 15px;
    font-weight: 750;
    margin-bottom: 10px;
}

/* ------------------------------------------------------------
   Buttons
   ------------------------------------------------------------ */

button {
    border-radius: 10px !important;
}

button.primary {
    font-weight: 700 !important;
}

button.secondary {
    font-weight: 650 !important;
}

/* ------------------------------------------------------------
   Inputs
   ------------------------------------------------------------ */

input,
textarea,
select {
    border-radius: 10px !important;
}

/* ------------------------------------------------------------
   Divider
   ------------------------------------------------------------ */

.bm-divider {
    height: 1px;
    background: var(--bm-border);
    margin: 7px 0 18px;
}

/* ------------------------------------------------------------
   Footer
   ------------------------------------------------------------ */

.bm-footer {
    text-align: center;
    color: #94a3b8;
    font-size: 11px;
    line-height: 1.6;
    padding: 20px 10px 4px;
}

/* ------------------------------------------------------------
   Mobile
   ------------------------------------------------------------ */

@media (max-width: 850px) {

    .gradio-container {
        padding: 14px !important;
    }

    .bm-header {
        padding: 23px;
    }

    .bm-brand {
        font-size: 27px;
    }

    .bm-header-top {
        flex-direction: column;
    }
}

/* ============================================================
   PROFESSIONAL MULTIMODAL UPLOAD
   ============================================================ */

.bm-upload-header {
    display: flex;
    align-items: center;
    justify-content: space-between;

    padding: 4px 2px 18px;

    border-bottom: 1px solid #e5eaf2;
    margin-bottom: 18px;
}

.bm-upload-title {
    font-size: 21px;
    font-weight: 750;
    color: #0f172a;
    letter-spacing: -0.25px;
}

.bm-upload-subtitle {
    margin-top: 4px;
    font-size: 13px;
    color: #718096;
}

.bm-upload-status {
    padding: 7px 12px;

    border-radius: 999px;

    background: #eff6ff;
    border: 1px solid #dbeafe;

    color: #2563eb;

    font-size: 10px;
    font-weight: 750;
    letter-spacing: 0.7px;
}


/* ------------------------------------------------------------
   Upload cards
   ------------------------------------------------------------ */

.bm-upload-grid {
    gap: 16px !important;
    align-items: stretch !important;
}

.bm-upload-card,
.bm-upload-ct-card {
    background: #ffffff !important;

    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;

    padding: 18px !important;

    box-shadow:
        0 2px 8px rgba(15, 23, 42, 0.035) !important;

    transition:
        border-color 0.2s ease,
        box-shadow 0.2s ease;
}

.bm-upload-card:hover,
.bm-upload-ct-card:hover {
    border-color: #bfdbfe !important;

    box-shadow:
        0 5px 18px rgba(37, 99, 235, 0.07) !important;
}

.bm-upload-ct-card {
    margin-top: 16px !important;
}


/* ------------------------------------------------------------
   Card header
   ------------------------------------------------------------ */

.bm-upload-card-head {
    display: flex;
    align-items: center;
    gap: 11px;

    margin-bottom: 14px;
}

.bm-step {
    display: flex;

    align-items: center;
    justify-content: center;

    width: 30px;
    height: 30px;

    flex-shrink: 0;

    border-radius: 9px;

    background: #eff6ff;
    border: 1px solid #dbeafe;

    color: #2563eb;

    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.4px;
}

.bm-upload-card-title {
    font-size: 14px;
    font-weight: 750;
    color: #172033;
}

.bm-upload-card-subtitle {
    margin-top: 2px;

    font-size: 11px;
    color: #94a3b8;
}


/* ------------------------------------------------------------
   File drop area
   ------------------------------------------------------------ */

.bm-file-drop {
    height: 120px !important;
    min-height: 120px !important;

    border: 1.5px dashed #cbd5e1 !important;
    border-radius: 12px !important;

    background:
        linear-gradient(
            180deg,
            #fbfdff 0%,
            #f8fafc 100%
        ) !important;

    transition:
        border-color 0.2s ease,
        background 0.2s ease;
}

.bm-file-drop:hover {
    border-color: #60a5fa !important;

    background: #f8fbff !important;
}


/* Hide unnecessary default label area */

.bm-file-drop label {
    display: none !important;
}


/* Upload icon / text */

.bm-file-drop .upload-container {
    min-height: 120px !important;
    height: 120px !important;

    border: none !important;

    padding: 16px !important;
}

.bm-file-drop svg {
    width: 24px !important;
    height: 24px !important;
}

.bm-file-drop p {
    margin: 5px 0 !important;

    font-size: 12px !important;
}


/* ------------------------------------------------------------
   File information
   ------------------------------------------------------------ */

.bm-file-hint {
    margin-top: 9px;

    font-size: 10px;

    color: #94a3b8;

    letter-spacing: 0.15px;
}


/* ------------------------------------------------------------
   CT row
   ------------------------------------------------------------ */

.bm-ct-upload-row {
    align-items: stretch !important;
    gap: 14px !important;
}

.bm-ct-upload-row .gr-number {
    height: 100% !important;
}

.bm-ct-upload-row input {
    height: 42px !important;
}


/* ------------------------------------------------------------
   Analyze uploaded button
   ------------------------------------------------------------ */

.bm-upload-analyze {
    margin-top: 18px !important;

    min-height: 48px !important;

    border-radius: 11px !important;

    font-size: 14px !important;
    font-weight: 750 !important;

    box-shadow:
        0 4px 12px rgba(37, 99, 235, 0.16) !important;
}


/* ------------------------------------------------------------
   Responsive
   ------------------------------------------------------------ */

@media (max-width: 850px) {

    .bm-upload-grid {
        flex-direction: column !important;
    }

    .bm-upload-status {
        display: none;
    }

    .bm-ct-upload-row {
        flex-direction: column !important;
    }
}
"""


with gr.Blocks(
    title=TITLE,
    theme=gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
    ),
    css=CUSTOM_CSS,
) as demo:

    # ========================================================
    # HEADER
    # ========================================================

    gr.HTML(
        """
        <div class="bm-header">
            <div class="bm-header-top">
                <div>
                    <div class="bm-brand">🦴 BoneMD-Net</div>

                    <div class="bm-tagline">
                        Multimodal bone disorder analysis using
                        lumbar X-ray and CT imaging.
                    </div>
                </div>

                <div class="bm-status-pill">
                    <span class="bm-status-dot"></span>
                    RESEARCH PROTOTYPE
                </div>
            </div>
        </div>
        """
    )

    # ========================================================
    # MAIN WORKSPACE
    # ========================================================

    with gr.Row(
        elem_classes=["bm-workspace"],
    ):

        # ----------------------------------------------------
        # LEFT — ANALYSIS INPUT
        # ----------------------------------------------------

        with gr.Column(
            scale=5,
            elem_classes=["bm-card"],
        ):

            gr.HTML(
                """
                <div class="bm-card-title">
                    Analysis Input
                </div>

                <div class="bm-card-subtitle">
                    Select a LUMOS patient or provide your own
                    multimodal DICOM data.
                </div>
                """
            )

            # Patient mode

            gr.HTML(
                """
                <div class="bm-mode">
                    <div class="bm-mode-title">
                        Patient-based analysis
                    </div>

                    <div class="bm-mode-text">
                        Analyze a patient from the cached LUMOS test set.
                    </div>
                </div>
                """
            )

            patient_id = gr.Number(
                label="LUMOS Patient ID",
                value=4,
                precision=0,
                minimum=1,
                maximum=803,
            )

            analyze_button = gr.Button(
                "Analyze Patient",
                variant="primary",
                size="lg",
            )

            gr.HTML('<div class="bm-divider"></div>')

            # ============================================================
            # UPLOAD MULTIMODAL DATA
            # ============================================================

            gr.HTML(
                """
                <div class="bm-upload-header">

                    <div>
                        <div class="bm-upload-title">
                            Upload Multimodal Data
                        </div>

                        <div class="bm-upload-subtitle">
                            Provide the three imaging inputs required by the multimodal model.
                        </div>
                    </div>

                    <div class="bm-upload-status">
                        3 INPUTS REQUIRED
                    </div>

                </div>
                """
            )

            with gr.Row(
                elem_classes=["bm-upload-grid"],
            ):

                # --------------------------------------------------------
                # AP X-RAY
                # --------------------------------------------------------

                with gr.Column(
                    elem_classes=["bm-upload-card"],
                ):

                    gr.HTML(
                        """
                        <div class="bm-upload-card-head">
                            <span class="bm-step">01</span>

                            <div>
                                <div class="bm-upload-card-title">
                                    AP X-ray
                                </div>

                                <div class="bm-upload-card-subtitle">
                                    Anteroposterior lumbar view
                                </div>
                            </div>
                        </div>
                        """
                    )

                    uploaded_ap = gr.File(
                        label="",
                        file_types=[".dcm"],
                        type="filepath",
                        height=120,
                        show_label=False,
                        elem_classes=["bm-file-drop"],
                    )

                    gr.HTML(
                        """
                        <div class="bm-file-hint">
                            DICOM · lumbar AP image
                        </div>
                        """
                    )


                # --------------------------------------------------------
                # LATERAL X-RAY
                # --------------------------------------------------------

                with gr.Column(
                    elem_classes=["bm-upload-card"],
                ):

                    gr.HTML(
                        """
                        <div class="bm-upload-card-head">
                            <span class="bm-step">02</span>

                            <div>
                                <div class="bm-upload-card-title">
                                    Lateral X-ray
                                </div>

                                <div class="bm-upload-card-subtitle">
                                    Lateral lumbar view
                                </div>
                            </div>
                        </div>
                        """
                    )

                    uploaded_lateral = gr.File(
                        label="",
                        file_types=[".dcm"],
                        type="filepath",
                        height=120,
                        show_label=False,
                        elem_classes=["bm-file-drop"],
                    )

                    gr.HTML(
                        """
                        <div class="bm-file-hint">
                            DICOM · lateral image
                        </div>
                        """
                    )


            # ------------------------------------------------------------
            # CT ARCHIVE
            # ------------------------------------------------------------

            with gr.Column(
                elem_classes=["bm-upload-ct-card"],
            ):

                gr.HTML(
                    """
                    <div class="bm-upload-card-head">

                        <span class="bm-step">03</span>

                        <div>
                            <div class="bm-upload-card-title">
                                CT Volume
                            </div>

                            <div class="bm-upload-card-subtitle">
                                LUMOS CT DICOM archive
                            </div>
                        </div>

                    </div>
                    """
                )

                with gr.Row(
                    elem_classes=["bm-ct-upload-row"],
                ):

                    with gr.Column(
                        scale=4,
                    ):

                        uploaded_ct = gr.File(
                            label="",
                            file_types=[".zip"],
                            type="filepath",
                            height=120,
                            show_label=False,
                            elem_classes=["bm-file-drop"],
                        )

                    with gr.Column(
                        scale=1,
                        min_width=150,
                    ):

                        uploaded_ct_patient_id = gr.Number(
                            label="CT Patient ID",
                            value=4,
                            precision=0,
                            minimum=1,
                            maximum=803,
                        )

                gr.HTML(
                    """
                    <div class="bm-file-hint">
                        ZIP archive · select the LUMOS patient contained in the archive
                    </div>
                    """
                )


            # ------------------------------------------------------------
            # UPLOAD ANALYSIS BUTTON
            # ------------------------------------------------------------

            with gr.Row():

                upload_analyze_button = gr.Button(
                    "Analyze Uploaded Images",
                    variant="primary",
                    size="lg",
                    elem_classes=["bm-upload-analyze"],
                )

            status = gr.Markdown(
                "Ready for analysis.",
            )

        # ----------------------------------------------------
        # RIGHT — MODEL RESULT
        # ----------------------------------------------------

        with gr.Column(
            scale=5,
            elem_classes=["bm-card", "bm-result-panel"],
        ):

            gr.HTML(
                """
                <div class="bm-card-title">
                    Model Prediction
                </div>

                <div class="bm-card-subtitle">
                    BoneMD-Net final teacher inference result.
                </div>
                """
            )

            prediction = gr.Textbox(
                label="Predicted diagnosis",
                value="Waiting for analysis...",
                interactive=False,
                elem_classes=["bm-prediction-box"],
            )

            gr.Markdown(
                "#### Confidence"
            )

            probabilities = gr.HTML(
                value=format_probability_cards({}),
                show_label=False,
                elem_classes=["bm-result-html"],
            )

            gr.Markdown(
                "#### Feature representation"
            )

            fusion = gr.HTML(
                value=format_feature_cards({}),
                show_label=False,
                elem_classes=["bm-result-html"],
            )

    # ========================================================
    # IMAGE VIEWER
    # ========================================================

    gr.HTML(
        """
        <div class="bm-evidence-header">
            <div>
                <div class="bm-evidence-title">
                    Image Evidence
                </div>

                <div class="bm-evidence-subtitle">
                    Multimodal inputs used by BoneMD-Net
                </div>
            </div>

            <div class="bm-evidence-badge">
                AP · LATERAL · CT
            </div>
        </div>
        """
    )

    with gr.Row():

        with gr.Column(
            scale=1,
            elem_classes=["bm-image-card"],
        ):

            ap_preview = gr.Image(
                label="AP X-ray",
                type="pil",
                interactive=False,
                elem_classes=["bm-image"],
                height=330,
            )

        with gr.Column(
            scale=1,
            elem_classes=["bm-image-card"],
        ):

            lateral_preview = gr.Image(
                label="Lateral X-ray",
                type="pil",
                interactive=False,
                elem_classes=["bm-image"],
                height=330,
            )

        with gr.Column(
            scale=1,
            elem_classes=["bm-image-card"],
        ):

            ct_preview = gr.Image(
                label="CT axial slice",
                type="pil",
                interactive=False,
                elem_classes=["bm-image"],
                height=330,
            )

    # CT controls BELOW the images

    with gr.Row(
        elem_classes=["bm-ct-controls"],
    ):

        with gr.Column(scale=5):

            ct_slice_slider = gr.Slider(
                minimum=0,
                maximum=191,
                value=96,
                step=1,
                label="CT slice",
                interactive=True,
            )

        with gr.Column(
            scale=1,
            elem_classes=["bm-ct-info"],
        ):

            ct_slice_info = gr.Markdown(
                "Slice **96 / 191**"
            )

    ct_patient_state = gr.State(
        value=None
    )

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    with gr.Column(
        elem_classes=["bm-card"],
    ):

        gr.HTML(
            """
            <div class="bm-card-title">
                Model Information
            </div>

            <div class="bm-card-subtitle">
                Current deployed research configuration.
            </div>
            """
        )

        gr.Markdown(
            """
| Component | Configuration |
|---|---|
| Architecture | BoneMDTeacherNorm |
| Learning strategy | Multimodal supervised learning |
| Modalities | AP X-ray + Lateral X-ray + CT |
| Output | 3-class bone disorder classification |
| Parameters | 6.11M |
| Checkpoint | `FINAL_BEST_TEACHER_75_61.pth` |

**Classes:** Normal · Osteopenia · Osteoporosis
"""
        )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    gr.HTML(
        """
        <div class="bm-footer">

            <strong>BoneMD-Net</strong> · Multimodal medical
            image analysis research prototype.

            <br>

            This system is intended for research and demonstration
            purposes only and is <strong>not intended for clinical
            diagnosis or medical decision-making.</strong>

        </div>
        """
    )

    # ========================================================
    # EVENTS
    # ========================================================

    # Patient analysis

    analyze_button.click(
        fn=analyze_patient,
        inputs=patient_id,
        outputs=[
            ap_preview,
            lateral_preview,
            ct_preview,
            ct_patient_state,
            ct_slice_info,
            prediction,
            probabilities,
            fusion,
            status,
        ],
    )

    # Uploaded AP preview

    uploaded_ap.change(
        fn=load_uploaded_xray_preview,
        inputs=uploaded_ap,
        outputs=ap_preview,
    )

    # Uploaded lateral preview

    uploaded_lateral.change(
        fn=load_uploaded_xray_preview,
        inputs=uploaded_lateral,
        outputs=lateral_preview,
    )

    # Uploaded CT preview

    uploaded_ct.change(
        fn=load_uploaded_ct_preview,
        inputs=[
            uploaded_ct,
            uploaded_ct_patient_id,
        ],
        outputs=[
            ct_preview,
            ct_patient_state,
        ],
    )

    uploaded_ct_patient_id.change(
        fn=load_uploaded_ct_preview,
        inputs=[
            uploaded_ct,
            uploaded_ct_patient_id,
        ],
        outputs=[
            ct_preview,
            ct_patient_state,
        ],
    )

    # Uploaded multimodal analysis

    upload_analyze_button.click(
        fn=analyze_uploaded_images,
        inputs=[
            uploaded_ap,
            uploaded_lateral,
            uploaded_ct,
            uploaded_ct_patient_id,
        ],
        outputs=[
            prediction,
            probabilities,
            fusion,
        ],
    )

    # --------------------------------------------------------
    # CT slice navigation
    # --------------------------------------------------------

    def update_selected_ct_slice(
        slice_index,
        ct_volume,
    ):

        if ct_volume is None:
            return (
                None,
                "No CT volume loaded.",
            )

        image = update_ct_slice(
            ct_volume,
            slice_index,
        )

        return (
            image,
            f"Slice **{int(slice_index)} / 191**",
        )

    ct_slice_slider.change(
        fn=update_selected_ct_slice,
        inputs=[
            ct_slice_slider,
            ct_patient_state,
        ],
        outputs=[
            ct_preview,
            ct_slice_info,
        ],
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    demo.launch(
        css=CUSTOM_CSS,
    )
