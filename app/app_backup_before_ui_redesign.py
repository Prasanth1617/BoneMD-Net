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

XRay_ZIP = PROJECT_ROOT / "dataset" / "lumos_x_001_280_dcm.zip"

MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "final_multimodal_manifest.csv"
)

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


def predict_uploaded_images(ap_path, lateral_path, ct_zip_path):
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

    with zipfile.ZipFile(ct_zip_path, "r") as archive:
        folders = sorted({
            name.split("/")[0]
            for name in archive.namelist()
            if "/" in name and name.lower().endswith(".dcm")
        })

    if len(folders) != 1:
        raise ValueError(
            "CT ZIP must contain DICOM files for one patient."
        )

    ct_volume, ct_metadata = load_ct_volume(
        ct_zip_path,
        folders[0],
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

        fusion_weights = outputs["fusion_weights"][0]

    return {
        "predicted_class": predicted_class,
        "predicted_name": CLASS_NAMES[predicted_class],
        "probabilities": {
            CLASS_NAMES[i]: float(
                probabilities[i].item()
            )
            for i in range(3)
        },
        "fusion_weights": {
            "AP": float(fusion_weights[0].item()),
            "Lateral": float(fusion_weights[1].item()),
            "CT": float(fusion_weights[2].item()),
        },
    }


def analyze_uploaded_images(ap_path, lateral_path, ct_zip_path):
    result = predict_uploaded_images(
        ap_path,
        lateral_path,
        ct_zip_path,
    )

    prediction_text = (
        f"{result['predicted_name']} "
        f"(Class {result['predicted_class']})"
    )

    return (
        prediction_text,
        result["probabilities"],
        result["fusion_weights"],
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
        {},
        {},
        "No patient selected.",
    )

    if patient_id is None:
        return empty

    patient_id = int(patient_id)

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    matches = manifest[
        manifest["patient_id"] == patient_id
    ]

    if matches.empty:
        return (
            None,
            None,
            None,
            None,
            None,
            "Patient not available.",
            {},
            {},
            f"Patient {patient_id} is not present in the multimodal inference manifest.",
        )

    record = matches.iloc[0]

    ap_path = str(
        record["ap_dicom_file"]
    )

    lateral_path = str(
        record["lateral_dicom_file"]
    )

    ct_zip = (
        PROJECT_ROOT
        / "dataset"
        / str(record["ct_zip"])
    )

    ct_folder = str(
        record["ct_patient_folder"]
    )

    try:

        ap_image = load_preview(
            XRay_ZIP,
            ap_path,
        )

        lateral_image = load_preview(
            XRay_ZIP,
            lateral_path,
        )

        ct_volume = load_ct_volume_for_preview(
            ct_zip,
            ct_folder,
        )

        ct_image = ct_slice_to_image(
            ct_volume,
            96,
        )

        result = ENGINE.predict(
            XRay_ZIP,
            ap_path,
            XRay_ZIP,
            lateral_path,
            ct_zip,
            ct_folder,
        )

    except Exception as exc:

        return (
            None,
            None,
            None,
            None,
            None,
            "Inference error",
            {},
            {},
            str(exc),
        )

    prediction = (
        f"{result['predicted_name']} "
        f"(Class {result['predicted_class']})"
    )

    status = (
        f"Patient {patient_id} analyzed successfully. "
        f"Model checkpoint: epoch "
        f"{result['checkpoint_epoch']}."
    )

    return (
        ap_image,
        lateral_image,
        ct_image,
        ct_volume,
        "Slice **96 / 191**",
        prediction,
        result["probabilities"],
        result["fusion_weights"],
        status,
    )


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------

CUSTOM_CSS = """
/* ================================
   BoneMD-Net Research UI
   ================================ */

body {
    background: #f5f7fb !important;
}

.gradio-container {
    max-width: 1450px !important;
    margin: auto !important;
    padding: 24px 32px 40px !important;
}

/* Main title */

.bonemd-header {
    background: linear-gradient(135deg, #0f172a, #1e3a5f);
    color: white;
    border-radius: 18px;
    padding: 30px 34px;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
}

.bonemd-title {
    font-size: 34px;
    font-weight: 750;
    margin: 0;
    letter-spacing: -0.5px;
}

.bonemd-subtitle {
    font-size: 16px;
    opacity: 0.88;
    margin-top: 8px;
}

.bonemd-badge {
    display: inline-block;
    margin-top: 16px;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.22);
    font-size: 12px;
    font-weight: 600;
}

/* Section cards */

.bonemd-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
}

/* Section headings */

.bonemd-section-title {
    font-size: 20px;
    font-weight: 700;
    color: #172033;
    margin-bottom: 4px;
}

.bonemd-section-description {
    color: #64748b;
    font-size: 14px;
    margin-bottom: 16px;
}

/* Image panels */

.bonemd-image-panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 12px;
    min-height: 300px;
}

/* Prediction */

.bonemd-prediction {
    background: linear-gradient(135deg, #f8fafc, #eef4ff);
    border: 1px solid #dbe5f5;
    border-radius: 16px;
    padding: 24px;
}

/* Buttons */

button.primary {
    border-radius: 10px !important;
    font-weight: 650 !important;
}

button.secondary {
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* Inputs */

input,
textarea,
select {
    border-radius: 9px !important;
}

/* Footer */

.bonemd-footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding: 20px;
}

/* Mobile */

@media (max-width: 800px) {
    .gradio-container {
        padding: 16px !important;
    }

    .bonemd-title {
        font-size: 27px;
    }

    .bonemd-header {
        padding: 24px;
    }
}
"""


with gr.Blocks(
    title=TITLE
) as demo:

    # Header
    gr.Markdown(
        "# 🦴 BoneMD-Net",
        elem_id="title",
    )

    gr.Markdown(
        "**Multimodal Knowledge Distillation Network for "
        "Efficient Bone Disorder Diagnosis Using X-ray and CT Images**",
        elem_id="subtitle",
    )

    gr.Markdown(
        """
        **Research prototype — not for clinical use.**

        BoneMD-Net combines lumbar AP X-ray, lateral X-ray,
        and CT information using a lightweight multimodal
        knowledge-distilled model.
        """,
        elem_id="warning",
    )

    gr.Markdown("---")

    # Input modalities preview
    gr.Markdown("## Input Modalities")

    with gr.Row():

        ap_preview = gr.Image(
            label="AP X-ray",
            type="pil",
            interactive=False,
        )

        lateral_preview = gr.Image(
            label="Lateral X-ray",
            type="pil",
            interactive=False,
        )

        ct_preview = gr.Image(
            label="CT Axial Slice",
            type="pil",
            interactive=False,
        )

        ct_slice_slider = gr.Slider(
            minimum=0,
            maximum=191,
            value=96,
            step=1,
            label="CT Slice Index",
            interactive=True,
        )

        ct_slice_info = gr.Markdown(
            "Slice **96 / 191**"
        )

        ct_patient_state = gr.State(
            value=None
        )

    # Uploaded patient images
    gr.Markdown("## Upload Patient Images")

    with gr.Row():
        uploaded_ap = gr.File(
            label="AP X-ray (DICOM)",
            file_types=[".dcm"],
            type="filepath",
        )

        uploaded_lateral = gr.File(
            label="Lateral X-ray (DICOM)",
            file_types=[".dcm"],
            type="filepath",
        )

        uploaded_ct = gr.File(
            label="CT DICOM ZIP",
            file_types=[".zip"],
            type="filepath",
        )

        uploaded_ct_patient_id = gr.Number(
            label="CT Patient ID",
            value=4,
            precision=0,
            minimum=1,
            maximum=803,
        )

    # Patient input
    with gr.Row():

        with gr.Column(scale=1):

            gr.Markdown("## Patient Analysis")

            patient_id = gr.Number(
                label="LUMOS Patient ID",
                value=4,
                precision=0,
            )

            analyze_button = gr.Button(
                "🔍 Analyze Patient",
                variant="primary",
                size="lg",
            )

            upload_analyze_button = gr.Button(
                "Analyze Uploaded Images",
                variant="secondary",
                size="lg",
            )

            status = gr.Markdown(
                "Ready for analysis."
            )

        with gr.Column(scale=1):

            gr.Markdown("## Model Prediction")

            prediction = gr.Textbox(
                label="Predicted Diagnosis",
                value="Waiting for analysis...",
                interactive=False,
            )

    gr.Markdown("---")

    # Results
    with gr.Row():

        with gr.Column():

            gr.Markdown(
                "## Class Probabilities"
            )

            probabilities = gr.Label(
                label="Model confidence distribution",
                value={},
                num_top_classes=3,
            )

        with gr.Column():

            gr.Markdown(
                "## Multimodal Fusion"
            )

            fusion = gr.Label(
                label="Relative modality weights",
                value={},
                num_top_classes=3,
            )

    gr.Markdown("---")

    # Architecture information
    gr.Markdown(
        """
        ## Model Information

        | Component | Configuration |
        |---|---|
        | Architecture | BoneMD-Net Student |
        | Learning strategy | Knowledge Distillation |
        | Modalities | AP X-ray + Lateral X-ray + CT |
        | Output | 3-class bone disorder classification |
        | Student parameters | 1.11M |
        | Checkpoint | `student_kd_v2_best.pth` |

        **Classes:** Normal · Osteopenia · Osteoporosis
        """
    )

    gr.Markdown(
        """
        ---
        
        **BoneMD-Net research prototype**  
        Multimodal medical image analysis for research and demonstration.
        
        *This system is not intended for clinical diagnosis or medical decision-making.*
        """
    )

    # Button connection
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

    uploaded_ap.change(
        fn=load_uploaded_xray_preview,
        inputs=uploaded_ap,
        outputs=ap_preview,
    )

    uploaded_lateral.change(
        fn=load_uploaded_xray_preview,
        inputs=uploaded_lateral,
        outputs=lateral_preview,
    )

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

    upload_analyze_button.click(
        fn=analyze_uploaded_images,
        inputs=[
            uploaded_ap,
            uploaded_lateral,
            uploaded_ct,
        ],
        outputs=[
            prediction,
            probabilities,
            fusion,
        ],
    )

    def update_selected_ct_slice(slice_index, ct_volume):
        if ct_volume is None:
            return None, "No CT volume loaded."

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


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )