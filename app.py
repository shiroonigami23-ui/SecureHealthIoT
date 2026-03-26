from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image
from pypdf import PdfReader

from disease_ml.inference import DiseasePredictor


BUNDLE_PATH = "artifacts/latest_model.joblib"

if not Path(BUNDLE_PATH).exists():
    raise RuntimeError(
        "Model bundle not found. Run: python -m disease_ml.train\n"
        "This will create artifacts/latest_model.joblib."
    )

predictor = DiseasePredictor(BUNDLE_PATH)
symptom_vocab = predictor.symptom_vocab


def _normalize_symptom(x: str) -> str:
    return x.strip().lower().replace(" ", "_")


def _confidence_band(prob: float) -> str:
    if prob >= 0.8:
        return "High"
    if prob >= 0.55:
        return "Moderate"
    return "Low"


def _format_precautions(items):
    if not items:
        return "No precaution data available."
    return "\n".join([f"- {p}" for p in items])


def _extract_text_from_report(file_path: str) -> str:
    if not file_path:
        return ""
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".csv"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        text_chunks = []
        reader = PdfReader(str(path))
        for page in reader.pages:
            text_chunks.append(page.extract_text() or "")
        return "\n".join(text_chunks)
    return ""


def _extract_symptoms_from_text(text: str):
    if not text.strip():
        return []
    normalized_text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()
    found = []
    for sym in symptom_vocab:
        phrase = sym.replace("_", " ")
        if phrase in normalized_text:
            found.append(sym)
    return sorted(set(found))


def _analyze_image_quality(image_path: str) -> str:
    if not image_path:
        return "No image uploaded."
    try:
        img = Image.open(image_path).convert("L")
        arr = np.asarray(img, dtype=np.float32)
        brightness = float(arr.mean())
        contrast = float(arr.std())
        quality = "Good"
        if brightness < 35 or brightness > 220 or contrast < 18:
            quality = "Needs Better Scan"
        return (
            f"Image analysis: {img.width}x{img.height}px, "
            f"brightness={brightness:.1f}, contrast={contrast:.1f}, quality={quality}."
        )
    except Exception as exc:
        return f"Image analysis unavailable: {exc}"


def infer(symptoms, source_note: str = "symptom_form"):
    if not symptoms:
        return "Please select at least one symptom.", "", "", ""

    preds = predictor.predict_top_k(symptoms, k=3)
    probs = {x["disease"]: round(x["probability"], 4) for x in preds}
    top = preds[0]
    band = _confidence_band(top["probability"])
    detail = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "source": source_note,
        "selected_symptoms": symptoms,
        "predicted_disease": top["disease"],
        "probability": top["probability"],
        "confidence_band": band,
        "description": top["description"],
        "precautions": top["precautions"],
    }
    analysis = (
        f"### Predicted condition: **{top['disease']}**\n"
        f"- Confidence: **{top['probability']:.3f} ({band})**\n"
        f"- Description: {top['description'] or 'Not available'}\n\n"
        f"### Suggested precautions\n{_format_precautions(top['precautions'])}\n\n"
        "### Safety note\n"
        "This tool is a decision-support aid, not a medical diagnosis. "
        "If symptoms are severe, worsening, or persistent, consult a licensed clinician."
    )
    return (
        probs,
        json.dumps(detail, indent=2),
        _format_precautions(top["precautions"]),
        analysis,
    )


def analyze_report(report_file, image_file, manual_symptoms):
    report_text = _extract_text_from_report(report_file) if report_file else ""
    from_report = _extract_symptoms_from_text(report_text)
    manual = [_normalize_symptom(s) for s in (manual_symptoms or [])]
    combined = sorted(set(from_report + manual))

    if not combined:
        msg = (
            "Could not detect known symptoms from uploaded report/image. "
            "Please select symptoms manually for reliable prediction."
        )
        return msg, "{}", "No precautions available.", "", _analyze_image_quality(image_file)

    probs, detail_json, precautions, analysis = infer(combined, source_note="report_image_assisted")
    detected = ", ".join(combined)
    image_quality = _analyze_image_quality(image_file)
    return probs, detail_json, precautions, detected, f"{analysis}\n\n{image_quality}"


with gr.Blocks(title="SecureHealthIoT Disease Predictor") as demo:
    gr.Markdown("# SecureHealthIoT: Disease Prediction")
    gr.Markdown(
        "Symptom-based disease prediction assistant with report/image-supported triage. "
        "Trained on Kaggle dataset `itachi9604/disease-symptom-description-dataset`."
    )
    gr.Markdown(
        "**Medical disclaimer:** This tool does not replace clinical diagnosis. "
        "Always consult a qualified healthcare professional for treatment decisions."
    )

    with gr.Tabs():
        with gr.Tab("Quick Symptom Predictor"):
            with gr.Row():
                symptom_input = gr.Dropdown(
                    choices=symptom_vocab,
                    multiselect=True,
                    label="Symptoms",
                    info="Search and select one or more symptoms",
                )
            with gr.Row():
                submit = gr.Button("Predict", variant="primary")
                clear = gr.Button("Clear")
            with gr.Row():
                prob_out = gr.Label(label="Top Predictions")
            with gr.Row():
                detail_out = gr.Code(label="Prediction Detail", language="json")
            with gr.Row():
                precautions_out = gr.Markdown(label="Precautions")
            with gr.Row():
                analysis_out = gr.Markdown(label="Clinical Decision Support Summary")

            submit.click(
                infer,
                inputs=[symptom_input],
                outputs=[prob_out, detail_out, precautions_out, analysis_out],
            )
            clear.click(
                lambda: (None, None, "", ""),
                inputs=[],
                outputs=[symptom_input, detail_out, precautions_out, analysis_out],
            )

        with gr.Tab("Report/Image Assisted Analysis"):
            gr.Markdown(
                "Upload a report (`.txt`, `.csv`, `.pdf`) and optional image for quality analysis. "
                "Detected symptoms are combined with manual selections for prediction."
            )
            with gr.Row():
                report_file = gr.File(label="Medical Report File", file_types=[".txt", ".csv", ".pdf"])
                image_file = gr.Image(type="filepath", label="Optional Clinical Image")
            manual_symptoms = gr.Dropdown(
                choices=symptom_vocab,
                multiselect=True,
                label="Manual Symptoms (optional)",
            )
            run_report = gr.Button("Analyze Report + Predict", variant="primary")
            with gr.Row():
                rep_prob = gr.Label(label="Top Predictions")
            with gr.Row():
                rep_detail = gr.Code(label="Prediction Detail", language="json")
            with gr.Row():
                rep_prec = gr.Markdown(label="Precautions")
            detected_box = gr.Textbox(label="Detected Symptoms Used", lines=2)
            rep_analysis = gr.Markdown(label="Report Analysis Summary")

            run_report.click(
                analyze_report,
                inputs=[report_file, image_file, manual_symptoms],
                outputs=[rep_prob, rep_detail, rep_prec, detected_box, rep_analysis],
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
