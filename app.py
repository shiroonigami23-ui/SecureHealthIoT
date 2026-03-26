from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from disease_ml.inference import DiseasePredictor


BUNDLE_PATH = "artifacts/latest_model.joblib"
REPORT_DIR = Path("artifacts/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

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


def _suggest_supportive_care(symptoms: list[str], abstained: bool) -> list[str]:
    sym_set = set(symptoms)
    recs = [
        "Hydration, rest, and regular monitoring of symptom progression.",
        "Avoid self-medication with antibiotics/steroids without physician advice.",
    ]
    if "fever" in sym_set:
        recs.append("For fever discomfort, discuss acetaminophen/paracetamol suitability with your doctor.")
    if "cough" in sym_set:
        recs.append("For cough, use warm fluids and seek doctor advice before any suppressant/antibiotic.")
    if "vomiting" in sym_set or "nausea" in sym_set:
        recs.append("Use oral rehydration and seek urgent care if dehydration signs appear.")
    if abstained:
        recs.insert(0, "Model abstained due to low confidence: prioritize in-person clinical evaluation.")
    recs.append("Prescription medications must be decided only by a licensed clinician after examination.")
    return recs


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


def _build_report_pdf(detail: dict, precautions_md: str, analysis_md: str, care_plan: list[str], prefix: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = REPORT_DIR / f"{prefix}_{ts}.pdf"
    c = canvas.Canvas(str(out), pagesize=A4)
    width, height = A4
    y = height - 40

    def write(line: str, step: int = 15):
        nonlocal y
        if y < 60:
            c.showPage()
            y = height - 40
        c.drawString(40, y, line[:130])
        y -= step

    write("SecureHealthIoT Clinical Decision Support Report", 18)
    write(f"Generated UTC: {detail.get('timestamp_utc', '')}")
    write(f"Source: {detail.get('source', '')}")
    write(f"Selected symptoms: {', '.join(detail.get('selected_symptoms', []))}", 18)
    write(f"Predicted disease: {detail.get('predicted_disease', 'ABSTAINED')}")
    write(f"Probability: {detail.get('probability', 0.0):.3f}")
    write(f"Confidence band: {detail.get('confidence_band', 'N/A')}", 18)
    write("Description:")
    for ln in (detail.get("description", "") or "N/A").split("\n"):
        write(f"  {ln}")
    write("Precautions:")
    for ln in precautions_md.replace("-", "").split("\n"):
        if ln.strip():
            write(f"  - {ln.strip()}")
    write("Supportive care / prescription guidance:")
    for r in care_plan:
        write(f"  - {r}")
    write("Safety disclaimer:")
    write("  This report is NOT a medical diagnosis or legal prescription.")
    write("  Consult a licensed doctor before any medication/treatment changes.", 18)
    write("Summary excerpt:")
    for ln in re.sub(r"[#*`]", "", analysis_md).split("\n"):
        if ln.strip():
            write(f"  {ln.strip()}")

    c.save()
    return str(out)


def infer(symptoms, abstain_threshold: float, source_note: str = "symptom_form"):
    if not symptoms:
        return "Please select at least one symptom.", "", "", "", None

    result = predictor.predict_with_abstain(symptoms, k=3, abstain_threshold=abstain_threshold)
    preds = result["predictions"]
    probs = {x["disease"]: round(x["probability"], 4) for x in preds} if preds else {}
    top = preds[0] if preds else {"disease": "ABSTAIN", "probability": 0.0, "description": "", "precautions": []}

    band = _confidence_band(top["probability"])
    care_plan = _suggest_supportive_care(symptoms, result["abstained"])
    detail = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "source": source_note,
        "selected_symptoms": sorted(set(symptoms)),
        "predicted_disease": top["disease"] if not result["abstained"] else "ABSTAINED",
        "probability": top["probability"],
        "confidence_band": band,
        "abstained": result["abstained"],
        "abstain_reason": result["reason"],
        "description": top["description"],
        "precautions": top["precautions"],
        "supportive_care_plan": care_plan,
    }
    lead_line = (
        f"### Prediction abstained\n- Reason: {result['reason']}\n"
        if result["abstained"]
        else f"### Predicted condition: **{top['disease']}**\n"
    )
    analysis = (
        f"{lead_line}"
        f"- Confidence: **{top['probability']:.3f} ({band})**\n"
        f"- Description: {top['description'] or 'Not available'}\n\n"
        f"### Suggested precautions\n{_format_precautions(top['precautions'])}\n\n"
        "### Preliminary supportive care guidance (not a prescription)\n"
        + "\n".join([f"- {x}" for x in care_plan])
        + "\n\n### Safety note\n"
        "This tool is a decision-support aid, not a medical diagnosis. "
        "Consult a licensed clinician for prescriptions and final treatment."
    )
    pdf_path = _build_report_pdf(detail, _format_precautions(top["precautions"]), analysis, care_plan, "quick")

    return (
        probs if probs else "ABSTAINED - Low confidence",
        json.dumps(detail, indent=2),
        _format_precautions(top["precautions"]),
        analysis,
        pdf_path,
    )


def analyze_report(report_file, image_file, manual_symptoms, abstain_threshold: float):
    report_text = _extract_text_from_report(report_file) if report_file else ""
    from_report = _extract_symptoms_from_text(report_text)
    manual = [_normalize_symptom(s) for s in (manual_symptoms or [])]
    combined = sorted(set(from_report + manual))
    image_quality = _analyze_image_quality(image_file)

    if not combined:
        msg = (
            "Could not detect known symptoms from uploaded report/image. "
            "Please select symptoms manually for reliable prediction."
        )
        detail = {
            "timestamp_utc": datetime.utcnow().isoformat(),
            "source": "report_image_assisted",
            "selected_symptoms": [],
            "predicted_disease": "ABSTAINED",
            "abstained": True,
            "abstain_reason": msg,
            "supportive_care_plan": _suggest_supportive_care([], True),
        }
        analysis = f"{msg}\n\n{image_quality}"
        pdf_path = _build_report_pdf(detail, "No precautions available.", analysis, detail["supportive_care_plan"], "report")
        return msg, "{}", "No precautions available.", "", analysis, pdf_path

    probs, detail_json, precautions, analysis, pdf_path = infer(
        combined, abstain_threshold=abstain_threshold, source_note="report_image_assisted"
    )
    detected = ", ".join(combined)
    return probs, detail_json, precautions, detected, f"{analysis}\n\n{image_quality}", pdf_path


with gr.Blocks(title="SecureHealthIoT Disease Predictor") as demo:
    gr.Markdown("# SecureHealthIoT: Disease Prediction")
    gr.Markdown(
        "Symptom-based disease prediction assistant with report/image-supported triage and clinician PDF export."
    )
    gr.Markdown(
        "**Medical disclaimer:** This tool does not replace clinical diagnosis. "
        "Any prescription decision must be made by a licensed physician."
    )

    with gr.Tabs():
        with gr.Tab("Quick Symptom Predictor"):
            symptom_input = gr.Dropdown(
                choices=symptom_vocab,
                multiselect=True,
                label="Symptoms",
                info="Search and select one or more symptoms",
            )
            abstain_quick = gr.Slider(
                minimum=0.3, maximum=0.9, value=0.55, step=0.01, label="Abstain Threshold"
            )
            submit = gr.Button("Predict", variant="primary")
            clear = gr.Button("Clear")
            prob_out = gr.Label(label="Top Predictions")
            detail_out = gr.Code(label="Prediction Detail", language="json")
            precautions_out = gr.Markdown(label="Precautions")
            analysis_out = gr.Markdown(label="Clinical Decision Support Summary")
            pdf_out = gr.File(label="Clinician PDF Report")

            submit.click(
                infer,
                inputs=[symptom_input, abstain_quick],
                outputs=[prob_out, detail_out, precautions_out, analysis_out, pdf_out],
            )
            clear.click(
                lambda: (None, None, "", "", None),
                inputs=[],
                outputs=[symptom_input, detail_out, precautions_out, analysis_out, pdf_out],
            )

        with gr.Tab("Report/Image Assisted Analysis"):
            gr.Markdown(
                "Upload a report (`.txt`, `.csv`, `.pdf`) and optional image for quality analysis. "
                "Detected symptoms are combined with manual selections for prediction."
            )
            report_file = gr.File(label="Medical Report File", file_types=[".txt", ".csv", ".pdf"])
            image_file = gr.Image(type="filepath", label="Optional Clinical Image")
            manual_symptoms = gr.Dropdown(
                choices=symptom_vocab,
                multiselect=True,
                label="Manual Symptoms (optional)",
            )
            abstain_report = gr.Slider(
                minimum=0.3, maximum=0.9, value=0.55, step=0.01, label="Abstain Threshold"
            )
            run_report = gr.Button("Analyze Report + Predict", variant="primary")
            rep_prob = gr.Label(label="Top Predictions")
            rep_detail = gr.Code(label="Prediction Detail", language="json")
            rep_prec = gr.Markdown(label="Precautions")
            detected_box = gr.Textbox(label="Detected Symptoms Used", lines=2)
            rep_analysis = gr.Markdown(label="Report Analysis Summary")
            rep_pdf = gr.File(label="Clinician PDF Report")

            run_report.click(
                analyze_report,
                inputs=[report_file, image_file, manual_symptoms, abstain_report],
                outputs=[rep_prob, rep_detail, rep_prec, detected_box, rep_analysis, rep_pdf],
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
