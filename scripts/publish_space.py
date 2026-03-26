from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


SPACE_APP = r'''
import json
import re
from datetime import datetime
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
from PIL import Image
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from huggingface_hub import hf_hub_download

MODEL_REPO_ID = "REPLACE_MODEL_REPO"
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

bundle_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename="latest_model.joblib")
bundle = joblib.load(bundle_path)
model = bundle["model"]
label_encoder = bundle["label_encoder"]
vectorizer = bundle.get("vectorizer")
symptom_binarizer = bundle.get("symptom_binarizer")
aux = bundle.get("aux", {"descriptions": {}, "precautions": {}})

def symptom_vocab():
    if vectorizer is not None and hasattr(vectorizer, "symptom_vocab"):
        return list(vectorizer.symptom_vocab)
    if symptom_binarizer is not None and hasattr(symptom_binarizer, "classes_"):
        return list(symptom_binarizer.classes_)
    return []

SYMPTOMS = symptom_vocab()

def normalize(symptoms):
    return sorted(set([s.strip().lower().replace(" ", "_") for s in symptoms if s and s.strip()]))

def transform(symptoms):
    if vectorizer is not None:
        return vectorizer.transform([symptoms])
    if symptom_binarizer is not None:
        return symptom_binarizer.transform([symptoms]).astype(np.float32)
    raise RuntimeError("No vectorizer/symptom_binarizer found in model bundle.")

def confidence_band(prob):
    if prob >= 0.8:
        return "High"
    if prob >= 0.55:
        return "Moderate"
    return "Low"

def format_precautions(items):
    if not items:
        return "No precaution data available."
    return "\n".join([f"- {x}" for x in items])

def supportive_care(symptoms, abstained):
    s = set(symptoms)
    recs = [
        "Hydration, rest, and symptom monitoring.",
        "Avoid unsupervised antibiotics/steroids.",
    ]
    if "fever" in s:
        recs.append("Discuss acetaminophen/paracetamol suitability with your doctor.")
    if "cough" in s:
        recs.append("Use warm fluids and seek medical advice before cough medicines.")
    if abstained:
        recs.insert(0, "Model abstained due to low confidence; prioritize doctor evaluation.")
    recs.append("Any prescription medication must be finalized by a licensed clinician.")
    return recs

def extract_text(file_path):
    if not file_path:
        return ""
    if file_path.lower().endswith((".txt", ".csv")):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        return "\n".join([(page.extract_text() or "") for page in reader.pages])
    return ""

def extract_symptoms_from_text(text):
    if not text.strip():
        return []
    t = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    t = re.sub(r"\s+", " ", t).strip()
    out = []
    for sym in SYMPTOMS:
        if sym.replace("_", " ") in t:
            out.append(sym)
    return sorted(set(out))

def image_quality(image_path):
    if not image_path:
        return "No image uploaded."
    try:
        img = Image.open(image_path).convert("L")
        arr = np.asarray(img, dtype=np.float32)
        b = float(arr.mean())
        c = float(arr.std())
        q = "Good"
        if b < 35 or b > 220 or c < 18:
            q = "Needs Better Scan"
        return f"Image analysis: {img.width}x{img.height}px, brightness={b:.1f}, contrast={c:.1f}, quality={q}."
    except Exception as exc:
        return f"Image analysis unavailable: {exc}"

def write_pdf(detail, precautions_md, analysis_md, care_plan, prefix):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = REPORT_DIR / f"{prefix}_{ts}.pdf"
    c = canvas.Canvas(str(out), pagesize=A4)
    width, height = A4
    y = height - 40

    def w(line, step=15):
        nonlocal y
        if y < 60:
            c.showPage()
            y = height - 40
        c.drawString(40, y, str(line)[:130])
        y -= step

    w("SecureHealthIoT Clinical Decision Support Report", 18)
    w(f"Generated UTC: {detail.get('timestamp_utc','')}")
    w(f"Source: {detail.get('source','')}")
    w(f"Symptoms: {', '.join(detail.get('selected_symptoms', []))}", 18)
    w(f"Predicted disease: {detail.get('predicted_disease','ABSTAINED')}")
    w(f"Probability: {detail.get('probability', 0.0):.3f}")
    w(f"Confidence: {detail.get('confidence_band','N/A')}", 18)
    w("Precautions:")
    for ln in precautions_md.replace("-", "").split("\n"):
        if ln.strip():
            w(f"  - {ln.strip()}")
    w("Supportive care / prescription guidance:")
    for r in care_plan:
        w(f"  - {r}")
    w("Safety disclaimer:")
    w("  Not a diagnosis or legal prescription.")
    w("  Consult a licensed physician for all medication decisions.")
    c.save()
    return str(out)

def infer(symptoms, abstain_threshold, source_note="symptom_form"):
    symptoms = normalize(symptoms or [])
    if not symptoms:
        return "Please select at least one symptom.", "{}", "", "", None
    X = transform(symptoms)
    probs = model.predict_proba(X)[0]
    top_idx = probs.argsort()[::-1][:3]
    pred = []
    for idx in top_idx:
        disease = label_encoder.inverse_transform([idx])[0]
        pred.append((disease, float(probs[idx])))
    top_disease, top_prob = pred[0]
    abstained = top_prob < abstain_threshold
    reason = ""
    if abstained:
        reason = f"Top confidence {top_prob:.3f} is below threshold {abstain_threshold:.3f}."
    care_plan = supportive_care(symptoms, abstained)
    details = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "source": source_note,
        "selected_symptoms": symptoms,
        "predicted_disease": "ABSTAINED" if abstained else top_disease,
        "probability": top_prob,
        "confidence_band": confidence_band(top_prob),
        "abstained": abstained,
        "abstain_reason": reason,
        "description": aux["descriptions"].get(top_disease, ""),
        "precautions": aux["precautions"].get(top_disease, []),
        "supportive_care_plan": care_plan,
    }
    pretty = {d: round(p, 4) for d, p in pred}
    precautions = format_precautions(details["precautions"])
    summary = (
        ("### Prediction abstained\n- " + reason + "\n" if abstained else f"### Predicted condition: **{top_disease}**\n")
        + f"- Confidence: **{top_prob:.3f} ({confidence_band(top_prob)})**\n"
        + f"- Description: {details['description'] or 'Not available'}\n\n"
        + f"### Suggested precautions\n{precautions}\n\n"
        + "### Preliminary supportive care guidance (not a prescription)\n"
        + "\n".join([f"- {x}" for x in care_plan])
        + "\n\n### Safety note\nThis tool is decision support only."
    )
    pdf_path = write_pdf(details, precautions, summary, care_plan, "quick")
    return ("ABSTAINED - Low confidence" if abstained else pretty, json.dumps(details, indent=2), precautions, summary, pdf_path)

def analyze_report(report_file, image_file, manual_symptoms, abstain_threshold):
    text = extract_text(report_file) if report_file else ""
    extracted = extract_symptoms_from_text(text)
    manual = normalize(manual_symptoms or [])
    symptoms = sorted(set(extracted + manual))
    iq = image_quality(image_file)
    if not symptoms:
        msg = "Could not detect known symptoms from the report. Add symptoms manually."
        detail = {
            "timestamp_utc": datetime.utcnow().isoformat(),
            "source": "report_image_assisted",
            "selected_symptoms": [],
            "predicted_disease": "ABSTAINED",
            "abstained": True,
            "abstain_reason": msg,
            "supportive_care_plan": supportive_care([], True),
        }
        summary = msg + "\n\n" + iq
        pdf_path = write_pdf(detail, "No precautions available.", summary, detail["supportive_care_plan"], "report")
        return msg, "{}", "No precautions available.", "", summary, pdf_path
    probs, details, precautions, summary, pdf_path = infer(symptoms, abstain_threshold, source_note="report_image_assisted")
    return probs, details, precautions, ", ".join(symptoms), summary + "\n\n" + iq, pdf_path

with gr.Blocks(title="SecureHealthIoT Disease Predictor") as demo:
    gr.Markdown("# SecureHealthIoT Disease Predictor")
    gr.Markdown("Symptom-based predictor with report/image-assisted analysis, abstain mode, and PDF export.")
    gr.Markdown("**Medical disclaimer:** Any prescription decision must be made by a licensed physician.")

    with gr.Tabs():
        with gr.Tab("Quick Symptom Predictor"):
            inp = gr.Dropdown(choices=SYMPTOMS, multiselect=True, label="Symptoms")
            thr = gr.Slider(minimum=0.3, maximum=0.9, value=0.55, step=0.01, label="Abstain Threshold")
            btn = gr.Button("Predict", variant="primary")
            out1 = gr.Label(label="Top Predictions")
            out2 = gr.Code(label="Prediction Details", language="json")
            out3 = gr.Markdown(label="Precautions")
            out4 = gr.Markdown(label="Summary")
            out5 = gr.File(label="Clinician PDF Report")
            btn.click(infer, inputs=[inp, thr], outputs=[out1, out2, out3, out4, out5])

        with gr.Tab("Report/Image Assisted Analysis"):
            rf = gr.File(label="Medical Report File", file_types=[".txt", ".csv", ".pdf"])
            im = gr.Image(type="filepath", label="Optional Clinical Image")
            ms = gr.Dropdown(choices=SYMPTOMS, multiselect=True, label="Manual Symptoms (optional)")
            thr2 = gr.Slider(minimum=0.3, maximum=0.9, value=0.55, step=0.01, label="Abstain Threshold")
            run = gr.Button("Analyze Report + Predict", variant="primary")
            rep1 = gr.Label(label="Top Predictions")
            rep2 = gr.Code(label="Prediction Details", language="json")
            rep3 = gr.Markdown(label="Precautions")
            rep4 = gr.Textbox(label="Detected Symptoms Used")
            rep5 = gr.Markdown(label="Report Summary")
            rep6 = gr.File(label="Clinician PDF Report")
            run.click(analyze_report, inputs=[rf, im, ms, thr2], outputs=[rep1, rep2, rep3, rep4, rep5, rep6])

if __name__ == "__main__":
    demo.launch()
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space-id", required=True, help="Example: ShiroOnigami23/securehealthiot-disease-app")
    parser.add_argument("--model-repo-id", required=True, help="Example: ShiroOnigami23/securehealthiot-disease-model")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN env var is required.")

    create_repo(args.space_id, token=token, repo_type="space", space_sdk="gradio", exist_ok=True)
    api = HfApi(token=token)

    readme = f"""---
title: SecureHealthIoT Disease Predictor
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
python_version: 3.11
app_file: app.py
pinned: false
---

# SecureHealthIoT Disease Predictor

Symptom-based disease prediction Space with report/image-assisted analysis, abstain mode, and clinician PDF export.
Model source: `{args.model_repo_id}`.
"""

    app_code = SPACE_APP.replace("REPLACE_MODEL_REPO", args.model_repo_id)

    tmp = Path("artifacts/space_publish_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "README.md").write_text(readme, encoding="utf-8")
    (tmp / "app.py").write_text(app_code, encoding="utf-8")
    (tmp / "requirements.txt").write_text(
        "gradio\njoblib\nhuggingface_hub\nscikit-learn\nnumpy\npypdf\nPillow\nreportlab\n",
        encoding="utf-8",
    )

    api.upload_folder(folder_path=str(tmp), repo_id=args.space_id, repo_type="space")
    print(f"Published Space: https://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()
