from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


SPACE_APP = r'''
import json
import gradio as gr
import joblib
from huggingface_hub import hf_hub_download

MODEL_REPO_ID = "REPLACE_MODEL_REPO"

bundle_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename="latest_model.joblib")
bundle = joblib.load(bundle_path)
model = bundle["model"]
label_encoder = bundle["label_encoder"]
vectorizer = bundle["vectorizer"]
aux = bundle["aux"]

def predict(symptoms):
    if not symptoms:
        return "Please select at least one symptom.", "", ""
    normalized = sorted(set([s.strip().lower().replace(" ", "_") for s in symptoms if s.strip()]))
    X = vectorizer.transform([normalized])
    probs = model.predict_proba(X)[0]
    top_idx = probs.argsort()[::-1][:3]
    pred = []
    for idx in top_idx:
        disease = label_encoder.inverse_transform([idx])[0]
        pred.append((disease, float(probs[idx])))
    top_disease = pred[0][0]
    details = {
        "predicted_disease": top_disease,
        "probability": pred[0][1],
        "description": aux["descriptions"].get(top_disease, ""),
        "precautions": aux["precautions"].get(top_disease, []),
    }
    pretty = {d: round(p, 4) for d, p in pred}
    precaution_text = "\n".join([f"- {p}" for p in details["precautions"]]) if details["precautions"] else "N/A"
    return pretty, json.dumps(details, indent=2), precaution_text

symptoms = list(vectorizer.mlb.classes_)

with gr.Blocks(title="SecureHealthIoT Disease Predictor") as demo:
    gr.Markdown("# SecureHealthIoT Disease Predictor")
    inp = gr.Dropdown(choices=symptoms, multiselect=True, label="Symptoms")
    btn = gr.Button("Predict", variant="primary")
    out1 = gr.Label(label="Top Predictions")
    out2 = gr.Code(label="Prediction Details", language="json")
    out3 = gr.Markdown(label="Precautions")
    btn.click(predict, inputs=[inp], outputs=[out1, out2, out3])

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

Symptom-based disease prediction Space.
Model source: `{args.model_repo_id}`.
"""

    app_code = SPACE_APP.replace("REPLACE_MODEL_REPO", args.model_repo_id)

    tmp = Path("artifacts/space_publish_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "README.md").write_text(readme, encoding="utf-8")
    (tmp / "app.py").write_text(app_code, encoding="utf-8")
    (tmp / "requirements.txt").write_text("gradio\njoblib\nhuggingface_hub\nscikit-learn\nnumpy\n", encoding="utf-8")

    api.upload_folder(folder_path=str(tmp), repo_id=args.space_id, repo_type="space")
    print(f"Published Space: https://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()

