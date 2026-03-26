from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from disease_ml.inference import DiseasePredictor


BUNDLE_PATH = "artifacts/latest_model.joblib"

if not Path(BUNDLE_PATH).exists():
    raise RuntimeError(
        "Model bundle not found. Run: python -m disease_ml.train\n"
        "This will create artifacts/latest_model.joblib."
    )

predictor = DiseasePredictor(BUNDLE_PATH)
symptom_vocab = predictor.vectorizer.symptom_vocab


def _format_precautions(items):
    if not items:
        return "No precaution data available."
    return "\n".join([f"- {p}" for p in items])


def infer(symptoms):
    if not symptoms:
        return "Please select at least one symptom.", "", ""

    preds = predictor.predict_top_k(symptoms, k=3)
    probs = {x["disease"]: round(x["probability"], 4) for x in preds}
    top = preds[0]
    detail = {
        "predicted_disease": top["disease"],
        "probability": top["probability"],
        "description": top["description"],
        "precautions": top["precautions"],
    }
    return (
        probs,
        json.dumps(detail, indent=2),
        _format_precautions(top["precautions"]),
    )


with gr.Blocks(title="SecureHealthIoT Disease Predictor") as demo:
    gr.Markdown("# SecureHealthIoT: Disease Prediction")
    gr.Markdown(
        "Select symptoms and get top disease predictions. "
        "This model is trained on Kaggle dataset `itachi9604/disease-symptom-description-dataset`."
    )

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

    submit.click(infer, inputs=[symptom_input], outputs=[prob_out, detail_out, precautions_out])
    clear.click(lambda: (None, "", ""), inputs=[], outputs=[symptom_input, detail_out, precautions_out])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

