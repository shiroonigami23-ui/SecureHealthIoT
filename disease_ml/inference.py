from __future__ import annotations

from typing import List

import joblib
import numpy as np


class DiseasePredictor:
    def __init__(self, bundle_path: str = "artifacts/latest_model.joblib"):
        bundle = joblib.load(bundle_path)
        self.model = bundle["model"]
        self.label_encoder = bundle["label_encoder"]
        self.vectorizer = bundle["vectorizer"]
        self.metrics = bundle["metrics"]
        self.aux = bundle["aux"]

    def predict_top_k(self, symptoms: List[str], k: int = 3):
        symptoms = sorted(set([s.strip().lower().replace(" ", "_") for s in symptoms if s.strip()]))
        X = self.vectorizer.transform([symptoms])
        probs = self.model.predict_proba(X)[0]
        top_idx = np.argsort(probs)[::-1][:k]
        out = []
        for idx in top_idx:
            disease = self.label_encoder.inverse_transform([idx])[0]
            out.append(
                {
                    "disease": disease,
                    "probability": float(probs[idx]),
                    "description": self.aux["descriptions"].get(disease, ""),
                    "precautions": self.aux["precautions"].get(disease, []),
                }
            )
        return out

