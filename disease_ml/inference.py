from __future__ import annotations

from typing import List

import joblib
import numpy as np


class DiseasePredictor:
    def __init__(self, bundle_path: str = "artifacts/latest_model.joblib"):
        bundle = joblib.load(bundle_path)
        self.model = bundle["model"]
        self.label_encoder = bundle["label_encoder"]
        self.vectorizer = bundle.get("vectorizer")
        self.symptom_binarizer = bundle.get("symptom_binarizer")
        self.metrics = bundle["metrics"]
        self.aux = bundle.get("aux", {"descriptions": {}, "precautions": {}})

    @property
    def symptom_vocab(self) -> List[str]:
        if self.vectorizer is not None and hasattr(self.vectorizer, "symptom_vocab"):
            return list(self.vectorizer.symptom_vocab)
        if self.symptom_binarizer is not None and hasattr(self.symptom_binarizer, "classes_"):
            return list(self.symptom_binarizer.classes_)
        return []

    def _transform(self, symptoms: List[str]):
        if self.vectorizer is not None:
            return self.vectorizer.transform([symptoms])
        if self.symptom_binarizer is not None:
            return self.symptom_binarizer.transform([symptoms]).astype(np.float32)
        raise RuntimeError("No vectorizer/symptom_binarizer found in model bundle.")

    def predict_top_k(self, symptoms: List[str], k: int = 3):
        symptoms = sorted(set([s.strip().lower().replace(" ", "_") for s in symptoms if s.strip()]))
        X = self._transform(symptoms)
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
