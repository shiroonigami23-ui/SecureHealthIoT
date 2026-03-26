from __future__ import annotations

from typing import List

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


class SymptomVectorizer:
    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def fit(self, samples: List[List[str]]) -> "SymptomVectorizer":
        self.mlb.fit(samples)
        return self

    def transform(self, samples: List[List[str]]) -> np.ndarray:
        return self.mlb.transform(samples).astype(np.float32)

    def fit_transform(self, samples: List[List[str]]) -> np.ndarray:
        return self.mlb.fit_transform(samples).astype(np.float32)

    @property
    def symptom_vocab(self) -> List[str]:
        return list(self.mlb.classes_)

