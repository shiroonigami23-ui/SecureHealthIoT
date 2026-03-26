from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd


SYMPTOM_PREFIX = "Symptom_"
PRESENT_STRINGS = {"1", "true", "yes", "y", "present", "positive"}


def ensure_kaggle_dataset(dataset_ref: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("*.csv")):
        return
    cmd = ["kaggle", "datasets", "download", "-d", dataset_ref, "-p", str(out), "--unzip"]
    subprocess.run(cmd, check=True)


def _resolve_disease_col(columns: List[str]) -> str:
    candidates = ["Disease", "disease", "prognosis", "Prognosis", "label", "Label"]
    for c in candidates:
        if c in columns:
            return c
    raise KeyError("Could not detect disease/label column in dataframe.")


def load_training_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    disease_col = _resolve_disease_col(list(df.columns))
    if disease_col != "Disease":
        df = df.rename(columns={disease_col: "Disease"})

    symptom_cols = [c for c in df.columns if c.startswith(SYMPTOM_PREFIX)]
    for col in symptom_cols:
        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )

    df["Disease"] = (
        df["Disease"]
        .astype(str)
        .str.strip()
        .str.replace("_", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )
    return df


def _is_present(v) -> bool:
    if pd.isna(v):
        return False
    if isinstance(v, (int, float)):
        return float(v) > 0.0
    s = str(v).strip().lower()
    return s in PRESENT_STRINGS


def _normalize_symptom_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("__", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def extract_samples(df: pd.DataFrame) -> tuple[List[List[str]], List[str]]:
    symptom_cols = [c for c in df.columns if c.startswith(SYMPTOM_PREFIX)]
    one_hot_cols = [c for c in df.columns if c != "Disease"]

    samples: List[List[str]] = []
    labels: List[str] = []
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        if symptom_cols:
            symptoms = [row_dict[c] for c in symptom_cols if row_dict[c]]
        else:
            symptoms = [_normalize_symptom_name(c) for c in one_hot_cols if _is_present(row_dict[c])]

        symptoms = sorted(set(symptoms))
        if symptoms:
            samples.append(symptoms)
            labels.append(str(row_dict["Disease"]).strip())
    return samples, labels


def load_aux_tables(description_csv: str, precaution_csv: str) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {"descriptions": {}, "precautions": {}}

    desc = pd.read_csv(description_csv)
    desc_disease_col = _resolve_disease_col(list(desc.columns))
    for row in desc.itertuples(index=False):
        disease = str(getattr(row, desc_disease_col)).strip()
        description = str(getattr(row, "Description")).strip()
        out["descriptions"][disease] = description

    pre = pd.read_csv(precaution_csv)
    pre_disease_col = _resolve_disease_col(list(pre.columns))
    for row in pre.itertuples(index=False):
        disease = str(getattr(row, pre_disease_col)).strip()
        vals = [
            str(getattr(row, "Precaution_1", "")).strip(),
            str(getattr(row, "Precaution_2", "")).strip(),
            str(getattr(row, "Precaution_3", "")).strip(),
            str(getattr(row, "Precaution_4", "")).strip(),
        ]
        out["precautions"][disease] = [v for v in vals if v and v.lower() != "nan"]

    return out
