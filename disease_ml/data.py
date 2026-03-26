from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd


SYMPTOM_PREFIX = "Symptom_"


def ensure_kaggle_dataset(dataset_ref: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset_csv = out / "dataset.csv"
    if dataset_csv.exists():
        return
    cmd = ["kaggle", "datasets", "download", "-d", dataset_ref, "-p", str(out), "--unzip"]
    subprocess.run(cmd, check=True)


def load_training_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
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
    df["Disease"] = df["Disease"].astype(str).str.strip()
    return df


def extract_samples(df: pd.DataFrame) -> tuple[List[List[str]], List[str]]:
    symptom_cols = [c for c in df.columns if c.startswith(SYMPTOM_PREFIX)]
    samples: List[List[str]] = []
    labels: List[str] = []
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        symptoms = [row_dict[c] for c in symptom_cols if row_dict[c]]
        symptoms = sorted(set(symptoms))
        if symptoms:
            samples.append(symptoms)
            labels.append(row_dict["Disease"])
    return samples, labels


def load_aux_tables(description_csv: str, precaution_csv: str) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {"descriptions": {}, "precautions": {}}

    desc = pd.read_csv(description_csv)
    for row in desc.itertuples(index=False):
        disease = str(getattr(row, "Disease")).strip()
        description = str(getattr(row, "Description")).strip()
        out["descriptions"][disease] = description

    pre = pd.read_csv(precaution_csv)
    for row in pre.itertuples(index=False):
        disease = str(getattr(row, "Disease")).strip()
        vals = [
            str(getattr(row, "Precaution_1", "")).strip(),
            str(getattr(row, "Precaution_2", "")).strip(),
            str(getattr(row, "Precaution_3", "")).strip(),
            str(getattr(row, "Precaution_4", "")).strip(),
        ]
        out["precautions"][disease] = [v for v in vals if v and v.lower() != "nan"]

    return out

