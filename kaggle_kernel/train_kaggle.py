import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


INPUT_ROOT = Path("/kaggle/input")
WORK_DIR = Path("/kaggle/working")
WORK_DIR.mkdir(parents=True, exist_ok=True)


def clean_symptom(x: str) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower().replace(" ", "_")


def main():
    dataset_candidates = list(INPUT_ROOT.rglob("dataset.csv"))
    if not dataset_candidates:
        raise FileNotFoundError("Could not find dataset.csv under /kaggle/input")
    data_csv = dataset_candidates[0]
    df = pd.read_csv(data_csv)
    symptom_cols = [c for c in df.columns if c.startswith("Symptom_")]
    for c in symptom_cols:
        df[c] = df[c].map(clean_symptom)
    df["Disease"] = df["Disease"].astype(str).str.strip()

    # save cleaned dataset for reproducibility
    cleaned_path = WORK_DIR / "cleaned_dataset.csv"
    df.to_csv(cleaned_path, index=False)

    samples = []
    labels = []
    for row in df.itertuples(index=False):
        d = row[0]
        syms = sorted(set([s for s in row[1:] if s]))
        if syms:
            samples.append(syms)
            labels.append(d)

    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(samples).astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    candidates = {
        "logreg": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(
            n_estimators=800, random_state=42, n_jobs=-1, class_weight="balanced_subsample"
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=1000, random_state=42, n_jobs=-1, class_weight="balanced"
        ),
    }
    accelerator = "cpu"
    if XGBClassifier is not None:
        candidates["xgboost_gpu"] = XGBClassifier(
            n_estimators=600,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            device="cuda",
            random_state=42,
            n_jobs=1,
        )
        accelerator = "gpu"

    results = {}
    best_name = None
    best_score = -1.0
    best_model = None
    best_preds = None

    for name, model in candidates.items():
        cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        holdout_acc = accuracy_score(y_test, preds)
        holdout_f1 = f1_score(y_test, preds, average="weighted")
        results[name] = {
            "cv_acc_mean": float(np.mean(cv_acc)),
            "holdout_acc": float(holdout_acc),
            "holdout_f1_weighted": float(holdout_f1),
        }
        if holdout_acc > best_score:
            best_score = holdout_acc
            best_name = name
            best_model = model
            best_preds = preds

    metrics = {
        "best_model": best_name,
        "results": results,
        "num_samples": int(len(samples)),
        "num_classes": int(len(le.classes_)),
        "num_symptoms": int(len(mlb.classes_)),
        "dataset": "itachi9604/disease-symptom-description-dataset",
        "accelerator_requested": accelerator,
        "classification_report": classification_report(y_test, best_preds, output_dict=True),
    }

    bundle = {
        "model": best_model,
        "label_encoder": le,
        "symptom_binarizer": mlb,
        "metrics": metrics,
    }

    joblib.dump(bundle, WORK_DIR / "model_bundle.joblib")
    (WORK_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print("Saved:", WORK_DIR / "model_bundle.joblib")
    print("Saved:", WORK_DIR / "metrics.json")
    print("Saved:", cleaned_path)


if __name__ == "__main__":
    main()
