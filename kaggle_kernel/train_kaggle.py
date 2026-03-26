import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_val_score, train_test_split
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


def grouped_split_indices(y, groups, test_size: float, seed: int):
    if len(set(groups)) > 10:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        idx_a, idx_b = next(gss.split(np.zeros(len(y)), y, groups))
        return np.asarray(idx_a), np.asarray(idx_b)
    return train_test_split(np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y)


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    one_hot = np.eye(probs.shape[1], dtype=np.float32)[y_true]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


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

    cleaned_path = WORK_DIR / "cleaned_dataset.csv"
    df.to_csv(cleaned_path, index=False)

    samples = []
    labels = []
    signatures = []
    for row in df.itertuples(index=False):
        disease = row[0]
        syms = sorted(set([s for s in row[1:] if s]))
        if syms:
            samples.append(syms)
            labels.append(disease)
            signatures.append("|".join(syms))

    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(samples).astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    dev_idx, ext_idx = grouped_split_indices(y, signatures, test_size=0.15, seed=42)
    X_dev, y_dev = X[dev_idx], y[dev_idx]
    sig_dev = [signatures[i] for i in dev_idx]
    X_ext, y_ext = X[ext_idx], y[ext_idx]
    sig_ext = [signatures[i] for i in ext_idx]

    train_rel, val_rel = grouped_split_indices(y_dev, sig_dev, test_size=0.2, seed=42)
    X_train, y_train = X_dev[train_rel], y_dev[train_rel]
    X_val, y_val = X_dev[val_rel], y_dev[val_rel]
    sig_train = [sig_dev[i] for i in train_rel]
    sig_val = [sig_dev[i] for i in val_rel]

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

    for name, model in candidates.items():
        cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        holdout_acc = accuracy_score(y_val, preds)
        holdout_f1 = f1_score(y_val, preds, average="weighted")
        results[name] = {
            "cv_acc_mean": float(np.mean(cv_acc)),
            "holdout_acc": float(holdout_acc),
            "holdout_f1_weighted": float(holdout_f1),
        }
        if holdout_acc > best_score:
            best_score = holdout_acc
            best_name = name
            best_model = model

    calibrated = CalibratedClassifierCV(estimator=clone(best_model), method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)
    val_pred = calibrated.predict(X_val)
    val_prob = calibrated.predict_proba(X_val)
    ext_pred = calibrated.predict(X_ext)
    ext_prob = calibrated.predict_proba(X_ext)

    metrics = {
        "best_model": best_name,
        "results": results,
        "num_samples": int(len(samples)),
        "num_classes": int(len(le.classes_)),
        "num_symptoms": int(len(mlb.classes_)),
        "dataset": "itachi9604/disease-symptom-description-dataset",
        "accelerator_requested": accelerator,
        "validation": {
            "internal_acc": float(accuracy_score(y_val, val_pred)),
            "internal_f1_weighted": float(f1_score(y_val, val_pred, average="weighted")),
            "internal_log_loss": float(
                log_loss(y_val, val_prob, labels=np.arange(len(le.classes_)))
            ),
            "internal_brier_multi": multiclass_brier(y_val, val_prob),
            "external_acc": float(accuracy_score(y_ext, ext_pred)),
            "external_f1_weighted": float(f1_score(y_ext, ext_pred, average="weighted")),
            "external_log_loss": float(
                log_loss(y_ext, ext_prob, labels=np.arange(len(le.classes_)))
            ),
            "external_brier_multi": multiclass_brier(y_ext, ext_prob),
        },
        "data_leakage_checks": {
            "duplicate_symptom_signature_rows": int(len(signatures) - len(set(signatures))),
            "overlap_signature_train_val": int(len(set(sig_train) & set(sig_val))),
            "overlap_signature_train_external": int(len(set(sig_train) & set(sig_ext))),
            "overlap_signature_val_external": int(len(set(sig_val) & set(sig_ext))),
            "external_split_strategy": "grouped_by_symptom_signature",
        },
        "classification_report_external": classification_report(y_ext, ext_pred, output_dict=True),
    }

    bundle = {
        "model": calibrated,
        "label_encoder": le,
        "symptom_binarizer": mlb,
        "metrics": metrics,
        "calibration": {"method": "sigmoid", "cv": 3},
        "aux": {"descriptions": {}, "precautions": {}},
    }

    joblib.dump(bundle, WORK_DIR / "model_bundle.joblib")
    (WORK_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print("Saved:", WORK_DIR / "model_bundle.joblib")
    print("Saved:", WORK_DIR / "metrics.json")
    print("Saved:", cleaned_path)


if __name__ == "__main__":
    main()
