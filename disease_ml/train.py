from __future__ import annotations

import argparse
import difflib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

from disease_ml.config import DataConfig, TrainConfig
from disease_ml.data import ensure_kaggle_dataset, extract_samples, load_aux_tables, load_training_dataframe
from disease_ml.features import SymptomVectorizer


@dataclass
class CandidateResult:
    name: str
    cv_acc_mean: float
    holdout_acc: float
    holdout_f1_weighted: float


def build_candidates(seed: int):
    return {
        "logreg": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(
            n_estimators=800, random_state=seed, n_jobs=-1, class_weight="balanced_subsample"
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=1000, random_state=seed, n_jobs=-1, class_weight="balanced"
        ),
    }


def evaluate_candidates(X_train, y_train, X_val, y_val, seed: int) -> Tuple[str, Dict[str, CandidateResult], object]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    results: Dict[str, CandidateResult] = {}
    fitted_models = {}

    for name, model in build_candidates(seed).items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        results[name] = CandidateResult(
            name=name,
            cv_acc_mean=float(np.mean(cv_scores)),
            holdout_acc=float(accuracy_score(y_val, preds)),
            holdout_f1_weighted=float(f1_score(y_val, preds, average="weighted")),
        )
        fitted_models[name] = model

    best_name = max(results.keys(), key=lambda k: (results[k].holdout_acc, results[k].cv_acc_mean))
    return best_name, results, fitted_models[best_name]


def _make_grouped_split_indices(y, groups, test_size: float, seed: int):
    if len(set(groups)) > 10:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        idx_a, idx_b = next(gss.split(np.zeros(len(y)), y, groups))
        return np.asarray(idx_a), np.asarray(idx_b)
    return train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )


def _multiclass_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    one_hot = np.eye(probs.shape[1], dtype=np.float32)[y_true]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def _normalize_label(s: str) -> str:
    t = str(s).strip().lower().replace("_", " ")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _align_label(norm_label: str, available_norm_labels: list[str]) -> str | None:
    if norm_label in available_norm_labels:
        return norm_label
    match = difflib.get_close_matches(norm_label, available_norm_labels, n=1, cutoff=0.82)
    return match[0] if match else None


def evaluate_ood_dataset(calibrated, label_encoder, vectorizer, ood_df):
    ood_samples, ood_labels = extract_samples(ood_df)
    if not ood_samples:
        return {"available": False, "reason": "No valid rows in OOD dataset"}

    train_label_norm = {_normalize_label(x): x for x in label_encoder.classes_}
    train_keys = list(train_label_norm.keys())
    ood_norm = [_normalize_label(x) for x in ood_labels]
    aligned = [_align_label(lbl, train_keys) for lbl in ood_norm]
    keep_idx = [i for i, lbl in enumerate(aligned) if lbl is not None]
    if len(keep_idx) < 20:
        return {"available": False, "reason": "Insufficient overlapping classes with training dataset"}

    X_ood = vectorizer.transform([ood_samples[i] for i in keep_idx])
    y_ood_labels = [train_label_norm[aligned[i]] for i in keep_idx]
    y_ood = label_encoder.transform(y_ood_labels)

    pred = calibrated.predict(X_ood)
    prob = calibrated.predict_proba(X_ood)
    return {
        "available": True,
        "num_samples": int(len(keep_idx)),
        "num_classes": int(len(set(y_ood_labels))),
        "acc": float(accuracy_score(y_ood, pred)),
        "f1_weighted": float(f1_score(y_ood, pred, average="weighted")),
        "log_loss": float(log_loss(y_ood, prob, labels=np.arange(len(label_encoder.classes_)))),
        "brier_multi": _multiclass_brier(y_ood, prob),
    }


def run_training(data_cfg: DataConfig, train_cfg: TrainConfig, out_note: str = "") -> Dict:
    ensure_kaggle_dataset(data_cfg.kaggle_dataset, data_cfg.data_dir)

    df = load_training_dataframe(data_cfg.train_csv)
    samples, labels = extract_samples(df)
    signatures = ["|".join(s) for s in samples]

    vectorizer = SymptomVectorizer()
    X = vectorizer.fit_transform(samples)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    dev_idx, ext_idx = _make_grouped_split_indices(
        y, signatures, test_size=train_cfg.external_val_size, seed=train_cfg.random_seed
    )
    X_dev, y_dev = X[dev_idx], y[dev_idx]
    sig_dev = [signatures[i] for i in dev_idx]
    X_ext, y_ext = X[ext_idx], y[ext_idx]
    sig_ext = [signatures[i] for i in ext_idx]

    train_rel, val_rel = _make_grouped_split_indices(
        y_dev, sig_dev, test_size=train_cfg.test_size, seed=train_cfg.random_seed
    )
    X_train, y_train = X_dev[train_rel], y_dev[train_rel]
    X_val, y_val = X_dev[val_rel], y_dev[val_rel]
    sig_train = [sig_dev[i] for i in train_rel]
    sig_val = [sig_dev[i] for i in val_rel]

    best_name, results, best_model = evaluate_candidates(
        X_train, y_train, X_val, y_val, seed=train_cfg.random_seed
    )
    calibrated = CalibratedClassifierCV(estimator=clone(best_model), method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)

    val_pred = calibrated.predict(X_val)
    val_prob = calibrated.predict_proba(X_val)
    ext_pred = calibrated.predict(X_ext)
    ext_prob = calibrated.predict_proba(X_ext)
    ood_report = {"available": False, "reason": "OOD validation disabled"}
    if train_cfg.enable_ood_validation:
        ensure_kaggle_dataset(data_cfg.ood_kaggle_dataset, data_cfg.ood_data_dir)
        ood_df = load_training_dataframe(data_cfg.ood_csv)
        ood_report = evaluate_ood_dataset(calibrated, label_encoder, vectorizer, ood_df)

    aux = load_aux_tables(data_cfg.description_csv, data_cfg.precaution_csv)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    version_dir = Path(train_cfg.model_registry_dir) / f"{ts}_{best_name}"
    version_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "best_model": best_name,
        "results": {k: asdict(v) for k, v in results.items()},
        "num_samples": int(len(samples)),
        "num_classes": int(len(label_encoder.classes_)),
        "num_symptoms": int(len(vectorizer.symptom_vocab)),
        "dataset_ref": data_cfg.kaggle_dataset,
        "ood_dataset_ref": data_cfg.ood_kaggle_dataset,
        "validation": {
            "internal_acc": float(accuracy_score(y_val, val_pred)),
            "internal_f1_weighted": float(f1_score(y_val, val_pred, average="weighted")),
            "internal_log_loss": float(
                log_loss(y_val, val_prob, labels=np.arange(len(label_encoder.classes_)))
            ),
            "internal_brier_multi": _multiclass_brier(y_val, val_prob),
            "external_acc": float(accuracy_score(y_ext, ext_pred)),
            "external_f1_weighted": float(f1_score(y_ext, ext_pred, average="weighted")),
            "external_log_loss": float(
                log_loss(y_ext, ext_prob, labels=np.arange(len(label_encoder.classes_)))
            ),
            "external_brier_multi": _multiclass_brier(y_ext, ext_prob),
        },
        "ood_validation": ood_report,
        "data_leakage_checks": {
            "total_rows_after_cleaning": int(len(samples)),
            "duplicate_symptom_signature_rows": int(len(signatures) - len(set(signatures))),
            "overlap_signature_train_val": int(len(set(sig_train) & set(sig_val))),
            "overlap_signature_train_external": int(len(set(sig_train) & set(sig_ext))),
            "overlap_signature_val_external": int(len(set(sig_val) & set(sig_ext))),
            "external_split_strategy": "grouped_by_symptom_signature",
        },
        "note": out_note,
        "trained_at_utc": ts,
    }
    if metrics["validation"]["external_acc"] < 0.7 or (
        metrics["ood_validation"].get("available") and metrics["ood_validation"].get("acc", 0.0) < 0.7
    ):
        metrics["reliability_note"] = (
            "External validation is weak; retrain with richer data before real-world use."
        )
    else:
        metrics["reliability_note"] = (
            "Model passed external validation on held-out symptom signatures. "
            "Use as decision support only, not definitive diagnosis."
        )

    bundle = {
        "model": calibrated,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "metrics": metrics,
        "aux": aux,
        "calibration": {"method": "sigmoid", "cv": 3},
    }

    bundle_path = version_dir / "model_bundle.joblib"
    metrics_path = version_dir / "metrics.json"
    latest_path = Path(train_cfg.latest_bundle_path)
    latest_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, bundle_path)
    joblib.dump(bundle, latest_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {"bundle_path": str(bundle_path), "latest_path": str(latest_path), "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-ref", default=DataConfig.kaggle_dataset)
    parser.add_argument("--data-dir", default=DataConfig.data_dir)
    parser.add_argument("--train-csv", default=DataConfig.train_csv)
    parser.add_argument("--ood-dataset-ref", default=DataConfig.ood_kaggle_dataset)
    parser.add_argument("--ood-data-dir", default=DataConfig.ood_data_dir)
    parser.add_argument("--ood-csv", default=DataConfig.ood_csv)
    parser.add_argument("--description-csv", default=DataConfig.description_csv)
    parser.add_argument("--precaution-csv", default=DataConfig.precaution_csv)
    parser.add_argument("--test-size", type=float, default=TrainConfig.test_size)
    parser.add_argument("--external-val-size", type=float, default=TrainConfig.external_val_size)
    parser.add_argument("--disable-ood-validation", action="store_true")
    parser.add_argument("--seed", type=int, default=TrainConfig.random_seed)
    parser.add_argument("--registry-dir", default=TrainConfig.model_registry_dir)
    parser.add_argument("--latest-path", default=TrainConfig.latest_bundle_path)
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    data_cfg = DataConfig(
        kaggle_dataset=args.dataset_ref,
        data_dir=args.data_dir,
        train_csv=args.train_csv,
        ood_kaggle_dataset=args.ood_dataset_ref,
        ood_data_dir=args.ood_data_dir,
        ood_csv=args.ood_csv,
        severity_csv=DataConfig.severity_csv,
        description_csv=args.description_csv,
        precaution_csv=args.precaution_csv,
    )
    train_cfg = TrainConfig(
        random_seed=args.seed,
        test_size=args.test_size,
        external_val_size=args.external_val_size,
        enable_ood_validation=not args.disable_ood_validation,
        model_registry_dir=args.registry_dir,
        latest_bundle_path=args.latest_path,
    )

    output = run_training(data_cfg, train_cfg, out_note=args.note)
    print(json.dumps(output["metrics"], indent=2))
    print(f"Saved versioned bundle: {output['bundle_path']}")
    print(f"Saved latest bundle: {output['latest_path']}")


if __name__ == "__main__":
    main()
