from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
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


def evaluate_candidates(X_train, y_train, X_test, y_test, seed: int) -> Tuple[str, Dict[str, CandidateResult], object]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    results: Dict[str, CandidateResult] = {}
    fitted_models = {}

    for name, model in build_candidates(seed).items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = CandidateResult(
            name=name,
            cv_acc_mean=float(np.mean(cv_scores)),
            holdout_acc=float(accuracy_score(y_test, preds)),
            holdout_f1_weighted=float(f1_score(y_test, preds, average="weighted")),
        )
        fitted_models[name] = model

    best_name = max(results.keys(), key=lambda k: (results[k].holdout_acc, results[k].cv_acc_mean))
    return best_name, results, fitted_models[best_name]


def run_training(data_cfg: DataConfig, train_cfg: TrainConfig, out_note: str = "") -> Dict:
    ensure_kaggle_dataset(data_cfg.kaggle_dataset, data_cfg.data_dir)

    df = load_training_dataframe(data_cfg.train_csv)
    samples, labels = extract_samples(df)
    vectorizer = SymptomVectorizer()
    X = vectorizer.fit_transform(samples)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=train_cfg.test_size, random_state=train_cfg.random_seed, stratify=y
    )

    best_name, results, best_model = evaluate_candidates(
        X_train, y_train, X_test, y_test, seed=train_cfg.random_seed
    )

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
        "note": out_note,
        "trained_at_utc": ts,
    }

    bundle = {
        "model": best_model,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "metrics": metrics,
        "aux": aux,
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
    parser.add_argument("--description-csv", default=DataConfig.description_csv)
    parser.add_argument("--precaution-csv", default=DataConfig.precaution_csv)
    parser.add_argument("--test-size", type=float, default=TrainConfig.test_size)
    parser.add_argument("--seed", type=int, default=TrainConfig.random_seed)
    parser.add_argument("--registry-dir", default=TrainConfig.model_registry_dir)
    parser.add_argument("--latest-path", default=TrainConfig.latest_bundle_path)
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    data_cfg = DataConfig(
        kaggle_dataset=args.dataset_ref,
        data_dir=args.data_dir,
        train_csv=args.train_csv,
        severity_csv=DataConfig.severity_csv,
        description_csv=args.description_csv,
        precaution_csv=args.precaution_csv,
    )
    train_cfg = TrainConfig(
        random_seed=args.seed,
        test_size=args.test_size,
        model_registry_dir=args.registry_dir,
        latest_bundle_path=args.latest_path,
    )

    output = run_training(data_cfg, train_cfg, out_note=args.note)
    print(json.dumps(output["metrics"], indent=2))
    print(f"Saved versioned bundle: {output['bundle_path']}")
    print(f"Saved latest bundle: {output['latest_path']}")


if __name__ == "__main__":
    main()
