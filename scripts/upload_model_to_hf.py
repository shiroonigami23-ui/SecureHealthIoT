from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="Example: ShiroOnigami23/securehealthiot-disease-model")
    parser.add_argument("--bundle-path", default="artifacts/latest_model.joblib")
    parser.add_argument("--registry-dir", default="artifacts/model_registry")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN env var is required.")

    bundle = Path(args.bundle_path)
    if not bundle.exists():
        raise FileNotFoundError(f"Model bundle not found: {bundle}")

    create_repo(args.repo_id, token=token, private=args.private, exist_ok=True)
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(bundle),
        path_in_repo="latest_model.joblib",
        repo_id=args.repo_id,
        repo_type="model",
    )

    reg = Path(args.registry_dir)
    if reg.exists():
        for metrics_file in reg.glob("*/metrics.json"):
            rel = metrics_file.relative_to(reg)
            api.upload_file(
                path_or_fileobj=str(metrics_file),
                path_in_repo=f"metrics_history/{rel.as_posix()}",
                repo_id=args.repo_id,
                repo_type="model",
            )

    card = {
        "model_type": "symptom-based multi-class disease classifier",
        "bundle_file": "latest_model.joblib",
        "framework": "scikit-learn",
    }
    card_path = Path("artifacts/model_card.json")
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="model_card.json",
        repo_id=args.repo_id,
        repo_type="model",
    )

    print(f"Uploaded model to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()

