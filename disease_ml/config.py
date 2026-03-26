from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    kaggle_dataset: str = "itachi9604/disease-symptom-description-dataset"
    data_dir: str = "data/kaggle_raw"
    train_csv: str = "data/kaggle_raw/dataset.csv"
    severity_csv: str = "data/kaggle_raw/Symptom-severity.csv"
    description_csv: str = "data/kaggle_raw/symptom_Description.csv"
    precaution_csv: str = "data/kaggle_raw/symptom_precaution.csv"


@dataclass(frozen=True)
class TrainConfig:
    random_seed: int = 42
    test_size: float = 0.2
    model_registry_dir: str = "artifacts/model_registry"
    latest_bundle_path: str = "artifacts/latest_model.joblib"

