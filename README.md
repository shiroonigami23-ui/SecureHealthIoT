# SecureHealthIoT Disease Prediction

End-to-end symptom-based disease prediction pipeline with:

- Kaggle dataset ingestion
- reproducible training + model versioning
- Gradio inference app
- Hugging Face Model + Space publishing scripts

## Live Deployment

- Hugging Face Model: https://huggingface.co/ShiroOnigami23/securehealthiot-disease-model
- Hugging Face Space (Direct Use): https://huggingface.co/spaces/ShiroOnigami23/securehealthiot-disease-app

## Latest Training Snapshot

- Dataset: `itachi9604/disease-symptom-description-dataset`
- Samples: `4920`
- Classes: `41`
- Best model: `LogisticRegression`
- Holdout Accuracy: `1.00`
- Weighted F1: `1.00`

## Pipeline

1. Download dataset from Kaggle  
2. Train multiple models (`LogisticRegression`, `RandomForest`, `ExtraTrees`)  
3. Select best model by holdout/CV accuracy  
4. Save versioned artifacts in `artifacts/model_registry/...`  
5. Publish model to Hugging Face  
6. Publish app to Hugging Face Spaces

## Dataset

Default Kaggle dataset:

- `itachi9604/disease-symptom-description-dataset`

Main file used:

- `data/kaggle_raw/dataset.csv`

## Install

```bash
pip install -r requirements.txt
```

## Kaggle Auth Setup

Put your Kaggle API token JSON at:

- `%USERPROFILE%\.kaggle\kaggle.json` (Windows)

Helper:

```bash
python scripts/setup_kaggle_auth.py
```

## Train + Version

```bash
python -m disease_ml.train --note "baseline_v1"
```

Outputs:

- `artifacts/latest_model.joblib`
- `artifacts/model_registry/<timestamp>_<model>/model_bundle.joblib`
- `artifacts/model_registry/<timestamp>_<model>/metrics.json`

## Run Local App

```bash
python app.py
```

Open:

- `http://localhost:7860`

## Publish Model to Hugging Face

Set token:

```bash
set HF_TOKEN=YOUR_TOKEN
```

Upload:

```bash
python scripts/upload_model_to_hf.py --repo-id ShiroOnigami23/securehealthiot-disease-model
```

## Publish Hugging Face Space

```bash
python scripts/publish_space.py --space-id ShiroOnigami23/securehealthiot-disease-app --model-repo-id ShiroOnigami23/securehealthiot-disease-model
```

## Docker

```bash
docker build -t securehealthiot-disease .
docker run -p 7860:7860 securehealthiot-disease
```

## Notes

- The production disease prediction stack is in `disease_ml/`, `scripts/`, and root `app.py`.
