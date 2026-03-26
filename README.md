# SecureHealthIoT Disease Prediction

Production-style symptom-based disease prediction stack with:

- leakage-aware grouped splits
- external validation split
- probability calibration (`CalibratedClassifierCV`)
- model versioning + reproducible Kaggle GPU training
- Hugging Face model + dataset + Space deployment
- user-friendly app with symptom mode + report/image-assisted mode

## Live Links

- Model: https://huggingface.co/ShiroOnigami23/securehealthiot-disease-model
- Dataset: https://huggingface.co/datasets/ShiroOnigami23/securehealthiot-disease-dataset
- App Space: https://huggingface.co/spaces/ShiroOnigami23/securehealthiot-disease-app

## Core Reliability Features

1. Grouped splitting by symptom signature to reduce leakage across splits.
2. Separate external validation set.
3. Calibration using sigmoid scaling for more reliable probabilities.
4. Leakage report in metrics (`overlap_signature_*` checks).
5. Versioned artifacts in `artifacts/model_registry`.

## App Features

1. Quick Symptom Predictor:
   - multi-select symptoms
   - top-3 predictions
   - confidence band + precaution summary
2. Report/Image Assisted Analysis:
   - upload `.txt`, `.csv`, or `.pdf` report
   - optional image upload with quality diagnostics
   - symptom extraction from report text + combined prediction
3. Medical safety disclaimer and decision-support framing.

## Install

```bash
pip install -r requirements.txt
```

## Local Training

```bash
python -m disease_ml.train --note "reliability_pass_v2"
```

Outputs:

- `artifacts/latest_model.joblib`
- `artifacts/model_registry/<timestamp>_<model>/model_bundle.joblib`
- `artifacts/model_registry/<timestamp>_<model>/metrics.json`

## Kaggle GPU Training

Push + run:

```bash
kaggle kernels push -p kaggle_kernel
python scripts/wait_kaggle_kernel.py
kaggle kernels output aryansingh21fd/securehealthiot-disease-trainer-v1 -p kaggle_pull
```

Sync outputs to HF:

```bash
set HF_TOKEN=YOUR_TOKEN
python scripts/upload_kaggle_outputs_to_hf.py --outputs-dir kaggle_pull
```

## Run App Locally

```bash
python app.py
```

Open:

- `http://localhost:7860`

## Publish/Refresh Hugging Face Space

```bash
set HF_TOKEN=YOUR_TOKEN
python scripts/publish_space.py --space-id ShiroOnigami23/securehealthiot-disease-app --model-repo-id ShiroOnigami23/securehealthiot-disease-model
```

## Medical Notice

This software is for educational and decision-support usage. It is not a medical device and not a replacement for licensed clinical judgment.
