# DDXPlus LightGBM (Full Dataset)

Train a LightGBM model on the DDXPlus dataset and serve predictions with a small Flask UI.

This repo includes:
- Training script: `second.py`
- Backend API: `app.py`
- Frontend UI: `index.html`
- Evidence/condition maps: `release_evidences.json`, `release_conditions.json`

## Requirements

- Python 3.10+ (3.12 works)
- CPU (no GPU required)
- Disk space for dataset + model artifacts

Python packages:
- lightgbm
- pandas
- scikit-learn
- flask
- joblib

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install lightgbm pandas scikit-learn flask joblib
```

## Dataset Layout

The training script expects:

- `./DataSet/release_train_patients`
- `./DataSet/release_validate_patients`
- `./DataSet/release_test_patients`

These can be CSV or Parquet files. Directories are supported (all `*.csv` and `*.parquet` are loaded and concatenated).

## Train the Full Model

Run:

```bash
python second.py
```

Outputs:
- `lgbm_ddxplus_full.joblib` (model)
- `lgbm_ddxplus_full_meta.joblib` (metadata: evidence vocab + categorical values)

Training uses:
- Full dataset (no row caps)
- Evidence vocabulary built from all training data
- CPU only (`n_jobs=-1`)

Key training settings are at the top of `second.py`:
- `TOP_EVIDENCES = None` to use all evidence codes
- `MAX_*_ROWS = None` to load full data

## Run the API + UI

Start the Flask server:

```bash
python app.py
```

Open:
- `http://127.0.0.1:5000/`

Features:
- Searchable evidence list
- Initial evidence datalist
- Top-5 predicted conditions with probabilities

## How Prediction Works

1. User selects evidences in the UI
2. `app.py` maps selections to feature columns
3. Evidence codes are filtered to the training vocabulary
4. Model predicts top conditions
5. Condition names are mapped using `release_conditions.json`

## Evidence and Condition Names

Mappings:
- `release_evidences.json` provides evidence code -> question text
- `release_conditions.json` provides condition name -> English label

`app.py` exposes `/evidences` to power the evidence picker with readable labels.

## File Overview

- `second.py`: training script
- `app.py`: Flask API server
- `index.html`: UI
- `release_evidences.json`: evidence map
- `release_conditions.json`: condition map
- `lgbm_ddxplus_full.joblib`: trained model (generated)
- `lgbm_ddxplus_full_meta.joblib`: metadata (generated)

## Notes

- Large runs can take time on CPU. Early stopping is enabled on validation data.
- If you want faster experiments, reintroduce row caps or limit evidence vocabulary.

## Troubleshooting

- If the UI is blank, ensure `app.py` is running and files are in the repo root.
- If you see LightGBM “no further splits with positive gain” messages, it is expected for some branches and not a crash.
- If you update `second.py`, re-run training to regenerate the model and metadata.
