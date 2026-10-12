# Agent Instructions — Salary Prediction App

## Overview
This project is a **binary classification** machine learning application
that predicts adult income (≤ $50K vs > $50K) from census features.

## Dataset
- **File:** `data/adult.csv`
- **Source:** UCI Adult Income Dataset
- **Rows:** ~48,842  |  **Columns:** 15
- **Target column:** `income`  (string: `<=50K` or `>50K`)
- **Missing values:** represented as `?` — rows are dropped during preprocessing

## Training (`train_model.py`)
1. Load CSV with `pandas.read_csv`.
2. Strip whitespace, replace `?` with `NaN`, drop missing rows.
3. Label-encode the target (`<=50K` → 0, `>50K` → 1).
4. Drop `fnlwgt` (census weight — not a useful predictor).
5. Build a `ColumnTransformer`: `StandardScaler` for numeric cols, `OneHotEncoder` for categoricals.
6. Compare four classifiers via 5-fold stratified CV (ROC-AUC).
7. Retrain the winner on the full training split.
8. Save a `dict` artifact via `joblib` containing the pipeline and metadata.

## Inference (`app.py`)
- Loads the `joblib` artifact from `models/model.pkl`.
- Renders a Streamlit form with all input features (numeric + categorical).
- Calls `pipeline.predict` and `pipeline.predict_proba` on a single-row DataFrame.
- Displays the predicted class label and confidence probability.

## Extending the Project
- To add a new model, insert it into the `CANDIDATES` dict in `train_model.py`.
- To change the metric, update the `scoring` parameter in `cross_val_score`.
- To add SHAP explainability, `pip install shap` and call `shap.TreeExplainer`
  after loading the pipeline in `app.py`.

## Deployment Notes
- All paths use `pathlib.Path(__file__).parent` — no absolute paths.
- No API keys or secrets are required.
- Compatible with Streamlit Community Cloud and Render.
