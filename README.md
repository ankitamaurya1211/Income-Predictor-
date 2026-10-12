# Salary Prediction App

A Machine Learning web application that predicts whether an individual's
annual income is **≤ $50K** or **> $50K**, built on the
[Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
(UCI Machine Learning Repository).

## Tech Stack

| Layer | Library |
|-------|---------|
| UI | Streamlit |
| ML | scikit-learn |
| Data | pandas, numpy |
| Persistence | joblib |

## Project Structure

```
project/
├── app.py                # Streamlit web application
├── train_model.py        # Data preprocessing, model training & evaluation
├── requirements.txt      # Python dependencies
├── README.md
├── agent_instructions.md
├── .env.example
├── data/
│   └── adult.csv         # Adult Income Dataset
└── models/
    └── model.pkl         # Trained ML pipeline (auto-generated)
```

## Quick Start — Local

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train_model.py
```

This will:
- Load and preprocess `data/adult.csv`
- Compare Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting
- Select the best model by cross-validated ROC-AUC
- Save the trained pipeline to `models/model.pkl`

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Deployment — Streamlit Community Cloud

1. Push the repository to GitHub (make sure `models/model.pkl` is committed,
   **or** add a post-deploy step to run `train_model.py`).
2. Visit [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Connect your GitHub repo, set the main file to `app.py`.
4. Click **Deploy** — Streamlit Cloud will install `requirements.txt` automatically.

> **Tip:** If you prefer not to commit the binary model, add a `packages.txt`
> file (if needed) and call `python train_model.py` before the app starts
> by placing it in a `setup.sh` or using Streamlit's
> [`st.cache_resource`](https://docs.streamlit.io/library/api-reference/performance/st.cache_resource)
> pattern with lazy training on first load.

---

## Model Details

| Item | Value |
|------|-------|
| Dataset | Adult Income (UCI) — 48,842 rows, 15 columns |
| Target | `income` — binary (`<=50K` / `>50K`) |
| Metric | ROC-AUC (5-fold stratified CV) |
| Models compared | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting |
| Preprocessing | StandardScaler (numeric) · OneHotEncoder (categorical) |

The best model is selected automatically and saved as a full `sklearn.pipeline.Pipeline`.

## Environment Variables

No API keys are required. See `.env.example` for reference.
