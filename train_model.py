"""
train_model.py
--------------
Loads the Adult Income dataset, preprocesses it, trains and compares
multiple classifiers, selects the best one, and saves the full pipeline
to models/model.pkl.

Target column : income  (binary: <=50K / >50K)
Problem type  : Binary Classification
"""

import os
import sys
import pathlib
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------
BASE_DIR = pathlib.Path(__file__).parent
DATA_PATH = BASE_DIR / "DATA" / "adult.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "model.pkl"

# ----------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------
print("=" * 60)
print("SALARY PREDICTION - ADULT INCOME DATASET")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[OK] Dataset loaded  : {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"     Columns          : {list(df.columns)}")

TARGET = "income"

# ----------------------------------------------------------
# 2. Basic cleaning
# ----------------------------------------------------------
str_cols = df.select_dtypes(include="object").columns
df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
df.replace("?", np.nan, inplace=True)

before = len(df)
df.dropna(inplace=True)
print(f"     Rows after clean : {len(df):,}  (removed {before - len(df):,} rows with '?')")

# Encode target: <=50K -> 0, >50K -> 1
le = LabelEncoder()
df[TARGET] = le.fit_transform(df[TARGET])
print(f"     Target classes   : {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ----------------------------------------------------------
# 3. Feature / target split
# ----------------------------------------------------------
DROP_COLS = ["fnlwgt", TARGET]
X = df.drop(columns=DROP_COLS)
y = df[TARGET]

NUMERIC_FEATURES = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
CATEGORICAL_FEATURES = X.select_dtypes(include="object").columns.tolist()

print(f"\n     Numeric features    : {NUMERIC_FEATURES}")
print(f"     Categorical features: {CATEGORICAL_FEATURES}")

# ----------------------------------------------------------
# 4. Train / Test split  (80 / 20, stratified)
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n     Train : {len(X_train):,}   Test : {len(X_test):,}")

# ----------------------------------------------------------
# 5. Preprocessor
# ----------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ]
)

# ----------------------------------------------------------
# 6. Model comparison  (5-fold stratified CV, ROC-AUC)
# ----------------------------------------------------------
CANDIDATES = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                      max_depth=4, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "=" * 60)
print("MODEL COMPARISON  (5-fold Stratified CV - ROC-AUC)")
print("=" * 60)

results = {}
for name, clf in CANDIDATES.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    results[name] = scores.mean()
    print(f"  {name:<25}  AUC = {scores.mean():.4f} +/- {scores.std():.4f}")

best_name = max(results, key=results.get)
print(f"\n[BEST] {best_name}  (AUC = {results[best_name]:.4f})")

# ----------------------------------------------------------
# 7. Retrain best model on full training set
# ----------------------------------------------------------
best_pipeline = Pipeline(
    [("preprocessor", preprocessor), ("classifier", CANDIDATES[best_name])]
)
best_pipeline.fit(X_train, y_train)

# ----------------------------------------------------------
# 8. Evaluate on held-out test set
# ----------------------------------------------------------
y_pred = best_pipeline.predict(X_test)
y_prob = best_pipeline.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("TEST SET EVALUATION")
print("=" * 60)
print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))

# ----------------------------------------------------------
# 9. Save pipeline + metadata
# ----------------------------------------------------------
artifact = {
    "pipeline":            best_pipeline,
    "model_name":          best_name,
    "feature_columns":     X.columns.tolist(),
    "numeric_features":    NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "target_classes":      le.classes_.tolist(),
    "label_encoder":       le,
    "category_options":    {
        col: sorted(df[col].dropna().unique().tolist())
        for col in CATEGORICAL_FEATURES
    },
}

joblib.dump(artifact, MODEL_PATH)
print(f"\n[SAVED] Pipeline saved -> {MODEL_PATH}")
print("=" * 60)
