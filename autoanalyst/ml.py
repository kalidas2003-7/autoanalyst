"""
ml.py — "ML" stage.

Baseline AutoML: auto-detects classification vs. regression, builds a
preprocessing + model pipeline, trains, evaluates, and returns feature
importances — corresponding to the survey's NL2Code / analysis-agent
pathway (Sec 2.1) applied end-to-end rather than via manual scripting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                              mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _infer_task(y: pd.Series) -> str:
    if y.dtype == object or str(y.dtype).startswith("category"):
        return "classification"
    if y.nunique() <= max(10, int(0.05 * len(y))) and np.all(y.dropna() == y.dropna().astype(int)):
        return "classification"
    return "regression"


def _expand_datetime_columns(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Convert obviously date-like columns into numeric year/month/day/dow
    features instead of leaving them as high-cardinality categoricals."""
    X = X.copy()
    dropped = []
    for c in list(X.columns):
        is_datetimeish = "date" in c.lower() or "time" in c.lower()
        if not is_datetimeish and str(X[c].dtype) != "datetime64[ns]":
            continue
        parsed = pd.to_datetime(X[c], errors="coerce")
        if parsed.notna().mean() < 0.8:
            continue
        X[f"{c}_year"] = parsed.dt.year
        X[f"{c}_month"] = parsed.dt.month
        X[f"{c}_day"] = parsed.dt.day
        X[f"{c}_dow"] = parsed.dt.dayofweek
        X = X.drop(columns=[c])
        dropped.append(c)
    return X, dropped


def train_model(context, target: str | None = None, test_size: float = 0.2,
                 random_state: int = 42, **_):
    df = context.primary_dataframe().dropna(subset=[target] if target else None)

    if not target or target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data. "
                          f"Available columns: {list(df.columns)}")

    y = df[target]
    X = df.drop(columns=[target])
    X, expanded_date_cols = _expand_datetime_columns(X)
    task = _infer_task(y)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    # guard against blowing up one-hot dimensionality on free-text/high-cardinality columns
    categorical_features = [c for c in categorical_features if X[c].nunique() <= 50]
    X = X[numeric_features + categorical_features]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())]), numeric_features),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
    ])

    if task == "classification":
        y_enc, classes = pd.factorize(y)
        model = RandomForestClassifier(n_estimators=200, random_state=random_state)
    else:
        y_enc, classes = y.values, None
        model = RandomForestRegressor(n_estimators=200, random_state=random_state)

    pipe = Pipeline([("prep", preprocessor), ("model", model)])

    stratify = y_enc if task == "classification" and len(set(y_enc)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=stratify)

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    metrics = {}
    if task == "classification":
        metrics["accuracy"] = round(float(accuracy_score(y_test, preds)), 4)
        metrics["f1_weighted"] = round(float(f1_score(y_test, preds, average="weighted")), 4)
    else:
        metrics["mae"] = round(float(mean_absolute_error(y_test, preds)), 4)
        metrics["rmse"] = round(float(mean_squared_error(y_test, preds) ** 0.5), 4)
        metrics["r2"] = round(float(r2_score(y_test, preds)), 4)

    # feature importances (map back through OHE)
    importances = {}
    try:
        ohe_names = []
        if categorical_features:
            ohe = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
            ohe_names = list(ohe.get_feature_names_out(categorical_features))
        feat_names = numeric_features + ohe_names
        raw_imp = pipe.named_steps["model"].feature_importances_
        importances = dict(sorted(zip(feat_names, raw_imp.tolist()),
                                   key=lambda kv: -kv[1])[:15])
    except Exception:
        pass

    result = {
        "task": task,
        "target": target,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "metrics": metrics,
        "feature_importances": importances,
        "classes": list(map(str, classes)) if classes is not None else None,
    }
    context.record("model", result)
    context.set_model(pipe, task=task, target=target)
    return result
