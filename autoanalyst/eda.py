"""
eda.py — "EDA" stage.

Implements the automated profiling / cleaning operators corresponding to
the survey's "Basic Literal Analysis" -> "Semantic Supported Analysis"
evolution (Sec 1.1 L3, O3): beyond raw filter/aggregate, we also surface
data-quality flags akin to the semi-structured-table pathologies in
Fig. 5 (wrong index, inconsistent content, etc.) generalized to tabular
data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def profile_data(context, **_):
    df = context.primary_dataframe()
    n_rows, n_cols = df.shape

    dtypes = {c: str(t) for c, t in df.dtypes.items()}
    missing = df.isna().sum()
    missing_pct = (missing / max(n_rows, 1) * 100).round(2)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    desc_numeric = df[numeric_cols].describe().round(3).to_dict() if numeric_cols else {}
    cardinality = {c: int(df[c].nunique()) for c in categorical_cols}
    duplicates = int(df.duplicated().sum())

    outliers = {}
    for c in numeric_cols:
        s = df[c].dropna()
        if len(s) < 5:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        if n_out:
            outliers[c] = n_out

    profile = {
        "shape": {"rows": n_rows, "cols": n_cols},
        "dtypes": dtypes,
        "missing_count": missing.to_dict(),
        "missing_pct": missing_pct.to_dict(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "numeric_summary": desc_numeric,
        "categorical_cardinality": cardinality,
        "duplicate_rows": duplicates,
        "outlier_counts": outliers,
    }
    context.record("profile", profile)
    return profile


def clean_data(context, strategy: str = "auto", **_):
    """Semantic-aware cleaning: impute, drop dup rows, cap outliers."""
    df = context.primary_dataframe().copy()
    report = {"actions": []}

    dup_before = df.duplicated().sum()
    if dup_before:
        df = df.drop_duplicates()
        report["actions"].append(f"Dropped {dup_before} duplicate rows.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    for c in numeric_cols:
        n_missing = df[c].isna().sum()
        if n_missing:
            median = df[c].median()
            df[c] = df[c].fillna(median)
            report["actions"].append(f"Filled {n_missing} missing values in '{c}' with median ({median}).")

    for c in cat_cols:
        n_missing = df[c].isna().sum()
        if n_missing:
            mode = df[c].mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else "Unknown"
            df[c] = df[c].fillna(fill)
            report["actions"].append(f"Filled {n_missing} missing values in '{c}' with mode ('{fill}').")

    for c in numeric_cols:
        s = df[c]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_capped = int(((s < lo) | (s > hi)).sum())
        if n_capped:
            df[c] = s.clip(lo, hi)
            report["actions"].append(f"Capped {n_capped} outliers in '{c}' to [{lo:.2f}, {hi:.2f}].")

    context.set_dataframe(context.primary_name(), df)
    report["final_shape"] = df.shape
    context.record("cleaning", report)
    return report
