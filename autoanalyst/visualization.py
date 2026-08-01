"""
visualization.py — "Visualization" stage.

Chart generation tool, invoked by the planner for correlation heatmaps,
distributions, trends and model-diagnostic plots (cf. survey Sec. 4.1
Chart tasks — here produced *from* data rather than interpreted).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_chart(context, chart_type: str = "distribution", columns: list | None = None, **_):
    df = context.primary_dataframe()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    paths = []

    if chart_type == "correlation_heatmap":
        if len(numeric_cols) < 2:
            return {"skipped": "Not enough numeric columns for a correlation heatmap."}
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(min(10, 1 + 0.6 * len(numeric_cols)),
                                         min(8, 1 + 0.6 * len(numeric_cols))))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(numeric_cols)), numeric_cols, rotation=45, ha="right")
        ax.set_yticks(range(len(numeric_cols)), numeric_cols)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Correlation Heatmap")
        path = context.new_artifact_path("correlation_heatmap.png")
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        paths.append(path)

    elif chart_type == "distribution":
        cols = columns or numeric_cols[:6]
        if not cols:
            return {"skipped": "No numeric columns to plot distributions for."}
        n = len(cols)
        ncols = min(3, n)
        nrows = -(-n // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = np.array(axes).reshape(-1)
        for i, c in enumerate(cols):
            axes[i].hist(df[c].dropna(), bins=30, color="#4C72B0")
            axes[i].set_title(c, fontsize=10)
        for j in range(len(cols), len(axes)):
            axes[j].axis("off")
        fig.suptitle("Distributions")
        fig.tight_layout()
        path = context.new_artifact_path("distributions.png")
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        paths.append(path)

    elif chart_type == "trend":
        date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if not date_cols or not numeric_cols:
            return {"skipped": "No date-like or numeric columns found for a trend plot."}
        dcol = date_cols[0]
        try:
            ts = df[[dcol] + numeric_cols[:3]].copy()
            ts[dcol] = pd.to_datetime(ts[dcol], errors="coerce")
            ts = ts.dropna(subset=[dcol]).sort_values(dcol)
        except Exception as e:
            return {"skipped": f"Could not parse '{dcol}' as datetime: {e}"}
        fig, ax = plt.subplots(figsize=(9, 4))
        for c in numeric_cols[:3]:
            ax.plot(ts[dcol], ts[c], label=c)
        ax.legend()
        ax.set_title(f"Trend over {dcol}")
        fig.autofmt_xdate()
        path = context.new_artifact_path("trend.png")
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        paths.append(path)

    elif chart_type == "model_diagnostics":
        model_result = context.results.get("model")
        if not model_result or not model_result.get("feature_importances"):
            return {"skipped": "No trained model with feature importances found."}
        importances = model_result["feature_importances"]
        fig, ax = plt.subplots(figsize=(7, 4))
        names, vals = list(importances.keys()), list(importances.values())
        ax.barh(names[::-1], vals[::-1], color="#55A868")
        ax.set_title("Feature Importances")
        path = context.new_artifact_path("feature_importances.png")
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        paths.append(path)

    else:
        return {"skipped": f"Unknown chart_type '{chart_type}'."}

    context.record(f"chart::{chart_type}::{len(context.artifacts)}", {"paths": paths})
    return {"paths": paths}
