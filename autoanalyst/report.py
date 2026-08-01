"""
report.py — "Report generation" stage.

Assembles all accumulated step outputs (profile, cleaning, model,
charts) into a single Markdown report — the "Down Stream Tasks"
output surface referenced in Fig. 3 of the survey (formatting/
standardization/visualization post-processing).
"""
from __future__ import annotations

import datetime as dt
import os


def generate_report(context, **_):
    lines = []
    lines.append(f"# AutoAnalyst Report\n")
    lines.append(f"_Generated {dt.datetime.now().isoformat(timespec='seconds')}_\n")
    lines.append(f"**Goal:** {context.goal}\n")

    profile = context.results.get("profile")
    if profile:
        lines.append("## Data Profile")
        lines.append(f"- Rows x Cols: {profile['shape']['rows']} x {profile['shape']['cols']}")
        lines.append(f"- Duplicate rows: {profile['duplicate_rows']}")
        miss = {k: v for k, v in profile["missing_pct"].items() if v > 0}
        if miss:
            lines.append("- Columns with missing data: " +
                          ", ".join(f"{k} ({v}%)" for k, v in miss.items()))
        else:
            lines.append("- No missing data detected.")
        if profile["outlier_counts"]:
            lines.append("- Outlier counts (IQR method): " +
                          ", ".join(f"{k}: {v}" for k, v in profile["outlier_counts"].items()))
        lines.append("")

    cleaning = context.results.get("cleaning")
    if cleaning:
        lines.append("## Cleaning Actions")
        for a in cleaning["actions"]:
            lines.append(f"- {a}")
        if not cleaning["actions"]:
            lines.append("- No cleaning necessary.")
        lines.append("")

    model = context.results.get("model")
    if model:
        lines.append("## Model")
        lines.append(f"- Task: {model['task']}")
        lines.append(f"- Target: `{model['target']}`")
        lines.append(f"- Train / Test size: {model['n_train']} / {model['n_test']}")
        lines.append("- Metrics: " + ", ".join(f"{k}={v}" for k, v in model["metrics"].items()))
        if model["feature_importances"]:
            top = list(model["feature_importances"].items())[:5]
            lines.append("- Top features: " + ", ".join(f"{k} ({v:.3f})" for k, v in top))
        lines.append("")

    charts = [k for k in context.results if k.startswith("chart::")]
    if charts:
        lines.append("## Visualizations")
        for k in charts:
            for p in context.results[k].get("paths", []):
                lines.append(f"![{os.path.basename(p)}]({p})")
        lines.append("")

    md = "\n".join(lines)
    report_path = context.new_artifact_path("report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    context.record("report", {"path": report_path, "markdown": md})
    return {"path": report_path, "markdown": md}
