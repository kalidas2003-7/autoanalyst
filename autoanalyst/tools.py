"""
tools.py — "Tool selection" stage.

A minimal, extensible tool registry. Each tool is a plain Python callable
with the signature `fn(context, **args) -> dict`, where `context` is the
shared PipelineContext (holds dataframes, artifacts, memory). This keeps
tool integration decoupled/tool-agnostic (cf. survey O2/L2: "Rigid Tool
Coupling" -> "Flexible Tools").
"""
from __future__ import annotations

from typing import Callable


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Callable] = {}
        self._descriptions: dict[str, str] = {}

    def register(self, name: str, fn: Callable, description: str = ""):
        self._tools[name] = fn
        self._descriptions[name] = description
        return fn

    def get(self, name: str) -> Callable:
        if name not in self._tools:
            raise KeyError(f"Unknown tool '{name}'. Available: {self.names()}")
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def describe(self) -> dict:
        return dict(self._descriptions)


def build_default_registry() -> ToolRegistry:
    """Wires up the standard AutoAnalyst toolset: EDA, cleaning, ML,
    visualization, code execution and reporting."""
    from . import eda, ml, visualization, report
    from .executor import run_python_snippet

    registry = ToolRegistry()
    registry.register("profile_data", eda.profile_data,
                       "Compute shape, dtypes, missingness, and summary stats.")
    registry.register("clean_data", eda.clean_data,
                       "Handle missing values, duplicates, and outliers.")
    registry.register("plot_chart", visualization.plot_chart,
                       "Generate a chart (correlation heatmap, distribution, trend, etc.).")
    registry.register("train_model", ml.train_model,
                       "Train + evaluate a baseline ML model (auto classification/regression).")
    registry.register("run_code", run_python_snippet,
                       "Execute an arbitrary pandas/sklearn Python snippet against the data.")
    registry.register("generate_report", report.generate_report,
                       "Assemble a Markdown report from all prior step outputs.")
    return registry
