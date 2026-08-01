"""
planner.py — "Agent planner" + "Tool selection" stages.

Implements a lightweight version of the task/action-graph decomposition
described for LLM-based analysis agents (survey Sec. 2.1, "Data
Interpreter"): a natural-language request is decomposed into an ordered
sequence of Steps, each bound to a Tool from the ToolRegistry.

Two planning backends:
  * LLMPlanner   — calls Claude (Anthropic API) to produce the plan as JSON.
                   Used automatically if ANTHROPIC_API_KEY is set.
  * RulePlanner  — deterministic keyword/heuristic fallback, so the whole
                   pipeline still works fully offline / without an API key.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from .tools import ToolRegistry


@dataclass
class Step:
    id: int
    tool: str
    description: str
    args: dict = field(default_factory=dict)


@dataclass
class Plan:
    goal: str
    steps: list[Step]

    def to_dict(self):
        return {"goal": self.goal, "steps": [s.__dict__ for s in self.steps]}


SYSTEM_PROMPT = """You are the planning module of AutoAnalyst, an agentic data
analysis system. Given a user request and a manifest describing the
currently loaded dataset(s), output ONLY a JSON object of the form:

{
  "goal": "<one sentence restatement of the user's analytical goal>",
  "steps": [
    {"tool": "<tool_name>", "description": "<what this step does>", "args": {...}}
  ]
}

Available tools: {tool_names}

Rules:
- Always start with "profile_data" unless the manifest shows profiling is unnecessary.
- Only include "train_model" if the user's request implies prediction/classification/
  regression/clustering, or explicitly asks for modeling.
- Only include "plot_chart" steps for visuals that are actually useful for the request.
- Always end with "generate_report".
- Keep the plan to at most 7 steps.
- Respond with raw JSON only, no markdown fences, no commentary.
"""


class RulePlanner:
    """Deterministic fallback planner (no API key required)."""

    def plan(self, query: str, manifest: dict, registry: ToolRegistry) -> Plan:
        q = query.lower()
        steps: list[Step] = []
        sid = 0

        def add(tool, desc, **args):
            nonlocal sid
            sid += 1
            steps.append(Step(id=sid, tool=tool, description=desc, args=args))

        add("profile_data", "Profile the dataset(s): shape, dtypes, missingness, stats.")

        if any(k in q for k in ["clean", "missing", "duplicate", "outlier", "fix"]):
            add("clean_data", "Clean data: handle missing values, duplicates, outliers.")

        if any(k in q for k in ["correlat", "relationship", "distribution", "explore",
                                 "summary", "understand", "insight"]):
            add("plot_chart", "Plot correlation heatmap and key distributions.",
                chart_type="correlation_heatmap")
            add("plot_chart", "Plot distribution of numeric columns.",
                chart_type="distribution")

        ml_keywords = ["predict", "classif", "regress", "cluster", "model", "forecast",
                        "train", "accuracy", "target"]
        if any(k in q for k in ml_keywords):
            target = self._guess_target(query, manifest)
            add("train_model", "Train and evaluate a baseline ML model.", target=target)
            add("plot_chart", "Plot feature importances / model diagnostics.",
                chart_type="model_diagnostics")

        if any(k in q for k in ["trend", "over time", "time series", "date"]):
            add("plot_chart", "Plot trend over time for date-like columns.",
                chart_type="trend")

        add("generate_report", "Assemble a Markdown report of all findings.")

        return Plan(goal=query.strip() or "Analyze the uploaded data.", steps=steps)

    @staticmethod
    def _guess_target(query: str, manifest: dict) -> str | None:
        cols = []
        for meta in manifest.values():
            cols.extend(meta.get("columns", []))
        for c in cols:
            if c.lower() in query.lower():
                return c
        # heuristics: common target-ish names
        for c in cols:
            if re.search(r"(target|label|class|churn|price|sales|outcome|y)$", c.lower()):
                return c
        return cols[-1] if cols else None


class LLMPlanner:
    """Calls Claude to produce the plan. Falls back to RulePlanner on any error."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self._fallback = RulePlanner()

    def plan(self, query: str, manifest: dict, registry: ToolRegistry) -> Plan:
        try:
            import anthropic
        except ImportError:
            return self._fallback.plan(query, manifest, registry)

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return self._fallback.plan(query, manifest, registry)

        try:
            client = anthropic.Anthropic(api_key=api_key)
            sys_prompt = SYSTEM_PROMPT.format(tool_names=", ".join(registry.names()))
            user_content = (
                f"User request: {query}\n\n"
                f"Dataset manifest:\n{json.dumps(manifest, indent=2, default=str)}"
            )
            resp = client.messages.create(
                model=self.model,
                max_tokens=1000,
                system=sys_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
            obj = json.loads(text)
            steps = [
                Step(id=i + 1, tool=s["tool"], description=s.get("description", ""),
                     args=s.get("args", {}))
                for i, s in enumerate(obj["steps"])
                if s["tool"] in registry.names()
            ]
            if not steps:
                raise ValueError("LLM produced no valid steps")
            return Plan(goal=obj.get("goal", query), steps=steps)
        except Exception:
            return self._fallback.plan(query, manifest, registry)


def get_planner():
    """Auto-select LLM planner if a key is available, else rule-based."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return LLMPlanner()
    return RulePlanner()
