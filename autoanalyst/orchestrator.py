"""
orchestrator.py — glues every stage of the pipeline together:

  Multiple files -> Agent planner -> Tool selection -> Python execution
  -> EDA -> ML -> Visualization -> Report generation -> Chat interface
  -> Memory -> Deployment

`AutoAnalystAgent` is the single entry point used by both the CLI
(main.py) and the chat UI (chat_app.py).
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field

import pandas as pd

from .data_loader import DataLoader
from .memory import ConversationMemory
from .planner import get_planner
from .tools import build_default_registry


class PipelineContext:
    """Shared mutable state passed to every tool during a run."""

    def __init__(self, dataframes: dict[str, pd.DataFrame], goal: str, artifact_dir: str):
        self.dataframes = dataframes
        self._primary_name = next(iter(dataframes)) if dataframes else None
        self.goal = goal
        self.artifact_dir = artifact_dir
        os.makedirs(artifact_dir, exist_ok=True)
        self.artifacts: list[str] = []
        self.results: dict = {}
        self.model = None
        self.model_meta = {}

    def primary_name(self) -> str:
        return self._primary_name

    def primary_dataframe(self) -> pd.DataFrame:
        if self._primary_name is None:
            raise ValueError("No tabular data available.")
        return self.dataframes[self._primary_name]

    def set_dataframe(self, name: str, df: pd.DataFrame):
        self.dataframes[name] = df

    def set_model(self, model, task: str, target: str):
        self.model = model
        self.model_meta = {"task": task, "target": target}

    def new_artifact_path(self, filename: str) -> str:
        base, ext = os.path.splitext(filename)
        path = os.path.join(self.artifact_dir, filename)
        i = 1
        while path in self.artifacts:
            path = os.path.join(self.artifact_dir, f"{base}_{i}{ext}")
            i += 1
        self.artifacts.append(path)
        return path

    def record(self, key: str, value):
        self.results[key] = value


@dataclass
class RunResult:
    plan: dict
    step_results: list = field(default_factory=list)
    report_markdown: str | None = None
    report_path: str | None = None
    errors: list = field(default_factory=list)


class AutoAnalystAgent:
    """High-level facade: load files, chat, and it runs the full pipeline."""

    def __init__(self, session_id: str | None = None, workdir: str = "."):
        self.workdir = workdir
        self.loader = DataLoader()
        self.registry = build_default_registry()
        self.planner = get_planner()
        self.memory = ConversationMemory(
            session_id=session_id, store_dir=os.path.join(workdir, "sessions"))
        self.session_id = self.memory.session_id
        self.artifact_root = os.path.join(workdir, "outputs", self.session_id)

    # ---- Stage 1: Multiple files -------------------------------------
    def load_files(self, paths: list[str]) -> dict:
        self.loader.load_many(paths)
        manifest = self.loader.summary()
        self.memory.add_turn("system", f"Loaded files: {list(self.loader.files.keys())}",
                              {"manifest": manifest})
        return manifest

    # ---- Stages 2-8: planner -> tools -> exec -> EDA/ML/viz -> report -
    def chat(self, user_message: str) -> RunResult:
        self.memory.add_turn("user", user_message)

        manifest = self.loader.summary()
        plan = self.planner.plan(user_message, manifest, self.registry)

        dataframes = {
            name: lf.data for name, lf in self.loader.files.items()
            if isinstance(lf.data, pd.DataFrame)
        }
        run_id = uuid.uuid4().hex[:6]
        context = PipelineContext(dataframes, plan.goal,
                                   os.path.join(self.artifact_root, run_id))

        step_results, errors = [], []
        for step in plan.steps:
            try:
                tool_fn = self.registry.get(step.tool)
                out = tool_fn(context, **step.args)
                step_results.append({"step": step.__dict__, "output": _brief(out)})
            except Exception as e:
                errors.append({"step": step.__dict__, "error": str(e)})
                step_results.append({"step": step.__dict__, "output": {"error": str(e)}})

        report = context.results.get("report", {})
        result = RunResult(
            plan=plan.to_dict(),
            step_results=step_results,
            report_markdown=report.get("markdown"),
            report_path=report.get("path"),
            errors=errors,
        )

        reply = self._compose_reply(result)
        self.memory.add_turn("assistant", reply, {"run": {
            "plan": result.plan, "report_path": result.report_path,
        }})
        return result

    @staticmethod
    def _compose_reply(result: RunResult) -> str:
        if result.report_markdown:
            return result.report_markdown
        lines = ["Ran the following plan but no report was generated:"]
        for sr in result.step_results:
            lines.append(f"- {sr['step']['tool']}: {sr['output']}")
        return "\n".join(lines)

    # ---- Stage: Python execution tool (ad-hoc) -------------------------
    def run_code(self, code: str, artifact_dir: str | None = None) -> dict:
        dataframes = {
            name: lf.data for name, lf in self.loader.files.items()
            if isinstance(lf.data, pd.DataFrame)
        }
        context = PipelineContext(dataframes, "ad-hoc code execution",
                                   artifact_dir or os.path.join(self.artifact_root, "adhoc"))
        return self.registry.get("run_code")(context, code=code)


def _brief(out):
    """Trim large outputs (e.g. full markdown, full DataFrames) before storing in the run log."""
    if isinstance(out, dict):
        return {k: (v if not isinstance(v, str) or len(v) < 500 else v[:500] + "...")
                for k, v in out.items()}
    return out
