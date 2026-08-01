"""
executor.py — "Python execution" stage.

Executes model/tool-generated pandas/sklearn/matplotlib code against the
loaded DataFrame(s) inside a restricted namespace, capturing stdout and
any resulting value/plot. This is the NL2Code execution surface referenced
throughout Section 2.1 of the survey (e.g. Data Interpreter, PACHINCO).

NOTE: this is a *lightweight* sandbox (restricted builtins + whitelisted
modules) suitable for trusted, single-user local/dev use — not a hardened
security boundary for executing untrusted third-party code.
"""
from __future__ import annotations

import contextlib
import io
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ALLOWED_BUILTINS = {
    "len", "range", "min", "max", "sum", "sorted", "list", "dict", "set",
    "tuple", "enumerate", "zip", "abs", "round", "print", "float", "int",
    "str", "bool", "map", "filter", "isinstance", "type",
}


def run_python_snippet(context, code: str, **_):
    """Run `code` with `df` (and any other loaded frames) in scope.

    Returns dict(success, stdout, error, result, figure_path)
    """
    safe_builtins = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                     for k in _ALLOWED_BUILTINS}

    local_ns = {
        "pd": pd, "np": np, "plt": plt,
        "df": context.primary_dataframe(),
        "dfs": context.dataframes,
    }
    global_ns = {"__builtins__": safe_builtins}

    buf = io.StringIO()
    result = {"success": False, "stdout": "", "error": None, "result": None, "figure_path": None}
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, global_ns, local_ns)
        result["success"] = True
        result["result"] = local_ns.get("result")
        if plt.get_fignums():
            fig_path = context.new_artifact_path("snippet_plot.png")
            plt.savefig(fig_path, bbox_inches="tight", dpi=120)
            plt.close("all")
            result["figure_path"] = fig_path
    except Exception:
        result["error"] = traceback.format_exc(limit=3)
    finally:
        result["stdout"] = buf.getvalue()
    return result
