"""
AutoAnalyst
===========
An LLM/Agent-as-Data-Analyst pipeline implementing:

  Multiple files -> Agent planner -> Tool selection -> Python execution
  -> EDA -> ML -> Visualization -> Report generation -> Chat interface
  -> Memory -> Deployment

Design is grounded in the four design goals distilled in
"LLM/Agent-as-Data-Analyst: A Survey" (Tang et al., 2025):
  O1: Cross-modal, open-world data support   -> data_loader.py
  O2: NL-based, tool-agnostic interaction    -> planner.py + tools.py
  O3: Semantic operators over literal ones   -> eda.py / ml.py / viz.py
  O4: Autonomous pipeline orchestration      -> orchestrator.py + memory.py
"""
from .orchestrator import AutoAnalystAgent

__all__ = ["AutoAnalystAgent"]
__version__ = "0.1.0"
