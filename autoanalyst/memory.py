"""
memory.py — "Memory" stage.

Persists conversation turns and pipeline run summaries per session so that
the Chat interface has continuity across turns (and across restarts, since
Claude has no memory between completions the same is true for most LLM
backends here). Kept intentionally simple (JSON on disk) — swap in a
vector store / DB for production use without changing the interface.
"""
from __future__ import annotations

import json
import os
import time
import uuid


class ConversationMemory:
    def __init__(self, session_id: str | None = None, store_dir: str = "sessions"):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)
        self.path = os.path.join(store_dir, f"{self.session_id}.json")
        self.turns: list[dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self.turns = json.load(f)

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.turns, f, indent=2, default=str)

    def add_turn(self, role: str, content: str, meta: dict | None = None):
        self.turns.append({
            "ts": time.time(), "role": role, "content": content, "meta": meta or {},
        })
        self._save()

    def history(self, last_n: int | None = None) -> list[dict]:
        return self.turns[-last_n:] if last_n else self.turns

    def context_string(self, last_n: int = 6) -> str:
        """Compact text summary of recent turns, useful as extra planner context."""
        recent = self.history(last_n)
        return "\n".join(f"[{t['role']}] {t['content']}" for t in recent)

    def clear(self):
        self.turns = []
        self._save()
