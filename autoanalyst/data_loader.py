"""
data_loader.py — "Multiple files" stage.

Ingests structured (csv/xlsx/parquet), semi-structured (json) and
raw-text data, normalizes each into a pandas DataFrame (or raw text
blob for unstructured docs), and reports basic modality metadata —
mirroring the survey's structured / semi-structured / unstructured
data taxonomy (Section 1.3).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class LoadedFile:
    name: str
    path: str
    modality: str  # "structured" | "semi-structured" | "unstructured"
    kind: str       # csv, xlsx, json, parquet, txt, ...
    data: Any        # DataFrame for tabular, str for text
    shape: tuple | None = None
    columns: list = field(default_factory=list)


class DataLoader:
    """Loads an arbitrary set of files into a uniform in-memory registry."""

    SUPPORTED_TABULAR = {".csv", ".tsv", ".xlsx", ".xls", ".parquet"}
    SUPPORTED_SEMI = {".json", ".jsonl"}
    SUPPORTED_TEXT = {".txt", ".md"}

    def __init__(self):
        self.files: dict[str, LoadedFile] = {}

    def load(self, path: str) -> LoadedFile:
        ext = os.path.splitext(path)[1].lower()
        name = os.path.basename(path)

        if ext in self.SUPPORTED_TABULAR:
            df = self._load_tabular(path, ext)
            lf = LoadedFile(
                name=name, path=path, modality="structured", kind=ext.strip("."),
                data=df, shape=df.shape, columns=list(df.columns),
            )
        elif ext in self.SUPPORTED_SEMI:
            df, raw = self._load_semi_structured(path, ext)
            lf = LoadedFile(
                name=name, path=path, modality="semi-structured", kind=ext.strip("."),
                data=df if df is not None else raw,
                shape=df.shape if df is not None else None,
                columns=list(df.columns) if df is not None else [],
            )
        elif ext in self.SUPPORTED_TEXT:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            lf = LoadedFile(name=name, path=path, modality="unstructured",
                             kind=ext.strip("."), data=text)
        else:
            raise ValueError(f"Unsupported file type: {ext}. "
                              f"Supported: {self.SUPPORTED_TABULAR | self.SUPPORTED_SEMI | self.SUPPORTED_TEXT}")

        self.files[name] = lf
        return lf

    def load_many(self, paths: list[str]) -> dict[str, LoadedFile]:
        for p in paths:
            self.load(p)
        return self.files

    @staticmethod
    def _load_tabular(path: str, ext: str) -> pd.DataFrame:
        if ext == ".csv":
            return pd.read_csv(path)
        if ext == ".tsv":
            return pd.read_csv(path, sep="\t")
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(path)
        if ext == ".parquet":
            return pd.read_parquet(path)
        raise ValueError(ext)

    @staticmethod
    def _load_semi_structured(path: str, ext: str):
        """Try to flatten JSON/JSONL into a relational DataFrame (markup
        extraction, cf. survey Sec. 3.1); fall back to raw dict/text."""
        try:
            if ext == ".jsonl":
                records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
            else:
                with open(path, encoding="utf-8") as f:
                    obj = json.load(f)
                records = obj if isinstance(obj, list) else [obj]
            df = pd.json_normalize(records)
            return df, None
        except Exception:
            with open(path, encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            return None, raw

    def summary(self) -> dict:
        """Manifest describing everything currently loaded (fed to the planner)."""
        out = {}
        for name, lf in self.files.items():
            entry = {"modality": lf.modality, "kind": lf.kind}
            if isinstance(lf.data, pd.DataFrame):
                entry["shape"] = lf.shape
                entry["columns"] = lf.columns
                entry["dtypes"] = {c: str(t) for c, t in lf.data.dtypes.items()}
            else:
                entry["preview"] = str(lf.data)[:300]
            out[name] = entry
        return out

    def get_dataframe(self, name: str | None = None) -> pd.DataFrame:
        """Convenience accessor: returns the named table, or the sole/primary one."""
        if name:
            lf = self.files[name]
        else:
            tabular = [f for f in self.files.values() if isinstance(f.data, pd.DataFrame)]
            if not tabular:
                raise ValueError("No tabular data loaded.")
            lf = tabular[0]
        if not isinstance(lf.data, pd.DataFrame):
            raise ValueError(f"{lf.name} is not tabular data.")
        return lf.data
