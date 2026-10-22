"""Loading DA-VINCIS data for Task 1 (binary) and Task 2 (incident category)."""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List

import pandas as pd

from .preprocess import clean_text

TASK_LABELS: Dict[int, List[str]] = {
    1: ["not violent", "violent"],
    2: ["accidente", "homicidio", "robo", "secuestro", "ninguno"],
}

_ALIASES = {
    1: {"0": "not violent", "1": "violent", "no violento": "not violent", "violento": "violent",
        "not violent incident": "not violent", "violent incident": "violent", "false": "not violent",
        "true": "violent"},
    2: {"accident": "accidente", "homicide": "homicidio", "theft": "robo", "robbery": "robo",
        "kidnapping": "secuestro", "none": "ninguno", "ninguna": "ninguno"},
}
_SPLIT_RE = re.compile(r"[;|,]")


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def normalise_label(label, task: int) -> str:
    key = _strip_accents(str(label).strip().lower())
    key = _ALIASES[task].get(key, key)
    if key not in TASK_LABELS[task]:
        raise ValueError(f"Unknown Task {task} label {label!r}; expected {TASK_LABELS[task]}")
    return key


def load_task(path: str, task: int, text_col: str = "text", label_col: str = "label",
              multilabel: bool = False, clean: bool = True) -> pd.DataFrame:
    """Return ``text`` (cleaned) plus ``label_id`` (int) or ``label_vec`` (multi-hot) columns."""
    df = pd.read_csv(path, sep="\t" if path.endswith(".tsv") else ",")
    if text_col not in df:
        raise KeyError(f"Column {text_col!r} not in {path}; columns: {list(df.columns)}")
    out = pd.DataFrame({"raw": df[text_col].fillna("").astype(str)})
    out["text"] = out["raw"].map(clean_text) if clean else out["raw"]
    labels = TASK_LABELS[task]
    if label_col in df:
        if multilabel:
            vecs = []
            for v in df[label_col].fillna("").astype(str):
                names = {normalise_label(x, task) for x in _SPLIT_RE.split(v) if x.strip()}
                vecs.append([1.0 if l in names else 0.0 for l in labels])
            out["label_vec"] = vecs
        else:
            out["label"] = [normalise_label(l, task) for l in df[label_col]]
            out["label_id"] = out["label"].map({l: i for i, l in enumerate(labels)})
    return out
