"""Evaluation. Task 1 uses F1 of the 'violent' class; Task 2 reports macro and weighted F1."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


def evaluate(y_true, y_pred, labels: List[str], task: int, multilabel: bool = False) -> Dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    res: Dict = {}
    if task == 1 and not multilabel:
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
        res.update(precision=p, recall=r, f1=f)
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0,
                                                     labels=None if multilabel else list(range(len(labels))))
        res[f"{avg}_precision"], res[f"{avg}_recall"], res[f"{avg}_f1"] = p, r, f
    if "f1" not in res:  # Task 2: headline F1 = macro F1
        res.update(precision=res["macro_precision"], recall=res["macro_recall"], f1=res["macro_f1"])
    res["accuracy"] = accuracy_score(y_true, y_pred)
    res["per_class"] = classification_report(y_true, y_pred, target_names=labels, output_dict=True,
                                             zero_division=0, labels=None if multilabel else list(range(len(labels))))
    return res


def summary_line(res: Dict) -> str:
    return (f"P {res['precision']:.4f} · R {res['recall']:.4f} · F1 {res['f1']:.4f} · "
            f"macro-F1 {res['macro_f1']:.4f} · acc {res['accuracy']:.4f}")


def save(res: Dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
