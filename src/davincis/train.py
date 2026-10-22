"""Fine-tune DistilBETO (or any Spanish encoder) for DA-VINCIS Task 1 or Task 2.

Paper settings: learning rate 1e-4, batch size 64, max length 128, dropout 0.2,
up to 10 epochs with early stopping on the validation set.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Dataset
from transformers import (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments)

from .data import TASK_LABELS, load_task
from .metrics import evaluate, save, summary_line

DROPOUT_FIELDS = ("dropout", "seq_classif_dropout", "hidden_dropout_prob", "attention_probs_dropout_prob",
                  "classifier_dropout")


def to_pred(logits: np.ndarray, multilabel: bool) -> np.ndarray:
    return (logits > 0).astype(int) if multilabel else logits.argmax(-1)


def build_model(name: str, labels, dropout: float, multilabel: bool):
    cfg = AutoConfig.from_pretrained(
        name, num_labels=len(labels), id2label=dict(enumerate(labels)),
        label2id={l: i for i, l in enumerate(labels)},
        problem_type="multi_label_classification" if multilabel else "single_label_classification")
    for f in DROPOUT_FIELDS:
        if hasattr(cfg, f):
            setattr(cfg, f, dropout)
    return AutoModelForSequenceClassification.from_pretrained(name, config=cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", type=int, choices=[1, 2], required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--test", default=None)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--multilabel", action="store_true", help="Task 2 with several categories per tweet")
    ap.add_argument("--config", default="configs/distilbeto.yaml")
    ap.add_argument("--model", default=None, help="override the model in the config")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    name = args.model or cfg["model"]
    ml = args.multilabel and args.task == 2
    labels = TASK_LABELS[args.task]
    out = Path(args.out or f"outputs/task{args.task}")
    load = lambda p: load_task(p, args.task, args.text_col, args.label_col, ml)  # noqa: E731
    train, dev = load(args.train), (load(args.dev) if args.dev else None)
    tok = AutoTokenizer.from_pretrained(name)

    def ds(df):
        cols = {"text": df["text"].tolist()}
        if "label_vec" in df:
            cols["labels"] = df["label_vec"].tolist()
        elif "label_id" in df:
            cols["labels"] = df["label_id"].astype(int).tolist()
        return Dataset.from_dict(cols).map(lambda b: tok(b["text"], truncation=True, max_length=cfg["max_length"]),
                                           batched=True, remove_columns=["text"])

    def gold(df):
        return np.array(df["label_vec"].tolist(), dtype=int) if ml else df["label_id"].values

    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"), learning_rate=float(cfg["learning_rate"]),
        per_device_train_batch_size=cfg["batch_size"], per_device_eval_batch_size=cfg["batch_size"],
        num_train_epochs=args.epochs or cfg["epochs"],
        eval_strategy="epoch" if dev is not None else "no", save_strategy="epoch" if dev is not None else "no",
        save_total_limit=1, load_best_model_at_end=dev is not None, metric_for_best_model="f1",
        report_to=[], seed=args.seed, fp16=torch.cuda.is_available(), logging_steps=20)
    trainer = Trainer(
        model=build_model(name, labels, cfg["dropout"], ml), args=targs, train_dataset=ds(train),
        eval_dataset=ds(dev) if dev is not None else None, data_collator=DataCollatorWithPadding(tok),
        compute_metrics=lambda p: {"f1": evaluate(p.label_ids.astype(int), to_pred(p.predictions, ml),
                                                  labels, args.task, ml)["f1"]},
        callbacks=[EarlyStoppingCallback(cfg["early_stopping_patience"])] if dev is not None else [])
    trainer.train()
    trainer.save_model(str(out / "model"))
    tok.save_pretrained(str(out / "model"))

    for split, path in (("dev", args.dev), ("test", args.test)):
        if not path:
            continue
        df = load(path)
        pred = to_pred(trainer.predict(ds(df)).predictions, ml)
        names = [";".join(l for l, v in zip(labels, row) if v) for row in pred] if ml else [labels[i] for i in pred]
        df.drop(columns=[c for c in ("label_vec", "label_id") if c in df]).assign(prediction=names).to_csv(
            out / f"{split}_predictions.csv", index=False)
        if "label_id" in df or "label_vec" in df:
            res = evaluate(gold(df), pred, labels, args.task, ml)
            print(f"Task {args.task} [{split}] {summary_line(res)}")
            save(res, out / f"{split}_metrics.json")


if __name__ == "__main__":
    main()
