"""End-to-end fine-tuning on the sample with a tiny random model (skipped without torch/transformers
or network access)."""
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from davincis.predict import main as predict_main  # noqa: E402
from davincis.train import main as train_main  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "data" / "sample"
TINY = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import AutoTokenizer
    try:
        AutoTokenizer.from_pretrained(TINY)
    except Exception as e:  # offline
        pytest.skip(f"cannot download {TINY}: {e}")
    return TINY


@pytest.mark.parametrize("task,label_col", [(1, "label"), (2, "category")])
def test_train_and_predict(tmp_path, tiny_model, task, label_col):
    out = tmp_path / f"task{task}"
    train_main(["--task", str(task), "--label-col", label_col, "--train", str(S / "train.csv"),
                "--dev", str(S / "dev.csv"), "--test", str(S / "test.csv"), "--model", tiny_model,
                "--config", str(ROOT / "configs/distilbeto.yaml"), "--epochs", "1", "--out", str(out)])
    res = json.loads((out / "test_metrics.json").read_text())
    assert 0.0 <= res["f1"] <= 1.0
    predict_main(["--model", str(out / "model"), "--input", str(S / "test.csv"), "--output", str(tmp_path / "p.csv")])
    assert (tmp_path / "p.csv").read_text().count("\n") == 4
