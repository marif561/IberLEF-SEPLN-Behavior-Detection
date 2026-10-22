from pathlib import Path

import pytest

from davincis.data import TASK_LABELS, load_task, normalise_label
from davincis.metrics import evaluate
from davincis.preprocess import clean_text

S = Path(__file__).resolve().parents[1] / "data" / "sample"


def test_clean_text():
    out = clean_text("Me robaron el celular 😡 en el metro https://t.co/x #inseguridad @policia ¡¡Ayuda!!")
    assert out == "Me robaron el celular en el metro Ayuda"
    assert clean_text("Volcadura en la carretera, ¿heridos?") == "Volcadura en la carretera heridos"


def test_load_task1_and_task2():
    t1 = load_task(str(S / "train.csv"), 1)
    t2 = load_task(str(S / "train.csv"), 2, label_col="category")
    assert set(t1["label"]) == set(TASK_LABELS[1])
    assert set(t2["label"]) <= set(TASK_LABELS[2]) and t2["label_id"].between(0, 4).all()


def test_label_aliases():
    assert normalise_label("Violento", 1) == "violent"
    assert normalise_label("Homicide", 2) == "homicidio"
    assert normalise_label("ACCIDENTE", 2) == "accidente"
    with pytest.raises(ValueError):
        normalise_label("fraude", 2)


def test_multilabel(tmp_path):
    f = tmp_path / "m.csv"
    f.write_text("text,category\nrobo y secuestro,robo;secuestro\nnada,ninguno\n", encoding="utf-8")
    df = load_task(str(f), 2, label_col="category", multilabel=True)
    assert df["label_vec"][0] == [0.0, 0.0, 1.0, 1.0, 0.0]


def test_metrics():
    r1 = evaluate([1, 1, 0, 0], [1, 0, 0, 0], TASK_LABELS[1], task=1)
    assert r1["precision"] == 1.0 and r1["recall"] == 0.5
    r2 = evaluate([0, 1, 2, 3, 4], [0, 1, 2, 3, 3], TASK_LABELS[2], task=2)
    assert r2["f1"] == r2["macro_f1"] and r2["accuracy"] == 0.8
