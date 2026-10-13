import os

import pytest
from fastapi.testclient import TestClient

from decision import decide, load_thresholds
from fake_model import build


def test_decide_bands():
    assert decide(0.10, 0.3, 0.75) == "ALLOW"
    assert decide(0.30, 0.3, 0.75) == "REVIEW"
    assert decide(0.74, 0.3, 0.75) == "REVIEW"
    assert decide(0.75, 0.3, 0.75) == "BLOCK"


def test_thresholds_file_then_env(tmp_path, monkeypatch):
    (tmp_path / "thresholds.yaml").write_text("t_low: 0.2\nt_high: 0.9\ntemperature: 1.5\n")
    cfg = load_thresholds(tmp_path)
    assert {k: cfg[k] for k in ("t_low", "t_high", "temperature")} == \
        {"t_low": 0.2, "t_high": 0.9, "temperature": 1.5}
    assert cfg["categories"] == {}
    monkeypatch.setenv("T_HIGH", "0.8")
    assert load_thresholds(tmp_path)["t_high"] == 0.8


def test_bad_thresholds_rejected(tmp_path):
    (tmp_path / "thresholds.yaml").write_text("t_low: 0.9\nt_high: 0.2\n")
    with pytest.raises(ValueError):
        load_thresholds(tmp_path)


def test_onnx_classifier_scores(tmp_path):
    from classifier import OnnxClassifier
    clf = OnnxClassifier(build(tmp_path))
    assert clf.score("you are an idiot") > 0.9
    assert clf.score("have a nice day") < 0.1
    assert 0.0 <= clf.score("") <= 1.0          # empty text still works
    assert clf.score("word " * 500) <= 1.0      # long text is truncated, not an error


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["MODEL_DIR"] = str(build(tmp_path_factory.mktemp("model")))
    os.environ["CONSUMER_ENABLED"] = "0"
    import app as app_module   # reads MODEL_DIR at import time
    with TestClient(app_module.app) as c:
        yield c


def test_api(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").status_code == 200
    out = client.post("/classify", json={"text": "you stupid moron"}).json()
    assert out["decision"] == "BLOCK" and out["model_version"] == os.environ.get("MODEL_VERSION", "v0")
    batch = client.post("/classify/batch", json={"texts": ["hello", "idiot idiot"]}).json()
    assert [b["decision"] for b in batch] == ["ALLOW", "BLOCK"]
    assert "moderation_decisions_total" in client.get("/metrics").text


def test_api_normalize_and_explain(client):
    plain = client.post("/classify", json={"text": "you are an 1d10t"}).json()
    assert plain["decision"] == "BLOCK" and plain["normalized"] == "you are an idiot"
    off = client.post("/classify", json={"text": "you are an 1d10t", "normalize": False}).json()
    assert off["decision"] == "ALLOW" and off["normalized"] is None
    ex = client.post("/explain", json={"text": "have a nice day idiot"}).json()
    top = max(ex["words"], key=lambda w: w["weight"])
    assert top["word"] == "idiot" and top["weight"] > 0.5
    assert [w["word"] for w in ex["words"]] == ["have", "a", "nice", "day", "idiot"]


def test_multilabel_model(tmp_path):
    from classifier import OnnxClassifier
    from decision import flags, load_thresholds
    d = build(tmp_path, n_labels=6, thresholds=(
        "t_low: 0.2\nt_high: 0.9\ntemperature: 1.0\n"
        "labels: [toxic, severe_toxic, obscene, threat, insult, identity_hate]\n"
        "temperatures: [1, 1, 1, 1, 1, 1]\ncategories: {threat: 0.5, insult: 0.5}\n"))
    cfg = load_thresholds(d)
    clf = OnnxClassifier(d, temperatures=cfg["temperatures"], labels=cfg["labels"])
    assert clf.multilabel
    main, cats = clf.score_full("i will kill you")
    assert main > 0.9 and cats["threat"] > 0.9 and cats["insult"] < 0.1
    assert flags(cats, cfg["categories"]) == ["threat"]
    main, cats = clf.score_full("have a nice day")
    assert main < 0.1 and flags(cats, cfg["categories"]) == []
