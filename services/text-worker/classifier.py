"""Toxicity scorer. Real mode: INT8 ONNX transformer + its tokenizer. Mock mode: keyword
rules, so the stack runs on a laptop without the model files.

Two kinds of model are supported, detected from the number of outputs:
  2 outputs  binary model (v0, v1): softmax, probability of "toxic"
  6 outputs  multi-label model (v2): sigmoid per category (toxic, severe_toxic, obscene,
             threat, insult, identity_hate); the main score is the highest category probability
Every call returns (main_score, {category: probability}); the dict is empty for binary models."""
import os
import re
from pathlib import Path

import numpy as np

MAX_LEN = int(os.environ.get("MAX_LEN", "128"))
THREADS = int(os.environ.get("ORT_THREADS", "1"))
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def _softmax_pos(logits):
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e[..., 1] / e.sum(axis=-1)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class OnnxClassifier:
    def __init__(self, model_dir, model_file="model.int8.onnx", temperature=1.0,
                 temperatures=None, labels=None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer" / "tokenizer.json"))
        self.tok.enable_truncation(MAX_LEN)
        self.tok.no_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = THREADS   # one thread per pod: more pods = more throughput
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(model_dir / model_file), opts,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.temperature = temperature
        self.n_out = self.sess.get_outputs()[0].shape[-1]
        if not isinstance(self.n_out, int):     # dynamic shape: ask the model once
            self.n_out = self._logits(["x"]).shape[-1]
        self.labels = labels or (LABELS if self.n_out == 6 else [])
        self.temperatures = np.array(temperatures or [temperature] * max(self.n_out, 1), dtype=np.float32)

    @property
    def multilabel(self):
        return self.n_out > 2

    def _logits(self, texts):
        encs = self.tok.encode_batch(list(texts))
        n = max(1, max(len(e.ids) for e in encs))
        ids = np.zeros((len(encs), n), dtype=np.int64)
        mask = np.zeros((len(encs), n), dtype=np.int64)
        for i, e in enumerate(encs):
            ids[i, :len(e.ids)] = e.ids
            mask[i, :len(e.ids)] = e.attention_mask
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feeds["token_type_ids"] = np.zeros_like(ids)
        return self.sess.run(None, {k: v for k, v in feeds.items() if k in self.inputs})[0]

    def score_many(self, texts):
        """[(main_score, {category: prob}), ...] for a list of texts, in one model call."""
        if not texts:
            return []
        logits = self._logits(texts)
        if not self.multilabel:
            probs = _softmax_pos(logits / self.temperature)
            return [(float(p), {}) for p in probs]
        probs = _sigmoid(logits / self.temperatures)
        return [(float(row.max()), {k: round(float(v), 4) for k, v in zip(self.labels, row)})
                for row in probs]

    def score_full(self, text):
        return self.score_many([text])[0]

    def score(self, text):
        return self.score_full(text)[0]


class MockClassifier:
    """Same interface; scores by keyword hits. Only for local runs and tests."""
    BAD = {"idiot", "stupid", "hate", "kill", "shut", "ugly", "loser", "trash", "moron", "dumb"}
    multilabel = False
    labels = []

    def __init__(self, *_, **__):
        pass

    def score_full(self, text):
        words = re.findall(r"[a-z']+", text.lower())
        hits = sum(w in self.BAD for w in words)
        return (min(0.05 + 0.45 * hits, 0.99) if words else 0.0), {}

    def score_many(self, texts):
        return [self.score_full(t) for t in texts]

    def score(self, text):
        return self.score_full(text)[0]


def explain(clf, text, max_words=60):
    """Leave-one-word-out occlusion: how much does the score drop when each word is removed?
    Positive weight = the word pushes the post towards toxic. One batched model call."""
    words = text.split()
    if not words:
        return 0.0, []
    words_used = words[:max_words]
    variants = [" ".join(words_used[:i] + words_used[i + 1:]) for i in range(len(words_used))]
    scores = clf.score_many([" ".join(words)] + variants)
    base = scores[0][0]
    out = [{"word": w, "weight": round(base - s[0], 4)} for w, s in zip(words_used, scores[1:])]
    out += [{"word": w, "weight": 0.0} for w in words[max_words:]]
    return base, out
