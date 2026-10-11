"""Toxicity scorer. Real mode: INT8 ONNX DistilBERT + its tokenizer. Mock mode: keyword
rules, so the stack runs on a laptop without the model files."""
import os
import re
from pathlib import Path

import numpy as np

MAX_LEN = int(os.environ.get("MAX_LEN", "128"))
THREADS = int(os.environ.get("ORT_THREADS", "1"))


class OnnxClassifier:
    def __init__(self, model_dir, model_file="model.int8.onnx", temperature=1.0):
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
        self.temperature = temperature

    def score(self, text):
        enc = self.tok.encode(text)
        feeds = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
        }
        logits = self.sess.run(None, feeds)[0][0] / self.temperature
        e = np.exp(logits - logits.max())
        return float(e[1] / e.sum())


class MockClassifier:
    """Same interface; scores by keyword hits. Only for local runs and tests."""
    BAD = {"idiot", "stupid", "hate", "kill", "shut", "ugly", "loser", "trash", "moron", "dumb"}

    def __init__(self, *_, **__):
        pass

    def score(self, text):
        words = re.findall(r"[a-z']+", text.lower())
        hits = sum(w in self.BAD for w in words)
        return min(0.05 + 0.45 * hits, 0.99) if words else 0.0
