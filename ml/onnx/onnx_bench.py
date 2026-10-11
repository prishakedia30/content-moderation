import inspect, os, shutil, time, json
import numpy as np
import torch
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = os.environ.get("MODEL", "martin-ha/toxic-comment-model")
OUT, THREADS, WARMUP, RUNS = os.environ.get("OUT", "artifacts"), 2, 20, 300
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(THREADS)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
tok.save_pretrained(f"{OUT}/tokenizer")
if os.path.isfile(os.path.join(MODEL, "thresholds.yaml")):   # fine-tuned model: keep its calibration
    shutil.copy(os.path.join(MODEL, "thresholds.yaml"), f"{OUT}/thresholds.yaml")

class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

fp32_path, int8_path = f"{OUT}/model.onnx", f"{OUT}/model.int8.onnx"
dummy = tok("export example", return_tensors="pt")
torch.onnx.export(
    Wrap(model).eval(), (dummy["input_ids"], dummy["attention_mask"]), fp32_path,
    input_names=["input_ids", "attention_mask"], output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
    opset_version=17,
    # newer PyTorch defaults to the dynamo exporter; the classic one is what we validated
    **({"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}),
)
model.eval()
quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)

TEXTS = [
    "Thanks for sharing, this was really helpful!",
    "You are an idiot and nobody wants you here.",
    "I disagree with your point about the budget.",
    "Shut up, you worthless piece of garbage.",
    "Great game last night, what a finish.",
    "People like you should just disappear.",
    "Can someone explain how this API works?",
    "This is the dumbest thing I have ever read, moron.",
    "Happy birthday! Hope you have a lovely day.",
    "I will find you and make you regret this.",
    "The new update fixed the login bug for me.",
    "What a stupid, pathetic excuse for an argument.",
    "Does anyone have notes from today's lecture?",
    "Go to hell, everyone hates you.",
    "The weather in Bangalore is lovely today.",
    "You're a complete waste of space.",
]

def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)

def torch_prob(t):
    enc = tok(t, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        return torch.softmax(model(**enc).logits, -1)[0, 1].item()

def make_ort(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = THREADS
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    def prob(t):
        enc = tok(t, return_tensors="np", truncation=True, max_length=128)
        feeds = {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)}
        return float(softmax(sess.run(None, feeds)[0])[0, 1])
    return prob

def bench(fn):
    times = []
    for i in range(WARMUP + RUNS):
        t = TEXTS[i % len(TEXTS)]
        s = time.perf_counter()
        fn(t)
        if i >= WARMUP:
            times.append((time.perf_counter() - s) * 1000)
    return [round(float(v), 2) for v in np.percentile(times, [50, 95, 99])]

torch_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
backends = {
    "PyTorch FP32": (torch_prob, torch_mb),
    "ONNX FP32": (make_ort(fp32_path), os.path.getsize(fp32_path) / 1e6),
    "ONNX INT8": (make_ort(int8_path), os.path.getsize(int8_path) / 1e6),
}

ref = [torch_prob(t) for t in TEXTS]
rows = []
for name, (fn, mb) in backends.items():
    probs = [fn(t) for t in TEXTS]
    agree = np.mean([(p >= 0.5) == (r >= 0.5) for p, r in zip(probs, ref)]) * 100
    diff = max(abs(p - r) for p, r in zip(probs, ref))
    p50, p95, p99 = bench(fn)
    rows.append({"backend": name, "size_mb": round(mb, 1), "p50_ms": p50, "p95_ms": p95, "p99_ms": p99,
                 "agree_with_pytorch_pct": round(float(agree), 1), "max_prob_diff": round(float(diff), 4)})

lines = [f"Model: {MODEL} | threads: {THREADS} | runs: {RUNS} (latency includes tokenization)", "",
         "| Backend | Size (MB) | p50 (ms) | p95 (ms) | p99 (ms) | Agrees with PyTorch (%) | Max prob diff |",
         "|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['backend']} | {r['size_mb']} | {r['p50_ms']} | {r['p95_ms']} | {r['p99_ms']} | {r['agree_with_pytorch_pct']} | {r['max_prob_diff']} |")
open(f"{OUT}/results.md", "w").write("\n".join(lines) + "\n")
json.dump(rows, open(f"{OUT}/results.json", "w"), indent=2)
print("\n".join(lines))
