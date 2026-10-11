"""Video ML result: does the word list help the span finder? Compares model-only, word-list-only
and model + word list on hand-labelled transcripts.
Each file in --clips is a JSON list of words: {"start", "end", "text", "toxic": true/false}.
  python eval_spans.py --clips clips --model-dir ~/onnx/artifacts        # local ONNX model
  python eval_spans.py --clips clips --classify-url http://<ip>/classify  # deployed API"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "services" / "video-worker"))
sys.path.insert(0, str(ROOT / "services" / "text-worker"))
from spans import Word, find_spans, load_wordlist  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--clips", required=True)
ap.add_argument("--model-dir", help="folder with model.int8.onnx + tokenizer/")
ap.add_argument("--classify-url", help="e.g. http://13.211.204.53/classify")
ap.add_argument("--threshold", type=float, default=0.75)
a = ap.parse_args()

if a.model_dir:
    from classifier import OnnxClassifier
    score = OnnxClassifier(Path(a.model_dir).expanduser()).score
elif a.classify_url:
    from classify_client import make_scorer
    score = make_scorer(a.classify_url)
else:
    sys.exit("give --model-dir or --classify-url")

wordlist = load_wordlist(ROOT / "services" / "video-worker" / "wordlist.txt")
clips = [json.loads(p.read_text()) for p in sorted(Path(a.clips).glob("*.json"))]
configs = {"Model only": (True, False), "Word list only": (False, True), "Model + word list": (True, True)}

print(f"{len(clips)} clips, {sum(len(c) for c in clips)} words, "
      f"{sum(w['toxic'] for c in clips for w in c)} labelled toxic\n")
print("| Config | Precision | Recall | F1 | Clean speech muted (s) |")
print("|---|---|---|---|---|")
for name, (use_model, use_list) in configs.items():
    tp = fp = fn = 0
    clean_muted = 0.0
    for clip in clips:
        words = [Word(w["start"], w["end"], w["text"]) for w in clip]
        spans = find_spans(words, score, wordlist, threshold=a.threshold,
                           use_model=use_model, use_wordlist=use_list)
        for w in clip:
            mid = (w["start"] + w["end"]) / 2
            muted = any(s.start_s <= mid <= s.end_s for s in spans)
            tp += muted and w["toxic"]
            fp += muted and not w["toxic"]
            fn += (not muted) and w["toxic"]
            if muted and not w["toxic"]:
                clean_muted += w["end"] - w["start"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    print(f"| {name} | {p:.2f} | {r:.2f} | {f1:.2f} | {clean_muted:.1f} |")
