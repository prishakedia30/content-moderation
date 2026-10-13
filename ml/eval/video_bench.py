"""Video-censoring benchmark on the labelled clips from make_clips.py.

  python ml/eval/video_bench.py --clips ml/eval/bench --model-dir <deploy dir> --out results/video
  (or --classify-url http://<server>/classify instead of --model-dir)

1. Whisper tiny / base / small (faster-whisper, CPU int8, same settings as the video-worker):
   word error rate, word timestamp error (ms), speed (seconds of audio per second of compute).
2. The censoring itself, end to end on each Whisper size: toxic words muted (recall), share of muted
   time that really was abuse (precision), clean speech muted, boundary error in ms, and
   model-only vs word-list-only vs both.
A toxic word counts as muted when at least half of its true duration is silenced."""
import argparse
import difflib
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "services" / "video-worker"))
sys.path.insert(0, str(ROOT / "services" / "text-worker"))
from spans import Word, find_spans, load_wordlist  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--clips", default=str(ROOT / "ml" / "eval" / "bench"))
ap.add_argument("--model-dir")
ap.add_argument("--classify-url")
ap.add_argument("--sizes", default="tiny,base,small")
ap.add_argument("--threshold", type=float, default=0.75, help="SPAN_THRESHOLD of the video-worker")
ap.add_argument("--out", default="results/video")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

if a.model_dir:
    from classifier import OnnxClassifier
    from decision import load_thresholds
    cfg = load_thresholds(a.model_dir)
    clf = OnnxClassifier(a.model_dir, "model.int8.onnx", cfg["temperature"], temperatures=cfg["temperatures"],
                         labels=cfg["labels"])
    _cache = {}

    def score(text):
        if text not in _cache:
            _cache[text] = clf.score(text)
        return _cache[text]
elif a.classify_url:
    from classify_client import make_scorer
    score = make_scorer(a.classify_url)
else:
    sys.exit("give --model-dir or --classify-url")

WORDLIST = load_wordlist(ROOT / "services" / "video-worker" / "wordlist.txt")
clips = []
for j in sorted(Path(a.clips).glob("clip*.json")):
    clips.append({"name": j.stem, "words": json.loads(j.read_text()), "audio": j.with_suffix(".ogg")})
manifest = {m["clip"]: m for m in json.loads((Path(a.clips) / "manifest.json").read_text())}


def load_audio(path):
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", "16000",
                        "-f", "s16le", f.name], check=True)
        return np.fromfile(f.name, dtype=np.int16).astype(np.float32) / 32768.0


def norm(t):
    return re.sub(r"[^a-z0-9']", "", t.lower())


def wer_and_alignment(ref, hyp):
    """Word error rate (edit distance / reference words) and matched (ref_i, hyp_j) pairs."""
    r, h = [norm(w["text"]) for w in ref], [norm(w.text) for w in hyp]
    h_idx = [j for j, x in enumerate(h) if x]
    h = [h[j] for j in h_idx]
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0], d[0, :] = np.arange(len(r) + 1), np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    pairs = []
    for blk in difflib.SequenceMatcher(None, r, h, autojunk=False).get_matching_blocks():
        pairs += [(blk.a + k, h_idx[blk.b + k]) for k in range(blk.size)]
    return d[len(r), len(h)] / max(len(r), 1), pairs


def censor_metrics(ref, spans):
    """Word-level recall, time-based precision, clean speech muted, boundary errors."""
    def muted_time(s, e):
        return sum(max(0.0, min(e, sp.end_s) - max(s, sp.start_s)) for sp in spans)
    tox = [w for w in ref if w["toxic"]]
    caught = [w for w in tox if muted_time(w["start"], w["end"]) >= 0.5 * (w["end"] - w["start"])]
    total_muted = sum(sp.end_s - sp.start_s for sp in spans)
    on_toxic = sum(muted_time(w["start"], w["end"]) for w in tox)
    clean = [w for w in ref if not w["toxic"]]
    clean_muted = sum(muted_time(w["start"], w["end"]) for w in clean)
    clean_words_muted = sum(muted_time(w["start"], w["end"]) >= 0.5 * (w["end"] - w["start"]) for w in clean)
    start_err, end_err = [], []
    for w in caught:
        sp = max(spans, key=lambda s: min(w["end"], s.end_s) - max(w["start"], s.start_s))
        start_err.append((w["start"] - sp.start_s) * 1000)   # > 0: muting began early (extra silence before)
        end_err.append((sp.end_s - w["end"]) * 1000)          # > 0: muting ran late (extra silence after)
    return {"toxic_words": len(tox), "caught": len(caught), "muted_s": total_muted, "muted_on_toxic_s": on_toxic,
            "clean_speech_s": sum(w["end"] - w["start"] for w in clean), "clean_muted_s": clean_muted,
            "clean_words_muted": int(clean_words_muted), "start_err_ms": start_err, "end_err_ms": end_err}


def summarize(ms):
    tw, cg = sum(m["toxic_words"] for m in ms), sum(m["caught"] for m in ms)
    muted, on_tox = sum(m["muted_s"] for m in ms), sum(m["muted_on_toxic_s"] for m in ms)
    cs, cm = sum(m["clean_speech_s"] for m in ms), sum(m["clean_muted_s"] for m in ms)
    se = [x for m in ms for x in m["start_err_ms"]]
    ee = [x for m in ms for x in m["end_err_ms"]]
    p = on_tox / muted if muted else 0.0
    r = cg / tw if tw else 0.0
    return {"toxic_word_recall": r, "muted_time_precision": p, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "clean_speech_muted_s": cm, "clean_speech_muted_pct": cm / cs if cs else 0.0,
            "clean_words_muted": int(sum(m["clean_words_muted"] for m in ms)),
            "boundary_start_ms_mean": float(np.mean(se)) if se else None, "boundary_end_ms_mean": float(np.mean(ee)) if ee else None,
            "boundary_abs_ms_mean": float(np.mean(np.abs(se + ee))) if se else None}


from faster_whisper import WhisperModel  # noqa: E402

audio = {c["name"]: load_audio(c["audio"]) for c in clips}
seconds = sum(len(x) / 16000 for x in audio.values())
res = {"clips": len(clips), "audio_seconds": seconds, "words": sum(len(c["words"]) for c in clips),
       "toxic_words": sum(w["toxic"] for c in clips for w in c["words"]), "threshold": a.threshold,
       "asr": [], "censor": [], "per_clip": []}
configs = {"model only": (True, False), "word list only": (False, True), "model + word list": (True, True)}
for size in a.sizes.split(","):
    model = WhisperModel(size, device="cpu", compute_type="int8")
    werrs, st_err, en_err, compute = [], [], [], 0.0
    hyps = {}
    for c in clips:
        t0 = time.perf_counter()
        segs, _ = model.transcribe(audio[c["name"]], word_timestamps=True, vad_filter=True)
        hyp = [Word(w.start, w.end, w.word) for s in segs for w in s.words or []]
        compute += time.perf_counter() - t0
        hyps[c["name"]] = hyp
        wer, pairs = wer_and_alignment(c["words"], hyp)
        werrs.append((wer, len(c["words"])))
        for i, j in pairs:
            st_err.append(abs(hyp[j].start - c["words"][i]["start"]) * 1000)
            en_err.append(abs(hyp[j].end - c["words"][i]["end"]) * 1000)
    total_words = sum(n for _, n in werrs)
    res["asr"].append({"whisper": size, "wer": sum(w * n for w, n in werrs) / total_words,
                       "start_err_ms_median": float(np.median(st_err)), "end_err_ms_median": float(np.median(en_err)),
                       "start_err_ms_mean": float(np.mean(st_err)), "end_err_ms_mean": float(np.mean(en_err)),
                       "compute_s": compute, "realtime_factor": seconds / compute})
    print(json.dumps(res["asr"][-1]), flush=True)
    # censoring on this transcript, and on the perfect transcript once (isolates the span finder)
    sources = {size: hyps} if size != a.sizes.split(",")[0] else \
        {"perfect transcript": {c["name"]: [Word(w["start"], w["end"], w["text"]) for w in c["words"]] for c in clips}, size: hyps}
    for src, hyp_map in sources.items():
        for cname, (um, uw) in configs.items():
            ms = []
            for c in clips:
                spans = find_spans(hyp_map[c["name"]], score, WORDLIST, threshold=a.threshold, use_model=um, use_wordlist=uw)
                m = censor_metrics(c["words"], spans)
                ms.append(m)
                if cname == "model + word list" and src == "base":
                    res["per_clip"].append({"clip": c["name"], "snr_db": manifest[c["name"]]["snr_db"],
                                            "voice": manifest[c["name"]]["voice"], **{k: v for k, v in m.items() if not k.endswith("err_ms")},
                                            "spans": [[round(s.start_s, 2), round(s.end_s, 2), s.text] for s in spans]})
            res["censor"].append({"transcript": src, "config": cname, **summarize(ms)})
            print(src, cname, json.dumps(res["censor"][-1]), flush=True)
    del model

# noise breakdown (base + model + word list)
by_noise = {}
for r in res["per_clip"]:
    by_noise.setdefault(str(r["snr_db"]), []).append(r)
res["by_noise_base"] = {k: {"clips": len(v), "recall": sum(x["caught"] for x in v) / max(sum(x["toxic_words"] for x in v), 1),
                            "clean_words_muted": sum(x["clean_words_muted"] for x in v)} for k, v in by_noise.items()}
(out / "video_bench.json").write_text(json.dumps(res, indent=1))

L = [f"# Video censoring benchmark", "", f"{res['clips']} synthetic clips ({seconds:.0f} s, {res['words']} words, {res['toxic_words']} toxic). "
     "Exact word timings; voices, accents, speeds and background noise vary; 6 clips contain no abuse.", "",
     "## Speech recognition", "", "| Whisper | WER | Start error median (ms) | End error median (ms) | Speed (× real time) |", "|---|---|---|---|---|"]
L += [f"| {r['whisper']} | {r['wer']:.1%} | {r['start_err_ms_median']:.0f} | {r['end_err_ms_median']:.0f} | {r['realtime_factor']:.1f}× |" for r in res["asr"]]
L += ["", "## Censoring", "", "| Transcript | Detector | Toxic words muted | Muted time that was abuse | Clean words muted | Clean speech muted | Boundary error, mean abs (ms) |", "|---|---|---|---|---|---|---|"]
L += [f"| {r['transcript']} | {r['config']} | {r['toxic_word_recall']:.0%} | {r['muted_time_precision']:.0%} | {r['clean_words_muted']} | "
      f"{r['clean_speech_muted_s']:.2f} s ({r['clean_speech_muted_pct']:.1%}) | {r['boundary_abs_ms_mean'] if r['boundary_abs_ms_mean'] is None else round(r['boundary_abs_ms_mean'])} |"
      for r in res["censor"]]
L += ["", "Boundary error: distance between where muting starts/ends and where the toxic word really starts/ends "
      "(the span finder pads 50 ms on each side on purpose).", "", "## Noise (Whisper base, model + word list)", "",
      "| SNR | Clips | Toxic words muted | Clean words muted |", "|---|---|---|---|"]
L += [f"| {'clean' if k == 'None' else k + ' dB'} | {v['clips']} | {v['recall']:.0%} | {v['clean_words_muted']} |" for k, v in sorted(res["by_noise_base"].items())]
(out / "VIDEO.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
