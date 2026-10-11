"""Censor one video: FFmpeg -> faster-whisper -> classifier -> mute toxic spans.
  python3 pipeline.py in.mp4 out.mp4
  python3 pipeline.py in.mp4 out.mp4 --transcript words.json   # skip Whisper
words.json = [{"start":0.0,"end":0.4,"text":"hello"}, ...]"""
import argparse, json, subprocess, sys, tempfile, time
from pathlib import Path
from classify_client import make_scorer
from spans import Word, find_spans, load_wordlist

HERE = Path(__file__).parent

def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def extract_audio(video, wav):
    run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav)])

def load_wav(path):
    """Read our 16 kHz mono wav into a float32 array. We pass samples straight to
    Whisper, which avoids its own audio decoder (PyAV) and a version clash with it."""
    import wave
    import numpy as np
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

def transcribe(wav, model_size="small"):
    from faster_whisper import WhisperModel          # imported late: heavy
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(load_wav(wav), word_timestamps=True, vad_filter=True)
    return [Word(w.start, w.end, w.word) for seg in segments for w in seg.words]

def mute(video, spans, out):
    if not spans:
        run(["ffmpeg", "-y", "-i", str(video), "-c", "copy", str(out)]); return
    expr = "+".join(f"between(t,{s.start_s:.3f},{s.end_s:.3f})" for s in spans)
    run(["ffmpeg", "-y", "-i", str(video), "-af", f"volume=enable='{expr}':volume=0",
         "-c:v", "copy", str(out)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("out")
    ap.add_argument("--transcript", help="JSON word list; skips Whisper")
    ap.add_argument("--classify-url", default="http://localhost:8000/classify")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--whisper-model", default="small")
    ap.add_argument("--wordlist", default=str(HERE / "wordlist.txt"))
    ap.add_argument("--no-model", action="store_true", help="word list only")
    ap.add_argument("--no-wordlist", action="store_true", help="model only")
    ap.add_argument("--spans-json", help="write spans here")
    a = ap.parse_args()

    t0 = time.time()
    if a.transcript:
        words = [Word(**w) for w in json.loads(Path(a.transcript).read_text())]
    else:
        with tempfile.TemporaryDirectory() as d:
            wav = Path(d) / "a.wav"
            extract_audio(a.video, wav)
            words = transcribe(wav, a.whisper_model)
    spans = find_spans(words, make_scorer(a.classify_url), load_wordlist(a.wordlist),
                       threshold=a.threshold, use_model=not a.no_model,
                       use_wordlist=not a.no_wordlist)
    mute(a.video, spans, a.out)
    if a.spans_json:
        Path(a.spans_json).write_text(json.dumps([s.__dict__ for s in spans], indent=2))
    print(f"{len(words)} words, {len(spans)} spans muted, {time.time()-t0:.1f}s")
    for s in spans:
        print(f"  {s.start_s:6.2f}-{s.end_s:6.2f}  {s.score:.2f}  {s.text}")

if __name__ == "__main__":
    main()