"""Find toxic time spans in a transcript. Pure logic, no network, no ffmpeg."""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

@dataclass
class Word:
    start: float
    end: float
    text: str

@dataclass
class Span:
    start_s: float
    end_s: float
    text: str
    score: float

def load_wordlist(path):
    words = set()
    for line in Path(path).read_text().splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            words.add(line)
    return words

def norm(token):
    return re.sub(r"[^a-z0-9']", "", token.lower())

def _join(words):
    return " ".join(w.text.strip() for w in words)

def _narrow(words, score_fn, threshold):
    """Shrink a flagged window from either edge while it stays toxic, so we mute
    as little speech as possible."""
    lo, hi = 0, len(words)
    while hi - lo > 1:
        left = score_fn(_join(words[lo + 1:hi]))
        right = score_fn(_join(words[lo:hi - 1]))
        if left >= threshold and left >= right:
            lo += 1
        elif right >= threshold:
            hi -= 1
        else:
            break
    return lo, hi

def find_spans(words, score_fn, wordlist=frozenset(), threshold=0.8,
               window=6, stride=3, pad_s=0.05, merge_gap_s=0.3,
               use_model=True, use_wordlist=True):
    """Return merged toxic spans.
    score_fn(text) -> probability toxic. use_model / use_wordlist let you run the
    'model only' vs 'model + word list' comparison for the report."""
    raw = []
    if use_wordlist:
        for w in words:
            if norm(w.text) in wordlist:
                raw.append(Span(w.start, w.end, w.text.strip(), 1.0))
    if use_model and words:
        n = len(words)
        starts = list(range(0, max(n - window, 0) + 1, stride))
        if starts[-1] + window < n:      # make sure the tail is covered
            starts.append(n - window)
        for i in starts:
            chunk = words[i:i + window]
            score = score_fn(_join(chunk))
            if score >= threshold:
                lo, hi = _narrow(chunk, score_fn, threshold)
                sub = chunk[lo:hi]
                raw.append(Span(sub[0].start, sub[-1].end, _join(sub),
                                score_fn(_join(sub))))
    return merge(raw, pad_s, merge_gap_s)

def merge(spans, pad_s=0.05, gap_s=0.3):
    spans = sorted(spans, key=lambda s: s.start_s)
    out = []
    for s in spans:
        s = Span(max(0.0, s.start_s - pad_s), s.end_s + pad_s, s.text, s.score)
        if out and s.start_s <= out[-1].end_s + gap_s:
            p = out[-1]
            p.end_s = max(p.end_s, s.end_s)
            if s.text not in p.text:
                p.text = f"{p.text} {s.text}"
            p.score = max(p.score, s.score)
        else:
            out.append(s)
    return out
