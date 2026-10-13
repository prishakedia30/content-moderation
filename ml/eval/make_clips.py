"""Build the synthetic video-censoring benchmark: 20 spoken clips with EXACT word timestamps.

Each word is synthesised separately (espeak-ng), trimmed and joined with short random pauses, so the
true start/end of every word is known to the millisecond. Clips vary voice, accent, speed and
background noise; 6 clips contain no abuse at all (including traps such as the name "Dick"), to
measure over-muting. Real phone recordings are still worth adding; this set gives exact ground truth.

  python ml/eval/make_clips.py --out ml/eval/bench     # needs espeak-ng and ffmpeg
Writes clipNN.ogg (16 kHz mono Opus) + clipNN.json ([{start, end, text, toxic}]) + manifest.json."""
import argparse
import json
import random
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

# *word* = toxic. "|" = sentence break (longer pause).
SCRIPTS = [
    "honestly you are such an *idiot* sometimes | can you pass me the remote please",
    "this game is *shit* and the *stupid* referee ruined the whole match",
    "shut up you *pathetic* *loser* | nobody asked for your opinion",
    "I told him the plan twice | he is a complete *moron* and never listens",
    "what the *fuck* were you thinking | we had one simple job today",
    "get out of my class you *worthless* little *bastard*",
    "the food was cold and the waiter was a rude *asshole* | never going back",
    "listen to me carefully | I will find you and *kill* you",
    "you are so *dumb* it hurts | read the instructions again",
    "nice try *bitch* | but you lost again",
    "my brother is an *idiot* but I still love him | he makes great pancakes",
    "stop sending me this *crap* | I am busy with exams this week",
    "that was a *fucking* brilliant goal | what a finish from the striker",
    "you *ugly* *freak* | everyone is laughing at you",
    # clean clips: nothing should be muted
    "the meeting starts at ten so please do not be late | bring your laptop",
    "my friend Dick is a great cook | his pasta is amazing",
    "we need to assess the class assignment before Friday | thanks everyone",
    "the weather in Bangalore is lovely today | let us go for a walk",
    "I hate waiting in long queues | but the museum was worth it",
    "this cloud project uses Kubernetes Redis and a small language model",
]
VOICES = ["en-us", "en-gb", "en-us+f3", "en-gb-scotland", "en-gb-x-rp", "en-us+m3", "en+f4"]
NOISE_DB = [None] * 10 + [20] * 6 + [10] * 4     # signal-to-noise ratio per clip (None = clean audio)


def synth(word, voice, speed, tmp):
    path = Path(tmp) / "w.wav"
    subprocess.run(["espeak-ng", "-v", voice, "-s", str(speed), "-w", str(path), word], check=True)
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    loud = np.where(np.abs(x) > 0.02)[0]
    return (x[loud[0]:loud[-1] + 1] if len(loud) else x), sr


def build(out, seed=7):
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    manifest = []
    order = list(range(len(SCRIPTS)))
    noise = NOISE_DB[:]
    rng.shuffle(noise)
    with tempfile.TemporaryDirectory() as tmp:
        for i in order:
            voice, speed = VOICES[i % len(VOICES)], rng.randint(145, 185)
            audio, words, t, sr = [], [], 0.0, None
            lead = np.zeros(int(0.4 * 22050), np.float32)
            audio.append(lead)
            t = len(lead) / 22050
            for sent_i, sent in enumerate(SCRIPTS[i].split("|")):
                if sent_i:
                    pause = np.zeros(int(rng.uniform(0.45, 0.7) * 22050), np.float32)
                    audio.append(pause)
                    t += len(pause) / 22050
                for tok in sent.split():
                    toxic = tok.startswith("*")
                    word = tok.strip("*")
                    x, sr = synth(word, voice, speed, tmp)
                    words.append({"start": round(t, 3), "end": round(t + len(x) / sr, 3), "text": word, "toxic": toxic})
                    audio.append(x)
                    t += len(x) / sr
                    gap = np.zeros(int(rng.uniform(0.08, 0.22) * sr), np.float32)
                    audio.append(gap)
                    t += len(gap) / sr
            audio.append(np.zeros(int(0.4 * sr), np.float32))
            x = np.concatenate(audio)
            snr = noise[i]
            if snr is not None:
                speech = x[np.abs(x) > 0.02]
                p = float(np.mean(speech ** 2)) if len(speech) else 1e-4
                x = x + np.random.default_rng(i).normal(0, np.sqrt(p / 10 ** (snr / 10)), len(x)).astype(np.float32)
            x = np.clip(x, -1, 1)
            raw = Path(tmp) / "clip.wav"
            with wave.open(str(raw), "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr)
                w.writeframes((x * 32767).astype(np.int16).tobytes())
            name = f"clip{i + 1:02d}"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-ac", "1", "-ar", "16000",
                            "-c:a", "libopus", "-b:a", "32k", str(out / f"{name}.ogg")], check=True)
            (out / f"{name}.json").write_text(json.dumps(words, indent=1))
            manifest.append({"clip": name, "voice": voice, "speed_wpm": speed, "snr_db": snr,
                             "seconds": round(len(x) / sr, 2), "words": len(words),
                             "toxic_words": sum(w["toxic"] for w in words), "transcript": SCRIPTS[i].replace("*", "")})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "bench"))
    m = build(Path(ap.parse_args().out))
    print(f"{len(m)} clips, {sum(c['seconds'] for c in m):.0f} s, {sum(c['words'] for c in m)} words, "
          f"{sum(c['toxic_words'] for c in m)} toxic")
