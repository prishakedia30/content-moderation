"""video-worker: turns Sanvi's pipeline into a service.
Reads the Redis stream videos.raw; for each video: download -> FFmpeg audio -> Whisper words ->
toxic spans (text model via /classify + word list) -> mute -> upload censored copy -> record spans.
Serves /health, /ready, /metrics like the other services."""
import logging
import os
import socket
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import redis
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from classify_client import make_scorer
from pipeline import extract_audio, get_whisper, has_audio, mute, transcribe
from spans import find_spans, load_wordlist

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("video-worker")

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CLASSIFY_URL = os.environ.get("CLASSIFY_URL", "http://text-worker:8000/classify")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
SPAN_THRESHOLD = float(os.environ.get("SPAN_THRESHOLD", "0.75"))
USE_WORDLIST = os.environ.get("USE_WORDLIST", "1") == "1"
STREAM, GROUP = "videos.raw", "video-workers"

PROCESSED = Counter("videos_processed_total", "Videos processed", ["status"])
PROC_S = Histogram("video_processing_seconds", "Time to censor one video",
                   buckets=(1, 2, 5, 10, 20, 30, 60, 120, 300, 600))
SPANS = Counter("video_spans_muted_total", "Toxic spans muted")
QUEUE_LAG = Gauge("moderation_queue_lag", "Messages waiting in a stream", ["stream"])

state = {"ready": False}
stop = threading.Event()


def process(db, storage, video_id, raw_key):
    t0 = time.time()
    db.execute("UPDATE videos SET status='processing', updated_at=now() WHERE video_id=%s", (video_id,))
    ext = Path(raw_key).suffix or ".mp4"
    clean_key = f"videos-clean/{video_id}{ext}"
    with tempfile.TemporaryDirectory() as d:
        src, out, wav = Path(d) / f"in{ext}", Path(d) / f"out{ext}", Path(d) / "a.wav"
        storage.get_file(raw_key, src)
        words = []
        if has_audio(src):
            extract_audio(src, wav)
            words = transcribe(wav, WHISPER_MODEL)
        spans = find_spans(words, make_scorer(CLASSIFY_URL), load_wordlist(HERE / "wordlist.txt"),
                           threshold=SPAN_THRESHOLD, use_wordlist=USE_WORDLIST)
        mute(src, spans, out)
        storage.put_file(out, clean_key)
    elapsed = time.time() - t0
    db.execute("DELETE FROM video_spans WHERE video_id=%s", (video_id,))
    for s in spans:
        db.execute("INSERT INTO video_spans (video_id, start_s, end_s, text, score) "
                   "VALUES (%s, %s, %s, %s, %s)",
                   (video_id, s.start_s, s.end_s, s.text, min(max(s.score, 0.0), 1.0)))
    db.execute("UPDATE videos SET status='done', clean_key=%s, n_words=%s, n_spans=%s, "
               "processing_s=%s, error=NULL, updated_at=now() WHERE video_id=%s",
               (clean_key, len(words), len(spans), elapsed, video_id))
    PROC_S.observe(elapsed)
    SPANS.inc(len(spans))
    log.info("video %s: %d words, %d spans muted, %.1fs", video_id, len(words), len(spans), elapsed)


def consume_forever():
    from common.db import DB
    from common.storage import Storage
    storage = Storage()
    db = DB().connect()
    db.ensure_schema()
    log.info("loading Whisper model %s", WHISPER_MODEL)
    get_whisper(WHISPER_MODEL)
    state["ready"] = True
    consumer = socket.gethostname()
    while not stop.is_set():
        try:
            r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            try:
                r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
            except redis.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise
            while not stop.is_set():
                msgs = r.xautoclaim(STREAM, GROUP, consumer, min_idle_time=600_000,
                                    start_id="0-0", count=1)[1]
                if not msgs:
                    resp = r.xreadgroup(GROUP, consumer, {STREAM: ">"}, count=1, block=5000) or []
                    msgs = [m for _, batch in resp for m in batch]
                for msg_id, fields in msgs:
                    if fields:
                        try:
                            process(db, storage, fields["video_id"], fields["s3_key"])
                            PROCESSED.labels("done").inc()
                        except Exception as e:
                            # Record the failure and move on, so one bad file can't block the queue.
                            log.exception("video %s failed", fields.get("video_id"))
                            PROCESSED.labels("failed").inc()
                            db.execute("UPDATE videos SET status='failed', error=%s, updated_at=now() "
                                       "WHERE video_id=%s", (str(e)[:500], fields.get("video_id")))
                    r.xack(STREAM, GROUP, msg_id)
                for g in r.xinfo_groups(STREAM):
                    if g["name"] == GROUP:
                        QUEUE_LAG.labels(STREAM).set((g.get("lag") or 0) + (g.get("pending") or 0))
        except redis.ConnectionError as e:
            log.warning("redis unavailable (%s); retrying", e)
            time.sleep(2)
        except Exception:
            log.exception("consumer loop crashed; restarting")
            time.sleep(2)


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=consume_forever, daemon=True, name="consumer").start()
    yield
    stop.set()


app = FastAPI(title="video-worker", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if not state["ready"]:
        raise HTTPException(503, "loading")
    return {"status": "ready", "whisper_model": WHISPER_MODEL}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
