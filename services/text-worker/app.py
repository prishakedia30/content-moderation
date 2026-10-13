"""text-worker: scores posts with the ONNX model.
- POST /classify, /classify/batch: synchronous scoring (used by the ingest API, video worker, load tests)
- background consumer: reads the Redis stream posts.raw, writes a row to `decisions`, emits posts.decided
- /health, /ready, /metrics for Kubernetes and Prometheus"""
import json
import logging
import os
import socket
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # makes `common` importable

import redis
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from classifier import MockClassifier, OnnxClassifier, explain
from decision import decide, flags, load_thresholds
from normalize import Normalizer, vocab_from_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("text-worker")

MODEL_VERSION = os.environ.get("MODEL_VERSION", "v0")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models/current"))
MODEL_FILE = os.environ.get("MODEL_FILE", "model.int8.onnx")
MODEL_MODE = os.environ.get("MODEL_MODE", "onnx")          # onnx | mock
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CONSUMER_ENABLED = os.environ.get("CONSUMER_ENABLED", "1") == "1"
NORMALIZE = os.environ.get("NORMALIZE", "1") == "1"   # default for callers that don't say
STREAM_IN, STREAM_OUT, GROUP = "posts.raw", "posts.decided", "text-workers"

DECISIONS = Counter("moderation_decisions_total", "Moderation decisions", ["decision", "source"])
INFER_S = Histogram("moderation_inference_seconds", "Model scoring time per text",
                    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1))
QUEUE_LAG = Gauge("moderation_queue_lag", "Messages waiting in a stream", ["stream"])
ERRORS = Counter("moderation_errors_total", "Errors while processing", ["where"])

state = {"clf": None, "cfg": None, "version": MODEL_VERSION, "norm": Normalizer()}
stop = threading.Event()


def fetch_model():
    """Use MODEL_DIR if it already has the model (docker-compose mount), else pull it from S3."""
    if (MODEL_DIR / MODEL_FILE).exists():
        return
    from common.storage import Storage
    prefix = f"models/{MODEL_VERSION}/"
    log.info("downloading model from storage prefix %s", prefix)
    skip = ("model.onnx",) if MODEL_FILE != "model.onnx" else ()
    Storage().download_prefix(prefix, MODEL_DIR, skip=skip)


def load_model():
    if MODEL_MODE == "mock":
        state["cfg"] = load_thresholds(None)
        state["clf"] = MockClassifier()
        state["version"] = "mock"
    else:
        fetch_model()
        state["cfg"] = load_thresholds(MODEL_DIR)
        cfg = state["cfg"]
        state["clf"] = OnnxClassifier(MODEL_DIR, MODEL_FILE, cfg["temperature"],
                                      temperatures=cfg["temperatures"], labels=cfg["labels"])
        state["norm"] = Normalizer(vocab_from_tokenizer(state["clf"].tok))
    log.info("model %s ready, thresholds %s", state["version"], state["cfg"])


def score_text(text, normalize):
    """Score the text as written and, if normalizing changes it, the normalized version too;
    keep whichever is more toxic. So normalization can only catch more, never hide a post."""
    norm = state["norm"](text) if normalize else text
    changed = norm != " ".join(text.split())
    if not changed:
        main, cats = state["clf"].score_full(text)
        return main, cats, None
    (s1, c1), (s2, c2) = state["clf"].score_many([text, norm])
    return (s2, c2, norm) if s2 >= s1 else (s1, c1, norm)


def classify(text, source, normalize=None):
    normalize = NORMALIZE if normalize is None else normalize
    t0 = time.perf_counter()
    score, cats, normalized = score_text(text, normalize)
    dt = time.perf_counter() - t0
    INFER_S.observe(dt)
    cfg = state["cfg"]
    decision = decide(score, cfg["t_low"], cfg["t_high"])
    DECISIONS.labels(decision, source).inc()
    return {"score": round(score, 4), "decision": decision,
            "categories": cats, "flags": flags(cats, cfg["categories"]),
            "normalized": normalized,
            "model_version": state["version"], "latency_ms": round(dt * 1000, 2)}


# ---------- queue consumer ----------

def ensure_group(r):
    try:
        r.xgroup_create(STREAM_IN, GROUP, id="0", mkstream=True)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


def handle(r, db, fields):
    post_id, text = fields.get("post_id"), fields.get("text", "")
    if not post_id:
        return
    out = classify(text, "stream")
    db.execute(
        """INSERT INTO decisions (post_id, score, decision, model_version, latency_ms, text,
                                 categories, flags, normalized)
           VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s) ON CONFLICT (post_id) DO NOTHING""",
        (post_id, out["score"], out["decision"], out["model_version"], out["latency_ms"], text,
         json.dumps(out["categories"]) if out["categories"] else None,
         ",".join(out["flags"]) or None, out["normalized"]))
    r.xadd(STREAM_OUT, {"post_id": post_id, "decision": out["decision"], "score": out["score"]},
           maxlen=10000, approximate=True)


def update_lag(r):
    try:
        for g in r.xinfo_groups(STREAM_IN):
            if g["name"] == GROUP:
                QUEUE_LAG.labels(STREAM_IN).set((g.get("lag") or 0) + (g.get("pending") or 0))
    except redis.ResponseError:
        pass


def consume_forever():
    from common.db import DB
    consumer = socket.gethostname()
    db = None
    last_claim = 0.0
    while not stop.is_set():
        try:
            r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            ensure_group(r)
            if db is None:
                db = DB().connect()
                db.ensure_schema()
            log.info("consuming %s as %s/%s", STREAM_IN, GROUP, consumer)
            while not stop.is_set():
                batches = []
                if time.time() - last_claim > 30:
                    # Take over messages a crashed pod read but never acknowledged.
                    claimed = r.xautoclaim(STREAM_IN, GROUP, consumer, min_idle_time=60_000,
                                           start_id="0-0", count=50)[1]
                    if claimed:
                        log.info("reclaimed %d stuck messages", len(claimed))
                        batches.append(claimed)
                    last_claim = time.time()
                for _, msgs in r.xreadgroup(GROUP, consumer, {STREAM_IN: ">"}, count=16, block=2000) or []:
                    batches.append(msgs)
                for msgs in batches:
                    for msg_id, fields in msgs:
                        if fields:
                            try:
                                handle(r, db, fields)
                            except Exception:
                                ERRORS.labels("handle").inc()
                                log.exception("failed on message %s", msg_id)
                                continue   # left pending; reclaimed and retried later
                        r.xack(STREAM_IN, GROUP, msg_id)
                update_lag(r)
        except redis.ConnectionError as e:
            ERRORS.labels("redis").inc()
            log.warning("redis unavailable (%s); retrying", e)
            time.sleep(2)
        except Exception:
            ERRORS.labels("consumer").inc()
            log.exception("consumer crashed; restarting loop")
            time.sleep(2)


# ---------- HTTP API ----------

@asynccontextmanager
async def lifespan(_app):
    load_model()
    if CONSUMER_ENABLED:
        threading.Thread(target=consume_forever, daemon=True, name="consumer").start()
    yield
    stop.set()


app = FastAPI(title="text-worker", lifespan=lifespan)


class TextIn(BaseModel):
    text: str = Field(max_length=10_000)
    source: Literal["api", "video"] = "api"   # video span checks are counted separately
    normalize: bool | None = None             # None = server default (on)


class BatchIn(BaseModel):
    texts: list[str] = Field(max_length=256)
    normalize: bool | None = None


class ExplainIn(BaseModel):
    text: str = Field(max_length=5_000)
    normalize: bool | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if state["clf"] is None:
        raise HTTPException(503, "model not loaded")
    cfg = state["cfg"]
    return {"status": "ready", "model_version": state["version"],
            "multilabel": bool(getattr(state["clf"], "multilabel", False)),
            "labels": list(getattr(state["clf"], "labels", []) or []),
            "normalize_default": NORMALIZE,
            **{k: cfg[k] for k in ("t_low", "t_high", "temperature", "categories")}}


@app.post("/classify")
def classify_one(body: TextIn):
    return classify(body.text, body.source, body.normalize)


@app.post("/classify/batch")
def classify_batch(body: BatchIn):
    return [classify(t, "api", body.normalize) for t in body.texts]


@app.post("/explain")
def explain_one(body: ExplainIn):
    """Word-level reasons for a score (occlusion), on the text the model actually scored."""
    out = classify(body.text, "explain", body.normalize)
    scored = out["normalized"] or body.text
    t0 = time.perf_counter()
    _, words = explain(state["clf"], scored)
    out["words"] = words
    out["explain_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
