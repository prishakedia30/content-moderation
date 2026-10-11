"""ingest: the public front door.
- POST /posts        -> queue a post on Redis stream posts.raw (async moderation), returns post_id
- GET  /posts/{id}   -> the decision once the text worker has written it
- POST /classify     -> synchronous score via the text worker (demo + load tests)
- POST /videos       -> upload a video to storage, queue it on videos.raw
- GET  /videos/{id}  -> status, toxic spans, links to the original and censored files
- GET  /             -> small demo page"""
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import redis
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from common.storage import Storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ingest")

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CLASSIFY_URL = os.environ.get("CLASSIFY_URL", "http://text-worker:8000/classify")
MAX_VIDEO_MB = int(os.environ.get("MAX_VIDEO_MB", "100"))
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}

REQS = Counter("ingest_requests_total", "Requests to the ingest API", ["endpoint", "status"])
REQ_S = Histogram("ingest_request_seconds", "Ingest request duration", ["endpoint"],
                  buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 10))

from redis.backoff import ExponentialBackoff
from redis.retry import Retry

# Retry briefly on connection errors, so a Redis restart costs a short pause instead of failed posts.
r = redis.Redis.from_url(REDIS_URL, decode_responses=True,
                         retry=Retry(ExponentialBackoff(cap=2), 5),
                         retry_on_error=[redis.ConnectionError, redis.TimeoutError])
storage = Storage()
db_holder = {"db": None}


def get_db():
    if db_holder["db"] is None:
        raise HTTPException(503, "database not ready yet")
    return db_holder["db"]


def init_db_forever():
    from common.db import DB
    while db_holder["db"] is None:
        try:
            db = DB().connect()
            db.ensure_schema()
            db_holder["db"] = db
            log.info("database ready")
        except Exception as e:
            log.warning("database init failed (%s); retrying", e)
            time.sleep(3)


@asynccontextmanager
async def lifespan(_app):
    if os.environ.get("DB_INIT", "1") == "1":
        threading.Thread(target=init_db_forever, daemon=True).start()
    # No keep-alive: each call opens a new connection, so the Kubernetes Service spreads
    # load across ALL text-worker pods (otherwise new pods added by the HPA sit idle).
    app.state.http = httpx.AsyncClient(timeout=10, limits=httpx.Limits(max_keepalive_connections=0))
    yield
    await app.state.http.aclose()


app = FastAPI(title="Moderation ingest API", lifespan=lifespan)


@app.middleware("http")
async def count_requests(request: Request, call_next):
    t0 = time.perf_counter()
    route = request.url.path.split("/")[1] or "index"
    response = await call_next(request)
    if route not in ("metrics", "health", "ready"):
        REQS.labels(route, response.status_code).inc()
        REQ_S.labels(route).observe(time.perf_counter() - t0)
    return response


class PostIn(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    user_id: str = Field(default="anon", max_length=100)


class TextIn(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    try:
        r.ping()
    except redis.RedisError:
        raise HTTPException(503, "redis unreachable")
    return {"status": "ready", "database": db_holder["db"] is not None}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/posts", status_code=202)
def create_post(post: PostIn):
    post_id = uuid.uuid4().hex
    r.xadd("posts.raw", {"post_id": post_id, "user_id": post.user_id, "text": post.text,
                         "ts": str(time.time())}, maxlen=100_000, approximate=True)
    return {"post_id": post_id, "status": "queued"}


@app.get("/posts/{post_id}")
def get_post(post_id: str):
    rows = get_db().execute(
        "SELECT post_id, score, decision, model_version, latency_ms, ts, review_status "
        "FROM decisions WHERE post_id = %s", (post_id,), fetch=True)
    return rows[0] if rows else {"post_id": post_id, "status": "pending"}


@app.get("/posts")
def recent_posts(limit: int = 20):
    return get_db().execute(
        "SELECT post_id, text, score, decision, ts, review_status FROM decisions "
        "ORDER BY ts DESC LIMIT %s", (min(limit, 200),), fetch=True)


@app.post("/classify")
async def classify(body: TextIn):
    try:
        resp = await app.state.http.post(CLASSIFY_URL, json={"text": body.text})
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(503, f"text worker unavailable: {e.__class__.__name__}")
    return resp.json()


@app.post("/videos", status_code=202)
async def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXT:
        raise HTTPException(400, f"unsupported file type {ext!r}; use one of {sorted(VIDEO_EXT)}")
    file.file.seek(0, 2)
    size_mb = file.file.tell() / 1e6
    file.file.seek(0)
    if size_mb > MAX_VIDEO_MB:
        raise HTTPException(413, f"video is {size_mb:.0f} MB; limit is {MAX_VIDEO_MB} MB")
    video_id = uuid.uuid4().hex
    raw_key = f"videos-raw/{video_id}{ext}"
    await run_in_threadpool(storage.put_fileobj, file.file, raw_key)
    await run_in_threadpool(
        get_db().execute,
        "INSERT INTO videos (video_id, filename, raw_key) VALUES (%s, %s, %s)",
        (video_id, file.filename, raw_key))
    r.xadd("videos.raw", {"video_id": video_id, "s3_key": raw_key, "ts": str(time.time())},
           maxlen=10_000, approximate=True)
    return {"video_id": video_id, "status": "queued"}


@app.get("/videos/{video_id}")
def get_video(video_id: str):
    db = get_db()
    rows = db.execute("SELECT * FROM videos WHERE video_id = %s", (video_id,), fetch=True)
    if not rows:
        raise HTTPException(404, "unknown video")
    video = rows[0]
    video["spans"] = db.execute(
        "SELECT start_s, end_s, text, score FROM video_spans WHERE video_id = %s ORDER BY start_s",
        (video_id,), fetch=True)
    video["original_url"] = storage.url(video["raw_key"])
    video["clean_url"] = storage.url(video["clean_key"]) if video["clean_key"] else None
    return video


@app.get("/videos")
def recent_videos(limit: int = 20):
    return get_db().execute(
        "SELECT video_id, filename, status, n_spans, processing_s, created_at FROM videos "
        "ORDER BY created_at DESC LIMIT %s", (min(limit, 200),), fetch=True)


@app.get("/files/{key:path}")
def local_file(key: str):
    """Only used with STORAGE=local (docker-compose); in the cloud, links are presigned S3 URLs."""
    if storage.mode != "local":
        raise HTTPException(404)
    try:
        path = storage.local_path(key)
    except ValueError:
        raise HTTPException(400, "bad path")
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).with_name("index.html")).read_text()
