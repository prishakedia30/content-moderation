import io

import fakeredis
import pytest
from fastapi.testclient import TestClient

import app as ingest


class FakeDB:
    def __init__(self):
        self.calls = []

    def execute(self, sql, params=None, fetch=False):
        self.calls.append((sql, params))
        return [] if fetch else None


@pytest.fixture
def client(monkeypatch):
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(ingest, "r", fake)
    ingest.db_holder["db"] = FakeDB()
    with TestClient(ingest.app) as c:
        yield c, fake


def test_health_ready(client):
    c, _ = client
    assert c.get("/health").status_code == 200
    assert c.get("/ready").json()["status"] == "ready"


def test_post_is_queued(client):
    c, fake = client
    resp = c.post("/posts", json={"text": "hello world"})
    assert resp.status_code == 202
    (msg_id, fields), = fake.xrange("posts.raw")
    assert fields["post_id"] == resp.json()["post_id"] and fields["text"] == "hello world"


def test_empty_post_rejected(client):
    c, _ = client
    assert c.post("/posts", json={"text": ""}).status_code == 422


def test_unknown_post_is_pending(client):
    c, _ = client
    assert c.get("/posts/abc").json()["status"] == "pending"


def test_video_upload_queues_and_stores(client):
    c, fake = client
    resp = c.post("/videos", files={"file": ("clip.mp4", io.BytesIO(b"fake-bytes"), "video/mp4")})
    assert resp.status_code == 202
    (_, fields), = fake.xrange("videos.raw")
    assert fields["s3_key"].startswith("videos-raw/") and fields["s3_key"].endswith(".mp4")
    assert ingest.storage.exists(fields["s3_key"])


def test_video_wrong_type_rejected(client):
    c, _ = client
    assert c.post("/videos", files={"file": ("x.txt", io.BytesIO(b"x"), "text/plain")}).status_code == 400


def test_index_page(client):
    c, _ = client
    assert "moderation console" in c.get("/").text.lower()


def test_proxies_forward_to_text_worker(client, monkeypatch):
    c, _ = client
    calls = []

    async def fake_worker(method, path, **kw):
        calls.append((method, path, kw.get("json")))
        return {"ok": path}
    monkeypatch.setattr(ingest, "worker", fake_worker)
    assert c.post("/classify", json={"text": "hi"}).json() == {"ok": "/classify"}
    assert c.post("/explain", json={"text": "hi", "normalize": False}).json() == {"ok": "/explain"}
    assert c.post("/classify/batch", json={"texts": ["a", "b"]}).json() == {"ok": "/classify/batch"}
    assert c.get("/config").json() == {"ok": "/ready"}
    assert calls[0][2] == {"text": "hi"} and calls[1][2] == {"text": "hi", "normalize": False}


def test_stats(client):
    c, _ = client

    class StatsDB(FakeDB):
        def execute(self, sql, params=None, fetch=False):
            if "GROUP BY decision" in sql:
                return [{"decision": "ALLOW", "n": 7}, {"decision": "BLOCK", "n": 3}]
            return [{"n": 2}]
    ingest.db_holder["db"] = StatsDB()
    out = c.get("/stats").json()
    assert out["total"] == 10 and out["ALLOW"] == 7 and out["REVIEW"] == 0 and out["review_pending"] == 2
