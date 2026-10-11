-- Schema for the moderation platform. Runs once, when the Postgres volume is first created.
-- Columns in the first block of each table are the agreed interface (roadmap, Integration plan).
-- Columns marked "extra" are additions for the review UI; they are nullable so writers can ignore them.

CREATE TABLE IF NOT EXISTS decisions (
    post_id        TEXT        PRIMARY KEY,
    score          REAL        NOT NULL CHECK (score BETWEEN 0 AND 1),
    decision       TEXT        NOT NULL CHECK (decision IN ('ALLOW', 'REVIEW', 'BLOCK')),
    model_version  TEXT        NOT NULL,
    latency_ms     REAL        NOT NULL,
    ts             TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- extra: lets a reviewer read the post, and records their verdict
    text           TEXT,
    review_status  TEXT        CHECK (review_status IN ('ALLOW', 'BLOCK')),
    reviewed_at    TIMESTAMPTZ
);

-- The review queue: REVIEW-band posts nobody has judged yet, oldest first.
CREATE INDEX IF NOT EXISTS decisions_review_queue
    ON decisions (ts) WHERE decision = 'REVIEW' AND review_status IS NULL;

CREATE TABLE IF NOT EXISTS video_spans (
    id        BIGSERIAL PRIMARY KEY,
    video_id  TEXT             NOT NULL,
    start_s   DOUBLE PRECISION NOT NULL,
    end_s     DOUBLE PRECISION NOT NULL,
    text      TEXT             NOT NULL,
    score     REAL             NOT NULL CHECK (score BETWEEN 0 AND 1),
    CHECK (end_s > start_s)
);

CREATE INDEX IF NOT EXISTS video_spans_video ON video_spans (video_id);