-- Schema for the moderation platform. Every service runs this on startup, and
-- Postgres runs it once when its volume is first created. All statements are
-- idempotent (IF NOT EXISTS), so running it again is harmless.
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

-- extra (v2): category probabilities from a multi-label model, flagged categories
-- ("threat,insult") and the normalized text when obfuscation was undone. Added with
-- ALTER so databases created before this change get the columns too.
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS categories JSONB;
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS flags      TEXT;
ALTER TABLE decisions ADD COLUMN IF NOT EXISTS normalized TEXT;

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

-- One row per uploaded video: where it is in the pipeline, and where its files are.
CREATE TABLE IF NOT EXISTS videos (
    video_id      TEXT        PRIMARY KEY,
    filename      TEXT,
    status        TEXT        NOT NULL DEFAULT 'queued'
                  CHECK (status IN ('queued', 'processing', 'done', 'failed')),
    raw_key       TEXT        NOT NULL,
    clean_key     TEXT,
    n_words       INTEGER,
    n_spans       INTEGER,
    processing_s  REAL,
    error         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
