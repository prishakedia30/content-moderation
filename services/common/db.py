"""Small Postgres helper shared by the services: retrying connect, idempotent schema."""
import logging
import os
import re
import threading
import time
from pathlib import Path

import psycopg

log = logging.getLogger("db")
SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def schema_statements():
    sql = re.sub(r"--[^\n]*", "", SCHEMA_PATH.read_text())
    return [s.strip() for s in sql.split(";") if s.strip()]


class DB:
    def __init__(self, url=None):
        self.url = url or os.environ["DATABASE_URL"]
        self.conn = None
        self.lock = threading.Lock()

    def connect(self, attempts=60, wait_s=2.0):
        for i in range(attempts):
            try:
                self.conn = psycopg.connect(self.url, autocommit=True, connect_timeout=5)
                return self
            except psycopg.OperationalError as e:
                log.warning("database not ready (%s); retry %d/%d", str(e).strip(), i + 1, attempts)
                time.sleep(wait_s)
        raise RuntimeError("database unreachable")

    def ensure_schema(self):
        for attempt in range(3):
            try:
                for stmt in schema_statements():
                    self.execute(stmt)
                return
            except (psycopg.errors.UniqueViolation, psycopg.errors.DuplicateColumn,
                    psycopg.errors.DuplicateObject, psycopg.errors.InternalError_):
                # Two pods created/altered the same table at the same moment; the other one won.
                time.sleep(1)
        log.warning("schema creation raced 3 times; continuing")

    def execute(self, sql, params=None, fetch=False):
        with self.lock:
            for attempt in range(2):
                try:
                    if self.conn is None or self.conn.closed:
                        self.connect()
                    with self.conn.cursor() as cur:
                        cur.execute(sql, params)
                        if fetch:
                            cols = [c.name for c in cur.description]
                            return [dict(zip(cols, row)) for row in cur.fetchall()]
                        return None
                except psycopg.OperationalError:
                    self.conn = None
                    if attempt:
                        raise
