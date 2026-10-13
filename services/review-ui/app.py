"""Human review UI: moderators clear the REVIEW queue, see the decision mix, and check censored videos."""
import html
import json
import os
import urllib.request

import pandas as pd
import psycopg
import streamlit as st

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://moderation:moderation@localhost:5432/moderation")
INGEST_URL = os.environ.get("INGEST_URL", "http://ingest:8000").rstrip("/")

st.set_page_config(page_title="Moderation review", layout="wide")


@st.cache_resource
def connect():
    return psycopg.connect(DATABASE_URL, autocommit=True)


def query(sql, params=None):
    conn = connect()
    if conn.closed:
        st.cache_resource.clear()
        conn = connect()
    with conn.cursor() as cur:
        cur.execute(sql, params)
        if cur.description is None:
            return None
        return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])


@st.cache_data(ttl=600, show_spinner=False)
def explain(text):
    """Per-word contributions from the text worker (via the ingest API); None if unavailable."""
    req = urllib.request.Request(f"{INGEST_URL}/explain", data=json.dumps({"text": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.load(resp)
    except Exception:
        return None


def highlighted(words):
    """Words as HTML, background darker the more the word pushes the score up."""
    top = max([0.15] + [w["weight"] for w in words])
    parts = []
    for w in words:
        a = min(max(w["weight"], 0) / top, 1) * 0.6 if w["weight"] > 0.01 else 0
        parts.append(f'<span title="{w["weight"]:+.3f}" style="background:rgba(214,60,80,{a:.2f});'
                     f'padding:1px 3px;border-radius:4px">{html.escape(w["word"])}</span>')
    return '<div style="font-size:1.1rem;line-height:2">' + " ".join(parts) + "</div>"


def set_verdict(post_id, verdict):
    query("UPDATE decisions SET review_status=%s, reviewed_at=now() WHERE post_id=%s", (verdict, post_id))


st.title("Moderation review")
queue_tab, overview_tab, video_tab = st.tabs(["Review queue", "Overview", "Videos"])

with queue_tab:
    pending = query("SELECT post_id, text, score, ts, flags, normalized, categories FROM decisions "
                    "WHERE decision='REVIEW' AND review_status IS NULL ORDER BY ts LIMIT 25")
    st.caption(f"{len(pending)} post(s) waiting, oldest first. "
               "These scored between the two thresholds, so a person decides.")
    if pending.empty:
        st.success("Queue is empty.")
    for row in pending.itertuples():
        with st.container(border=True):
            ex = explain(row.text) if row.text else None
            if ex and ex.get("words"):
                st.markdown(highlighted(ex["words"]), unsafe_allow_html=True)
            else:
                st.write(row.text or "(no text)")
            notes = []
            if row.flags:
                cats = row.categories or {}
                notes.append("**flagged:** " + ", ".join(
                    f"{f.replace('_', ' ')} {cats.get(f, 0):.2f}" for f in row.flags.split(",")))
            if row.normalized:
                notes.append(f"**read as:** _{row.normalized}_")
            if notes:
                st.markdown(" &nbsp; ".join(notes))
            c1, c2, c3 = st.columns([1, 1, 6])
            c3.caption(f"score {row.score:.3f} · {row.ts:%Y-%m-%d %H:%M:%S} · "
                       "red = words that push the score up")
            if c1.button("Allow", key=f"a{row.post_id}", type="primary"):
                set_verdict(row.post_id, "ALLOW")
                st.rerun()
            if c2.button("Block", key=f"b{row.post_id}"):
                set_verdict(row.post_id, "BLOCK")
                st.rerun()

with overview_tab:
    counts = query("SELECT decision, count(*) AS posts FROM decisions "
                   "WHERE ts > now() - interval '24 hours' GROUP BY decision")
    total = int(counts["posts"].sum()) if not counts.empty else 0
    by = dict(zip(counts["decision"], counts["posts"])) if total else {}
    cols = st.columns(4)
    cols[0].metric("Posts (24 h)", total)
    for col, d in zip(cols[1:], ["ALLOW", "REVIEW", "BLOCK"]):
        n = int(by.get(d, 0))
        col.metric(d, n, f"{n / total:.0%} of posts" if total else None, delta_color="off")
    reviewed = query("SELECT review_status, count(*) AS posts FROM decisions "
                     "WHERE review_status IS NOT NULL GROUP BY review_status")
    if not reviewed.empty:
        st.caption("Human verdicts on REVIEW posts: " +
                   ", ".join(f"{r.review_status} {r.posts}" for r in reviewed.itertuples()))
    st.subheader("Latest decisions")
    st.dataframe(query("SELECT ts, decision, round(score::numeric, 3) AS score, flags, model_version, "
                       "round(latency_ms::numeric, 2) AS latency_ms, review_status, text "
                       "FROM decisions ORDER BY ts DESC LIMIT 100"),
                 use_container_width=True, hide_index=True)

with video_tab:
    videos = query("SELECT video_id, filename, status, n_spans, processing_s, created_at "
                   "FROM videos ORDER BY created_at DESC LIMIT 50")
    if videos.empty:
        st.info("No videos yet. Upload one on the main page.")
    else:
        st.dataframe(videos, use_container_width=True, hide_index=True)
        choice = st.selectbox("Open a video", videos["video_id"],
                              format_func=lambda v: f"{videos.set_index('video_id').loc[v, 'filename']} ({v[:8]})")
        try:
            with urllib.request.urlopen(f"{INGEST_URL}/videos/{choice}", timeout=10) as resp:
                v = json.load(resp)
        except Exception as e:
            st.error(f"Could not load the video details from the ingest API: {e}")
        else:
            left, right = st.columns(2)
            left.caption("Original")
            left.video(v["original_url"])
            right.caption("Censored")
            if v.get("clean_url"):
                right.video(v["clean_url"])
            else:
                right.info(f"Status: {v['status']}")
            if v["spans"]:
                st.dataframe(pd.DataFrame(v["spans"]), use_container_width=True, hide_index=True)
