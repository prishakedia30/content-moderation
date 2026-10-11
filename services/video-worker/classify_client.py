"""Call the shared POST /classify endpoint: {text} -> {score, decision, model_version, latency_ms}."""
import json, time, urllib.error, urllib.request

def make_scorer(url="http://localhost:8000/classify", timeout=5.0):
    cache = {}
    def score(text):
        if text in cache:
            return cache[text]
        req = urllib.request.Request(
            url, data=json.dumps({"text": text, "source": "video"}).encode(),
            headers={"Content-Type": "application/json"})
        for attempt in range(3):            # the text worker may be scaling or restarting
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    s = float(json.load(r)["score"])
                break
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)
        cache[text] = s
        return s
    return score
