"""Call the shared POST /classify endpoint: {text} -> {score, decision, model_version, latency_ms}."""
import json, urllib.request

def make_scorer(url="http://localhost:8000/classify", timeout=5.0):
    cache = {}
    def score(text):
        if text in cache:
            return cache[text]
        req = urllib.request.Request(
            url, data=json.dumps({"text": text}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            s = float(json.load(r)["score"])
        cache[text] = s
        return s
    return score
