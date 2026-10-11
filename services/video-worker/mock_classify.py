"""Stand-in for the real /classify so the video lane can be built and tested alone.
Scores by a few keywords. Run: python3 mock_classify.py 8000"""
import json, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer

BAD = {"idiot", "stupid", "hate", "kill", "shut", "ugly", "loser", "trash"}

def score(text):
    words = [w.strip(".,!?").lower() for w in text.split()]
    hits = sum(w in BAD for w in words)
    return min(0.05 + 0.8 * hits, 0.99) if words else 0.0

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        t0 = time.time()
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        s = score(body.get("text", ""))
        dec = "BLOCK" if s >= 0.8 else "REVIEW" if s >= 0.4 else "ALLOW"
        out = json.dumps({"score": s, "decision": dec, "model_version": "mock",
                          "latency_ms": (time.time() - t0) * 1000}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
    def log_message(self, *a): pass

if __name__ == "__main__":
    HTTPServer(("0.0.0.0", int(sys.argv[1]) if len(sys.argv) > 1 else 8000), H).serve_forever()
