// Spike test: ramps to 40 virtual users hitting the public API, holds, ramps down.
// Watch Grafana "Content moderation" or `kubectl get hpa -w` to see text-worker pods scale out.
import http from 'k6/http';
import { check, sleep } from 'k6';

const TARGET = __ENV.TARGET || 'http://ingest:8000';
export const options = {
  stages: [
    { duration: '30s', target: 5 },
    { duration: '1m', target: 40 },
    { duration: '3m', target: 40 },
    { duration: '30s', target: 0 },
  ],
  thresholds: {
    http_req_failed: ['rate<0.02'],
    'http_req_duration{endpoint:classify}': ['p(95)<500'],
  },
};

const TEXTS = [
  'Thanks for sharing, this was really helpful!', 'You are an idiot and nobody wants you here.',
  'I disagree with your point about the budget.', 'Shut up, you worthless piece of garbage.',
  'Great game last night, what a finish.', 'Can someone explain how this API works?',
  'This is the dumbest thing I have ever read.', 'Happy birthday! Hope you have a lovely day.',
];
const json = { headers: { 'Content-Type': 'application/json' } };

export default function () {
  const text = `${TEXTS[Math.floor(Math.random() * TEXTS.length)]} #${Math.floor(Math.random() * 1e6)}`;
  const r = http.post(`${TARGET}/classify`, JSON.stringify({ text }), { ...json, tags: { endpoint: 'classify' } });
  check(r, { 'classify 200': (x) => x.status === 200 });
  if (Math.random() < 0.1) {   // 10% also go through the queue
    http.post(`${TARGET}/posts`, JSON.stringify({ text }), { ...json, tags: { endpoint: 'posts' } });
  }
  sleep(0.05);
}
