# Real-time content moderation platform (Team SAP)

A post is accepted instantly, queued, scored by an INT8 ONNX DistilBERT model and logged; only
borderline scores reach a human. Videos get their toxic speech muted by the same model.
Runs on AWS (Terraform, k3s Kubernetes, Helm), with CI/CD, autoscaling and monitoring.

```
                         ┌────────────── k3s on EC2 (Terraform) ───────────────────────────┐
browser / k6 ──► Traefik │  ingest API ──► Redis streams ──► text-worker (ONNX INT8) ──► Postgres
   /  /review  /grafana  │      │   posts.raw / videos.raw        ▲  /classify           ▲   │
                         │      │                                 │                      │   │
                         │      └──► S3 videos-raw ──► video-worker (FFmpeg, Whisper, mute)  │
                         │                              └──► S3 videos-clean                 │
                         │  review UI (Streamlit) ◄── Postgres     Prometheus ──► Grafana    │
                         └────────────────────────────────────────────────────────────────────┘
GitHub Actions: test → build → Trivy scan → push to GHCR → helm upgrade --atomic
```

| Path | What |
|---|---|
| `services/ingest` | Public API + demo page: `POST /posts` (queued), `POST /classify` (instant), `POST /videos` |
| `services/text-worker` | ONNX classifier, ALLOW/REVIEW/BLOCK policy, Redis consumer with crash recovery |
| `services/video-worker` | FFmpeg → faster-whisper → toxic span finder → mute (Sanvi's pipeline, as a service) |
| `services/review-ui` | Streamlit queue for human moderators, decision overview, video before/after |
| `services/common` | Shared storage (S3 / local), database helper, schema |
| `ml/` | Baseline, fine-tuning + calibration, ONNX export, evaluations ([ml/README.md](ml/README.md)) |
| `infra/terraform` | S3, IAM role, security group, EC2 + k3s |
| `deploy/helm/moderation` | All services, Postgres, Redis, autoscaling, ingress, ServiceMonitors, Grafana dashboard |
| `.github/workflows/ci.yml` | CI/CD pipeline |
| `scripts/` | `deploy.sh`, `loadtest.sh`, `demo-rollback.sh`, `chaos.sh` |

**Deploying and demoing: see [RUNBOOK.md](RUNBOOK.md).**

Run everything on a laptop (needs Docker Desktop): `docker compose up --build`, then open
http://localhost:8000 and http://localhost:8501/review. Tests: `python -m pytest services/<name>/tests`.
