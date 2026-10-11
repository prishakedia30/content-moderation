# Runbook: deploy, demo, troubleshoot

Two terminal tabs, as before:
- **Tab 1, tunnel** (leave it running):
  `cd ~/moderation/infra/terraform && while true; do ssh -i ~/.ssh/moderation -N -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -L 6443:localhost:6443 ubuntu@$(terraform output -raw public_ip); sleep 2; done`
- **Tab 2, work**: everything below, run from `~/moderation`.

Paste each grey block exactly as it is.

---

## 1. Get the code onto your Mac and push it (5 min)

The new code arrives as `full-build.bundle` (sent in chat). Save it to **Downloads**, then:

```bash
cd ~/moderation
git pull
git fetch ~/Downloads/full-build.bundle full-build:full-build
git merge --ff-only full-build
git push
```

`git push` starts GitHub Actions (step 3).

## 2. Let pods use the server's AWS role (2 min)

Pods need one extra network hop to reach the server's credentials (for S3). Terraform changes a
setting on the running server; **it is not rebuilt and its IP stays the same**.

```bash
cd ~/moderation/infra/terraform
terraform plan
```

Check the plan says `1 to change` (`aws_instance.k3s` "updated in-place", `metadata_options`) and
**`0 to destroy`**. If it says anything will be destroyed or replaced, stop and ask. Otherwise:

```bash
terraform apply
cd ~/moderation
```

## 3. Wait for GitHub Actions to build the images (10–15 min the first time)

GitHub → your repo → **Actions** tab → the newest "CI/CD" run. Wait until all jobs are green:
`test` → `build (ingest)`, `build (text-worker)`, `build (video-worker)`, `build (review-ui)`.
`deploy` shows as skipped; that's expected.

Open a `build` job → **Security scan (Trivy)** step: that table is your security-scan screenshot.

## 4. Make the images public (2 min, once)

GitHub → your profile picture → **Your profile** → **Packages** tab. For each of the 4 packages
(`content-moderation/ingest`, `/text-worker`, `/video-worker`, `/review-ui`):
open it → **Package settings** (right side) → **Change visibility** → **Public** → type the name to confirm.

(Private images would need a pull secret on the cluster; public is fine for a student project.)

## 5. Deploy (5 min)

```bash
cd ~/moderation
./scripts/deploy.sh
```

This deploys the commit you just pushed. Helm waits for every pod to become ready; if they don't
within 8 minutes, it **rolls back automatically**. In another tab you can watch:

```bash
kubectl get pods -w
```

The first start is slower: the video-worker image is about 1.5 GB to download, and the text worker
fetches the model from S3. When it finishes, `kubectl get pods` shows everything `1/1 Running`.

## 6. Check it works

| Open | You should see |
|---|---|
| `http://<server-ip>/` | Demo page. Click **Score instantly** → a BLOCK pill. **Submit through the queue** → the same decision, written by the queue worker |
| `http://<server-ip>/review/` | Review UI: Overview tab shows counts; posts scoring between the thresholds appear in Review queue |
| `http://<server-ip>/grafana/` → Dashboards → **Content moderation** | Live decisions/s, latency p50/p95/p99, pods, queue |
| `http://<server-ip>/docs` | Auto-generated API documentation |

Get the IP with `cd infra/terraform && terraform output -raw public_ip && cd ../..`.

**Video:** record a 10–20 second clip on your phone where you say a couple of normal sentences and
one rude word, AirDrop it to the Mac, upload it on the demo page. Keep it under ~15 MB (uploads to
Sydney are slow). Within a minute you get the original and the censored version side by side.

## 7. Demo: autoscaling under load (6 min)

```bash
./scripts/loadtest.sh
```

In another tab: `kubectl get hpa -w`. Within 1–2 minutes the text-worker HPA goes from 1 to more
replicas. In Grafana, the **Autoscaling** and **CPU per pod** panels show it. k6 ends with a summary
(requests/s, p95 latency, failure rate): screenshot it.

## 8. Demo: automatic rollback (2 min)

```bash
./scripts/demo-rollback.sh
```

Deploys an image tag that doesn't exist; pods can't start; after 90 s Helm rolls back on its own.
`helm history moderation` shows the failed revision and the rollback. The site stays up throughout.

## 9. Demo: resilience (3 min)

```bash
./scripts/chaos.sh
```

Kills a text-worker pod and the Redis pod while posts are flowing. Kubernetes restarts them, Redis
reloads its queue from disk, and unfinished messages are re-claimed by another worker after 60 s.
The final count should be **70**.

## 10. Switch to your own fine-tuned model (after Kaggle)

Follow [ml/README.md](ml/README.md); it ends with the model uploaded to S3 as `v1`. Then:

```bash
sed -i '' 's/MODEL_VERSION: v0/MODEL_VERSION: v1/' deploy/helm/moderation/values.yaml
git commit -am "Use fine-tuned model v1"
git push
```

Wait for GitHub Actions to go green, then `./scripts/deploy.sh`. The demo page shows `model v1`.

## 11. Optional: deploy automatically from GitHub Actions

Off by default, because it needs SSH open to GitHub's servers (key-only login, but still wider).
1. `infra/terraform/terraform.tfvars`: add the line `extra_ssh_cidrs = ["0.0.0.0/0"]`, then `terraform apply`.
2. GitHub → repo → Settings → Secrets and variables → Actions:
   - Secrets: `SERVER_IP` (the IP), `SSH_PRIVATE_KEY` (`cat ~/.ssh/moderation`),
     `KUBECONFIG_B64` (`base64 -i ~/.kube/moderation.yaml`), `PG_PASSWORD` (`cat ~/.moderation-pg-pass`)
   - Variables tab: `AUTO_DEPLOY` = `true`
3. The next push to main runs test → build → scan → **deploy**.

## Screenshots for the reports

- [ ] Terraform `apply` output · AWS console EC2 + S3 bucket (`models/`, `videos-raw/`, `videos-clean/`)
- [ ] `kubectl get pods -A` · `kubectl get hpa` during the load test
- [ ] GitHub Actions run (all green) + Trivy table
- [ ] Grafana "Content moderation" dashboard during the load test
- [ ] k6 summary · `helm history moderation` after the rollback demo · chaos script output
- [ ] Demo page with a BLOCK decision · review UI queue · video before/after with the muted span
- [ ] ML: baseline vs DistilBERT metrics, curves, reliability diagram, ONNX latency table, INT8 vs FP32 table

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ImagePullBackOff` | Images still private, or CI hasn't finished | Step 4; check Actions is green; rerun deploy |
| text-worker restarting, logs say `NoCredentialsError` / `Unable to locate credentials` | Step 2 not applied | Run step 2, then `kubectl rollout restart deploy/text-worker` |
| `kubectl` says `connection refused` on `127.0.0.1:6443` | Tunnel tab died | Restart the tab 1 command |
| `ssh: Operation timed out` | Your Wi-Fi/IP changed | `cd infra/terraform && sed -i '' "s|^my_ip.*|my_ip = \"$(curl -s https://checkip.amazonaws.com)/32\"|" terraform.tfvars && terraform apply` |
| `helm upgrade` "another operation in progress" | A deploy was interrupted | `helm rollback moderation` then deploy again |
| video stuck on `queued` | video-worker still loading | `kubectl logs deploy/video-worker` |
| Grafana dashboard missing | Sidecar hasn't picked it up | Wait a minute; or Dashboards → search "Content moderation" |

Look inside any pod: `kubectl logs deploy/<name> --tail 50` (names: ingest, text-worker, video-worker, review-ui).

## After submission

Stop paying: `cd infra/terraform && terraform destroy` (deletes everything, including the bucket).
Or just stop the server from the EC2 console (keeps the disk; the IP changes on restart).
