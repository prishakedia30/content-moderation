#!/usr/bin/env bash
# Run the k6 spike test inside the cluster, then watch the HPA add text-worker pods:
#   ./scripts/loadtest.sh            (in another tab:  kubectl get hpa -w)
set -euo pipefail
cd "$(dirname "$0")/.."
kubectl create configmap k6-script --from-file=loadtest/spike.js --dry-run=client -o yaml | kubectl apply -f -
kubectl delete job k6-spike --ignore-not-found
kubectl apply -f loadtest/k6-job.yaml
kubectl wait --for=condition=ready pod -l job-name=k6-spike --timeout=120s || true
kubectl logs -f job/k6-spike
