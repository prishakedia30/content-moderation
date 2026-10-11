#!/usr/bin/env bash
# Deploy the commit that GitHub Actions built:  ./scripts/deploy.sh
# Deploy a specific image tag:                  TAG=<git sha> ./scripts/deploy.sh
# --atomic: if the new pods never become ready, Helm rolls back to the last working release.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${TAG:-$(git rev-parse HEAD)}"
TIMEOUT="${TIMEOUT:-8m}"
echo "Deploying image tag $TAG"
helm upgrade --install moderation deploy/helm/moderation \
  --atomic --timeout "$TIMEOUT" \
  --set image.tag="$TAG" \
  --set postgres.password="$(cat ~/.moderation-pg-pass)"
kubectl get pods
