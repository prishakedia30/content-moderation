#!/usr/bin/env bash
# Viva demo: deploy an image tag that doesn't exist, watch Helm roll back on its own.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Release history before:"; helm history moderation --max 3
TAG=this-tag-does-not-exist TIMEOUT=90s ./scripts/deploy.sh || echo ">>> Deploy failed, as expected; Helm rolled back."
echo "Release history after:"; helm history moderation --max 4
kubectl get pods
