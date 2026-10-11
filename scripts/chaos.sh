#!/usr/bin/env bash
# Resilience demo: kill a text-worker pod and Redis while posts are flowing; nothing is lost.
set -euo pipefail
IP="${IP:-$(cd "$(dirname "$0")/../infra/terraform" && terraform output -raw public_ip)}"
send() { for i in $(seq 1 "$1"); do curl -s -o /dev/null "http://$IP/posts" -H 'content-type: application/json' -d "{\"text\":\"chaos test $i\"}"; done; }
echo "1) Killing a text-worker pod while sending 50 posts"
kubectl delete pod -l app=text-worker --wait=false
send 50
echo "2) Killing Redis (the queue). Its append-only file is on a volume, so queued posts survive."
kubectl delete pod -l app=redis
kubectl rollout status deploy/redis --timeout=120s
sleep 5; send 20
kubectl get pods
echo "Check: expect 70. Messages a killed pod had read but not finished are re-claimed after 60 s:"
sleep 75
kubectl exec deploy/postgres -- psql -U moderation -tc "select count(*) from decisions where text like 'chaos test%'"
