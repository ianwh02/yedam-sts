#!/bin/bash
# Deploy yedam-sts to all configured providers.
#
# Usage:
#   ./deploy.sh              # build + push to all providers
#   ./deploy.sh runpod       # RunPod only
#   ./deploy.sh modal        # Modal only
#   ./deploy.sh build        # build image only (no push/deploy)
set -euo pipefail

IMAGE="ghcr.io/ianwh02/yedam-sts:latest"
TARGET="${1:-all}"

echo "=== Building Docker image ==="
docker build -f Dockerfile.slim -t "$IMAGE" .

if [ "$TARGET" = "build" ]; then
    echo "=== Build complete (no deploy) ==="
    exit 0
fi

if [ "$TARGET" = "all" ] || [ "$TARGET" = "runpod" ]; then
    echo "=== Pushing to GHCR (RunPod) ==="
    docker push "$IMAGE"
    echo "RunPod: restart pod to pick up new image"
fi

if [ "$TARGET" = "all" ] || [ "$TARGET" = "modal" ]; then
    echo "=== Deploying to Modal ==="
    python -m modal deploy deploy/modal/app.py
    echo "Modal: deployment complete"
fi

echo "=== Done ==="
