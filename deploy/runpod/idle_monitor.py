#!/usr/bin/env python3
"""
Idle monitor for RunPod on-demand pods.

Polls the orchestrator health endpoint and auto-terminates the pod
when no active translation sessions exist for a configurable timeout.

Env vars:
    RUNPOD_POD_ID      — the pod's own ID (auto-injected by RunPod)
    RUNPOD_API_KEY     — for the termination API call
    IDLE_TIMEOUT_MIN   — idle minutes before self-termination (default: 30)
"""

import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [idle-monitor] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

HEALTH_URL = "http://localhost:8080/health"
POLL_INTERVAL_S = 60
RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"


def get_active_sessions() -> int | None:
    """Poll orchestrator health. Returns active_sessions or None on error."""
    try:
        req = urllib.request.Request(HEALTH_URL, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return data.get("active_sessions", 0)
    except Exception:
        return None


def terminate_pod(pod_id: str, api_key: str) -> bool:
    """Call RunPod GraphQL API to stop (not terminate) this pod.

    podStop preserves the container disk so models don't need re-downloading.
    Storage cost while stopped: ~$0.006/hr for 40 GB.
    """
    query = {
        "query": 'mutation { podStop(input: { podId: "%s" }) }' % pod_id,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    try:
        data = json.dumps(query).encode()
        req = urllib.request.Request(RUNPOD_GRAPHQL_URL, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
            if "errors" in body:
                log.error("GraphQL errors: %s", body["errors"])
                return False
            log.info("Pod termination response: %s", body)
            return True
    except Exception as e:
        log.error("Failed to terminate pod: %s", e)
        return False


def main():
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    idle_timeout_min = int(os.environ.get("IDLE_TIMEOUT_MIN", "30"))

    if not pod_id:
        log.warning("RUNPOD_POD_ID not set — monitoring only, termination disabled")
    if not api_key:
        log.warning("RUNPOD_API_KEY not set — monitoring only, termination disabled")

    log.info(
        "Started: timeout=%d min, poll=%d s, pod=%s",
        idle_timeout_min, POLL_INTERVAL_S, pod_id or "(none)",
    )

    idle_since: float | None = None

    while True:
        sessions = get_active_sessions()

        if sessions is None:
            # Orchestrator not up yet or error — skip, don't reset timer
            log.debug("Health check failed (orchestrator may be booting), skipping")
            time.sleep(POLL_INTERVAL_S)
            continue

        if sessions > 0:
            if idle_since is not None:
                idle_min = (time.time() - idle_since) / 60
                log.info("Idle reset: %d active session(s) (was idle %.1f min)", sessions, idle_min)
            idle_since = None
        else:
            if idle_since is None:
                idle_since = time.time()
                log.info("Idle started: 0 active sessions")

            idle_min = (time.time() - idle_since) / 60

            if idle_min >= idle_timeout_min:
                log.info(
                    "TERMINATING: idle for %.1f min (threshold: %d min)",
                    idle_min, idle_timeout_min,
                )
                if pod_id and api_key:
                    if terminate_pod(pod_id, api_key):
                        log.info("Pod stop request sent. Exiting.")
                        sys.exit(0)
                    else:
                        log.error("Pod stop failed — will retry next cycle")
                else:
                    log.warning("Would stop pod, but RUNPOD_POD_ID or RUNPOD_API_KEY not set")

        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    main()
