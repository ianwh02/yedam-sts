"""GPU warm-up script for serverless deployment.

Call this before a church service starts to ensure all models are
loaded into VRAM and ready for inference. Reduces first-request
latency from minutes to milliseconds.

Usage:
    python warmup.py [--timeout 300]
"""

import asyncio
import argparse
import sys
import time

import httpx


SERVICES = {
    "stt": {"url": "http://localhost:9090", "check": "websocket"},
    "llm": {"url": "http://localhost:8000/health", "check": "http"},
    "tts": {"url": "http://localhost:7860/health", "check": "http"},
    "orchestrator": {"url": "http://localhost:8080/health", "check": "http"},
}


async def wait_for_service(name: str, config: dict, timeout: float) -> bool:
    start = time.time()
    client = httpx.AsyncClient(timeout=5.0)

    while time.time() - start < timeout:
        try:
            if config["check"] == "http":
                resp = await client.get(config["url"])
                if resp.status_code == 200:
                    elapsed = time.time() - start
                    print(f"  [OK] {name} ready ({elapsed:.1f}s)")
                    await client.aclose()
                    return True
            elif config["check"] == "websocket":
                import websockets

                async with websockets.connect(config["url"]):
                    elapsed = time.time() - start
                    print(f"  [OK] {name} ready ({elapsed:.1f}s)")
                    await client.aclose()
                    return True
        except Exception:
            pass

        await asyncio.sleep(2.0)

    print(f"  [FAIL] {name} not ready after {timeout:.0f}s")
    await client.aclose()
    return False


async def warmup(timeout: float):
    print(f"Warming up services (timeout: {timeout:.0f}s)...")
    start = time.time()

    results = await asyncio.gather(
        *[wait_for_service(name, config, timeout) for name, config in SERVICES.items()]
    )

    elapsed = time.time() - start
    all_ok = all(results)

    if all_ok:
        print(f"\nAll services ready in {elapsed:.1f}s")
    else:
        print(f"\nSome services failed to start after {elapsed:.1f}s")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warm up GPU services")
    parser.add_argument("--timeout", type=float, default=300, help="Timeout in seconds")
    args = parser.parse_args()
    asyncio.run(warmup(args.timeout))
