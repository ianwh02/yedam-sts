#!/usr/bin/env python3
"""Interactive LLM translation test via CLI.

Type Korean text, get English translation. Maintains rolling context window.

Usage:
    python scripts/test_llm.py
    python scripts/test_llm.py --target ko --source en   # English → Korean
    python scripts/test_llm.py --context 5                # 5-pair context window
"""

import argparse
import sys
import time

import httpx

LLM_URL = "http://localhost:8000/v1/chat/completions"
LLM_MODEL = "Qwen/Qwen3-4B-AWQ"

SYSTEM_PROMPT = """/no_think
You are a real-time translation engine for a Korean church sermon.

RULES:
1. Output ONLY the translation — no explanations, notes, or formatting.
2. Maintain consistent terminology across the conversation.
3. STT Error Correction — the input comes from speech recognition and may contain errors:
   - Korean phonetic English: 에이블=able, 히미=Him, 후이즈=who is, 나우=now, 아멘=amen
   - 능히 하신다 = "He is able" (not "able to cook" etc.)
   - 집사님=deacon (not pastor), 목사님=pastor, 장로님=elder
   - 청년=youth/young adult (not teenager)
   - Fix obvious STT errors using surrounding context before translating.
4. If a sentence is cut off mid-thought, translate what's there naturally."""

CONTEXT_WINDOW = 3
MAX_CHARS_PER_FLUSH = 50  # ~10s of Korean speech
MAX_CONTEXT_TOKENS = 1200
MODEL_CONTEXT = 2048
CHARS_PER_TOKEN = 2.8


def estimate_tokens(chars: int) -> int:
    return int(chars / CHARS_PER_TOKEN) + 1


def build_messages(source: str, target: str, text: str, context: list) -> list:
    direction = f"{source}→{target}"
    system = f"{SYSTEM_PROMPT}\n\nTranslation direction: {direction}"

    if context:
        ctx_lines = "\n".join(
            f"[{source}] {pair['source']}\n[{target}] {pair['target']}"
            for pair in context
        )
        system += f"\n\nRecent context (for terminology consistency):\n{ctx_lines}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]


def cap_max_tokens(messages: list) -> int:
    total_chars = sum(len(m["content"]) for m in messages)
    input_tokens = estimate_tokens(total_chars)
    available = MODEL_CONTEXT - input_tokens - 50  # safety margin
    return max(64, min(256, available))


def main():
    parser = argparse.ArgumentParser(description="Interactive LLM translation test")
    parser.add_argument("--source", default="ko", help="Source language (default: ko)")
    parser.add_argument("--target", default="en", help="Target language (default: en)")
    parser.add_argument("--context", type=int, default=CONTEXT_WINDOW, help=f"Context window size (default: {CONTEXT_WINDOW})")
    parser.add_argument("--stream", action="store_true", help="Stream LLM output token by token")
    parser.add_argument("--model", default=LLM_MODEL, help=f"Model name (default: {LLM_MODEL})")
    parser.add_argument("--url", default=LLM_URL, help=f"LLM endpoint (default: {LLM_URL})")
    args = parser.parse_args()

    print(f"=== LLM Translation Test ({args.source}→{args.target}) ===")
    print(f"Model: {args.model}")
    print(f"Context window: {args.context}")
    print(f"Streaming: {'yes' if args.stream else 'no'}")
    print(f"{'─' * 50}")
    print(f"Type text to translate. Ctrl+C to quit.\n")

    context = []
    client = httpx.Client(timeout=30.0)

    # Health check
    try:
        r = client.get(args.url.rsplit("/v1", 1)[0] + "/health")
        if r.status_code != 200:
            print(f"LLM health check failed: {r.status_code}")
            sys.exit(1)
        print("LLM server healthy.\n")
    except httpx.ConnectError:
        print(f"Cannot connect to LLM at {args.url}")
        sys.exit(1)

    try:
        while True:
            try:
                text = input(f"  [{args.source}] ").strip()
            except EOFError:
                break

            if not text:
                continue

            messages = build_messages(args.source, args.target, text, context[-args.context:])
            max_tokens = cap_max_tokens(messages)

            t0 = time.monotonic()

            if args.stream:
                # Streaming mode
                payload = {
                    "model": args.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                    "stream": True,
                }
                print(f"  [{args.target}] ", end="", flush=True)
                full_text = ""
                with client.stream("POST", args.url, json=payload) as resp:
                    for line in resp.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        import json
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            print(delta, end="", flush=True)
                            full_text += delta
                elapsed = time.monotonic() - t0
                print(f"  ({elapsed * 1000:.0f}ms)")
                translation = full_text.strip()
            else:
                # Non-streaming mode
                payload = {
                    "model": args.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                    "stream": False,
                }
                resp = client.post(args.url, json=payload)
                elapsed = time.monotonic() - t0
                result = resp.json()
                translation = result["choices"][0]["message"]["content"].strip()
                print(f"  [{args.target}] {translation}  ({elapsed * 1000:.0f}ms)")

            # Update context
            context.append({"source": text, "target": translation})

    except KeyboardInterrupt:
        print("\n\nDone.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
