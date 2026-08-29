#!/usr/bin/env python3
"""
A real OpenAI-compatible client, no dependencies beyond the Python
stdlib - hits a running llama-server instance and reports real measured
latency and throughput, computed from the server's own reported token
counts.

Run start_server.sh first (it launches llama-server on port 8899 with
Qwen2.5-1.5B-Instruct Q4_K_M), then:

    python3 bench_client.py

Companion to curriculum/06-choosing-an-engine.md.
"""

from __future__ import annotations

import json
import time
import urllib.request


SERVER_URL = "http://127.0.0.1:8899/v1/chat/completions"

PROMPTS = [
    "Explain what a KV cache is in LLM inference, in two sentences.",
    "What is continuous batching? Answer in one sentence.",
    "Name one trade-off of 2-bit quantization.",
]


def chat_completion(prompt: str, max_tokens: int = 150) -> dict:
    payload = json.dumps({
        "model": "qwen2.5-1.5b-instruct-q4_k_m",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 42,
        "max_tokens": max_tokens,
    }).encode()

    req = urllib.request.Request(
        SERVER_URL, data=payload, headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())
    elapsed = time.monotonic() - start
    return {"elapsed_s": elapsed, "body": body}


def main() -> None:
    print(f"Hitting {SERVER_URL} with {len(PROMPTS)} real requests...")
    print()
    total_completion_tokens = 0
    total_elapsed = 0.0

    for prompt in PROMPTS:
        try:
            result = chat_completion(prompt)
        except Exception as e:
            print(f"Request failed - is start_server.sh running? ({e})")
            raise SystemExit(1)

        usage = result["body"]["usage"]
        text = result["body"]["choices"][0]["message"]["content"]
        gen_tok_s = usage["completion_tokens"] / result["elapsed_s"]

        print(f"Prompt: {prompt}")
        print(f"  {usage['prompt_tokens']} prompt tokens, "
              f"{usage['completion_tokens']} completion tokens, "
              f"{result['elapsed_s']:.2f}s wall, {gen_tok_s:.1f} tok/s (naive, "
              f"includes prompt processing + network - not a pure decode-speed number)")
        print(f"  Response: {text[:150]}{'...' if len(text) > 150 else ''}")
        print()

        total_completion_tokens += usage["completion_tokens"]
        total_elapsed += result["elapsed_s"]

    print(f"Total: {total_completion_tokens} completion tokens in "
          f"{total_elapsed:.2f}s across {len(PROMPTS)} sequential requests "
          f"= {total_completion_tokens / total_elapsed:.1f} tok/s aggregate.")
    print()
    print("Notice this is ONE request at a time - sequential, not batched.")
    print("Stage 3's whole point was that concurrent requests sharing a batch")
    print("get far higher AGGREGATE throughput than the same requests run")
    print("one after another, at some cost to any single request's latency.")
    print("This script deliberately doesn't demonstrate that trade-off directly")
    print("- llama-server does support concurrent request queuing, but showing")
    print("continuous batching's real effect needs concurrent load, which is")
    print("exactly what code/03-continuous-batching's simulator isolates")
    print("without needing a live server or real network calls.")


if __name__ == "__main__":
    main()
