# 07 — A real serving stack, running and benchmarked

The capstone: launch a real OpenAI-compatible server, hit it with real
HTTP requests, and measure real latency and throughput - the one corner of
[Stage 6](../../curriculum/06-choosing-an-engine.md)'s four-engine
comparison this repo can actually run without an NVIDIA GPU.

Companion to [`curriculum/06-choosing-an-engine.md`](../../curriculum/06-choosing-an-engine.md).

## Run it

Run [`code/04-gguf-quantize-bench`](../04-gguf-quantize-bench)'s
`build.sh` and `convert_and_quantize.sh` first - this reuses that llama.cpp
build and the quantized 1.5B model. Then:

```bash
./start_server.sh   # launches llama-server in the background, waits until ready
python3 bench_client.py
./stop_server.sh    # when you're done
```

`bench_client.py` has zero dependencies beyond the Python stdlib
(`urllib`, `json`) - it's a real OpenAI-compatible client, not a special
llama.cpp-only tool, so the same script pattern works against any
OpenAI-compatible server, including vLLM/SGLang/TensorRT-LLM's own
OpenAI-compatible endpoints if you have GPU access to run one.

## What was measured

Verified 2026-08-30, llama.cpp commit `c841aeeb8` (see
[`code/04-gguf-quantize-bench/LLAMA_CPP_COMMIT`](../04-gguf-quantize-bench/LLAMA_CPP_COMMIT)),
Apple M5, Metal, Qwen2.5-1.5B-Instruct Q4_K_M, `-c 4096`:

Request: `"Explain what a KV cache is in LLM inference, in two sentences."`
(temp=0, seed=42, max_tokens=150).

| Metric | Value |
|---|---:|
| HTTP status | 200 |
| End-to-end latency | 0.689 s |
| Prompt tokens | 45 |
| Completion tokens | 67 |
| Prompt processing speed | 918.9 tok/s |
| Generation speed | 103.8 tok/s |

Sample response:

> "A KV cache in LLM (Large Language Model) inference refers to a type of
> memory that stores key-value pairs for quick access and retrieval during
> the processing of language models. This allows for efficient handling of
> frequently used data without needing to recompute or look up information
> every time it's needed, significantly speeding up the inference process."

## What this does and doesn't demonstrate

This sends requests **sequentially, one at a time** - it proves the
serving stack works end to end (real weights, real server, real HTTP,
real measured numbers), but it deliberately does not demonstrate
continuous batching's throughput advantage, because that needs genuinely
concurrent load. `llama-server` does support concurrent request handling
(it has its own internal batching), but isolating and measuring that
effect cleanly is exactly what
[`code/03-continuous-batching`](../03-continuous-batching)'s simulator
does, without needing a live server, a real model, or network calls in the
loop.

If you have NVIDIA GPU access (see
[`curriculum/00-environment.md`](../../curriculum/00-environment.md) for
current rental pricing), the natural next step is running this exact
client script (or a concurrent version of it) against vLLM's own
OpenAI-compatible server and comparing - vLLM's quickstart is linked in
[`resources/curated-resources.md`](../../resources/curated-resources.md).
