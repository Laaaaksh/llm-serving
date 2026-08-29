# The curriculum

Seven stages, in order. Each one names what you'll be able to do at the
end, what to read or watch, roughly how long it takes, and what to build to
prove it stuck. Do them in order the first time through - later stages
assume earlier ones, and Stage 4 onward literally import numbers computed
in Stage 1 and Stage 2's code.

| Stage | You'll be able to... | Time | Build |
|---|---|---|---|
| [0 — Environment & the hardware question](00-environment.md) | Know what needs a GPU and what doesn't, before you spend money on one | 30–60 min | Confirm your toolchain; no GPU required for this repo |
| [1 — Fundamentals](01-fundamentals.md) | Explain why decode is memory-bound and prefill is compute-bound, and why that gap is the entire case for batching | 3–5 hr | `code/01-token-cost-model` |
| [2 — Memory](02-memory.md) | Explain and compute KV-cache size, PagedAttention's block allocation, prefix caching, and the GQA/MLA trade-off | 3–5 hr | `code/02-kv-cache-paging` |
| [3 — Throughput](03-throughput.md) | Explain continuous batching and a real scheduler's admission/eviction policy | 3–4 hr | `code/03-continuous-batching` |
| [4 — Quantization](04-quantization.md) | Quantize a real model to GGUF, measure the accuracy/size/speed trade-off yourself, and choose GPTQ vs. AWQ vs. GGUF vs. bitsandbytes for a given constraint | 5–8 hr | `code/04-gguf-quantize-bench`, `code/05-quant-memory-calculator` |
| [5 — Speculative decoding](05-speculative-decoding.md) | Explain the draft/target mechanism and compute when it helps vs. hurts | 2–4 hr | `code/06-speculative-decoding` |
| [6 — Choosing and running an engine](06-choosing-an-engine.md) | Map vLLM/TensorRT-LLM/SGLang/llama.cpp to a real deployment constraint, and run a real serving stack end to end | 4–6 hr | `code/07-llama-server-bench` |

**Total: roughly 20–30 hours** of focused work, spread over however long
that takes you. There's no clock running.

## Prerequisites

You should be comfortable with Python, comfortable reading a paper or a
technical blog post without hand-holding, and have used an LLM through an
API or chat interface enough to have a rough sense of what "prompt" and
"completion" mean. You do **not** need prior GPU programming, CUDA, or
distributed-systems experience - this curriculum is about the serving
system's behavior, not about writing kernels. If you want that layer too,
see [`github.com/Laaaaksh/cuda-engineering`](https://github.com/Laaaaksh/cuda-engineering),
a sibling repo in this series.

## Read this before Stage 1: what hardware you actually need

**Stages 1, 2, 3, and 5's simulators, and Stage 4's decision-guide
calculator, are pure Python with no dependencies - they run on anything,
including a laptop with no GPU at all.** Stage 4's hands-on GGUF lab and
Stage 6's capstone need enough RAM and disk to build
[llama.cpp](https://github.com/ggml-org/llama.cpp) and run small (0.5B-1.5B
parameter) models on CPU or Apple Silicon - no NVIDIA GPU required for
those either. **This whole repository was built and verified without an
NVIDIA GPU** - see [Stage 0](00-environment.md) for exactly what that means
for what you can and can't verify yourself without one, and what it costs
to rent one for the parts that need it (multi-GPU tensor parallelism,
real vLLM/TensorRT-LLM/SGLang throughput numbers - Stage 6's decision guide
covers these engines but they are not runnable in this repo without
separate GPU access. Say so explicitly rather than implying otherwise.)

## How each stage is structured

- **What to read or watch** - a short, sequenced list, not an unordered
  pile. Full annotated details (what's covered well, how current it is,
  and what's stale but still worth it) live in
  [`resources/curated-resources.md`](../resources/curated-resources.md) -
  each stage links the specific entries relevant to it.
- **What to build** - a runnable sample in [`code/`](../code), with its own
  README. Every pure-Python sample actually runs and asserts its own
  result at the end (a "CHECKPOINT PASSED" line) - if you break the model
  assumptions, it tells you. The GGUF/llama.cpp labs give you exact
  commands and the real numbers this repo measured, so you can compare your
  own run against them.
- **Checkpoint questions** - answer these without looking anything up
  before moving on.

## If something stops you

[`resources/common-pitfalls.md`](../resources/common-pitfalls.md) collects
the things that stop most people starting this topic - the misconceptions
almost everyone starts with (conflating latency and throughput, thinking
quantization is "free" size reduction, assuming more batching is always
better), and the specific toolchain pain in building llama.cpp and
converting/quantizing a model. Check there before assuming you've found a
new problem.
