# Stage 1 — Fundamentals: the cost of one token

**You'll be able to:** explain, with real arithmetic, why processing a
prompt (prefill) is compute-bound while generating each new token (decode)
is memory-bandwidth-bound - and why that single fact is the reason batching
exists at all.

**Time:** 3–5 hours.

**Build:** [`code/01-token-cost-model`](../code/01-token-cost-model) - run
it, then break it (change the model config, the hardware numbers, the
workload) and see which conclusions still hold.

## Read, in this order

1. **[Kipply's Blog, "Transformer Inference Arithmetic"](https://kipp.ly/transformer-inference-arithmetic/)**
   (Kipply, 2022-03-30). The clearest first-principles derivation of
   arithmetic intensity and the prefill/decode cost split anywhere - read
   this first, before anything newer, because it teaches the *reasoning*.
   Its hardware numbers (A100-era) are dated; the physics isn't. This
   repo's `code/01-token-cost-model` follows the same reasoning with
   numbers checked directly against [NVIDIA's own A100 datasheet](https://www.nvidia.com/en-us/data-center/a100/)
   (fetched 2026-08-30) instead of relying on the post's figures.
2. **[Anyscale, "How continuous batching enables 23x throughput in LLM inference while reducing p50 latency"](https://www.anyscale.com/blog/continuous-batching-llm-inference)**
   (Cade Daniel, Chen Shen, Eric Liang, Richard Liaw, 2023-06-22). Aged but
   still the clearest plain-language case for *why* naive, padded batching
   wastes a GPU - "LLM inference is memory-IO bound, not compute bound," in
   the post's own words. Read this as the bridge from Kipply's arithmetic
   to a real system's design.
3. **["Inside vLLM: Anatomy of a High-Throughput LLM Inference System"](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)**
   (Aleksa Gordic, vLLM Project, 2025-09-05). The best single overview of
   an actual production engine - scheduling, PagedAttention, chunked
   prefill, prefix caching, speculative decoding, multi-GPU scaling, all in
   one document. Broad rather than deep on any one mechanism - read it here
   as an overview/map, then let Stages 2, 3, 5, and 6 go deep on the pieces
   it only summarizes. The author has said this is the first in a planned
   series - worth checking for later parts.
4. **[Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention"](https://arxiv.org/abs/2309.06180)**
   (SOSP 2023). You'll read this properly in Stage 2 - for now, read just
   the introduction and Section 2 (background), which lay out the
   prefill/decode cost split and naive-batching's memory-fragmentation
   failure mode from the paper that made continuous batching + paged memory
   practical at production scale.

## Do

Run [`code/01-token-cost-model/cost_model.py`](../code/01-token-cost-model/cost_model.py).
It computes real FLOPs and real bytes-moved for a Llama-2-7B-shaped model
on an A100, and prints:

- Prefill's arithmetic intensity rising with prompt length, crossing from
  memory-bound to compute-bound as the prompt gets longer.
- Decode at batch size 1 staying memory-bound *regardless* of context
  length - the FLOPs/byte ratio is pinned at 1.0 no matter how long the
  context gets, because both terms scale together.
- Batching decode requests together raising arithmetic intensity - and,
  just as important, the ceiling that KV-cache size puts on how far
  batching alone can push you there. This is the script's least obvious
  result - make sure you understand why it isn't monotonic without a limit.
- A concrete padding-waste number for a batch of mixed-length requests -
  the exact failure mode Stage 3's scheduler exists to fix.

Then edit the model config (try a smaller/larger model, a different GPU's
numbers) and re-run. If a change breaks one of the script's own assertions,
that assertion told you something true about the physics - figure out why
before moving on.

## Checkpoint

Answer these without looking anything up:

- Why is a single decode step at batch size 1 memory-bound almost
  regardless of what GPU you run it on?
- If someone tells you "just increase the batch size and decode gets
  compute-bound," under what condition is that false? (The script's
  ceiling analysis has the answer.)
- A request with a 4,000-token prompt and a 20-token expected reply, run
  next to nine requests each with a 20-token prompt and a 20-token expected
  reply, in a single padded static batch. Which resource - compute or
  memory bandwidth - does the long prompt waste most for its
  batch-mates, and at which phase (prefill or decode)?

Next: [Stage 2 — memory](02-memory.md).
