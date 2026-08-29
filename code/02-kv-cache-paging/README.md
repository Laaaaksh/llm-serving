# 02 — KV-cache paging

A from-scratch simulation of naive contiguous KV-cache allocation vs.
PagedAttention-style block-based allocation, plus prefix sharing across
requests with a common system prompt.

Companion to [`curriculum/02-memory.md`](../../curriculum/02-memory.md).

## Run it

```bash
python3 kv_cache_sim.py
```

No dependencies - pure Python 3 stdlib.

## What to look for

- The naive allocator reserves `max_seq_len` tokens per sequence
  regardless of how long the sequence actually turns out to be - watch the
  "actually used" percentage in the first section; it's normally under
  10% for a realistic mix of short and long requests.
- The paged allocator's waste is bounded to internal fragmentation in at
  most one block per sequence - watch how small the wasted-token count is
  compared to the naive approach's wasted gigabytes.
- The "at the same memory budget, paging fits Nx as many sequences" line -
  this is the concrete, measurable version of PagedAttention's real-world
  throughput claim (the paper reports 2-4x over prior systems; this
  script's exact multiplier depends on the workload's length variance, not
  a fixed constant - try changing the `random.randint` bounds in
  `main()`'s `workload` line and watch the multiplier change).
- The prefix-sharing section: increase `shared_prompt_len` and watch memory
  savings rise - this is the mechanism behind why a long, repeated system
  prompt is nearly free across many requests once cached, but expensive
  the first time.

## What this does and doesn't model

This is a mechanism simulation - block allocation, reference counting,
prefix matching - not vLLM's or SGLang's actual code, and not a real
attention kernel. It gets you to the same conclusion the PagedAttention
paper and vLLM's prefix-caching docs describe, with numbers you compute
yourself. For a from-scratch *implementation* that goes further (a real
paged-KV attention kernel over an actual model), see
[`skyzh/tiny-llm`](https://github.com/skyzh/tiny-llm)'s Week 2-3 material -
cited in [`resources/curated-resources.md`](../../resources/curated-resources.md).
