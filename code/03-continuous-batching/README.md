# 03 — Continuous batching

A discrete-event simulation comparing static (request-level) batching
against continuous (iteration-level) batching, on the identical synthetic
workload, isolating scheduling policy as the only variable.

Companion to [`curriculum/03-throughput.md`](../../curriculum/03-throughput.md).

## Run it

```bash
python3 scheduler_sim.py
```

No dependencies - pure Python 3 stdlib.

## What to look for

- **Slot utilization**: static batching typically lands well under 30% on
  a bursty, mixed-length workload; continuous batching typically clears
  85%+ on the identical workload. That gap *is* head-of-line blocking,
  measured rather than described.
- **Makespan and average latency**: both improve several-fold under
  continuous batching, using the exact same hardware capacity and the
  exact same requests - the only variable that changed is the scheduling
  policy.
- Try changing `capacity` in `main()` - the gap between the two policies
  should shrink as capacity grows relative to the workload's arrival rate
  (with enough slots, static batching rarely has to make anyone wait for a
  slow neighbor) and widen as capacity shrinks relative to demand.
- Try making the workload less bursty/more uniform-length in
  `make_workload()` - continuous batching's advantage should shrink,
  because head-of-line blocking only bites when sequence lengths actually
  vary within a batch.

## What this does and doesn't model

This isolates the scheduling *policy* - it doesn't run a real model
forward pass, so there's no actual GPU compute time here, only step
counts. That's deliberate: it lets the mechanism show up in under 150
lines, runnable in milliseconds, with nothing else (a real model, a GPU, a
framework) able to obscure what's actually happening. For a from-scratch
continuous-batching engine running a real model's forward pass, see
[`skyzh/tiny-llm`](https://github.com/skyzh/tiny-llm)'s Week 3 ("Build a
Mini vLLM") - cited in
[`resources/curated-resources.md`](../../resources/curated-resources.md).
