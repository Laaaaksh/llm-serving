# 01 — Token cost model

Real FLOPs and bytes-moved arithmetic for a Llama-2-7B-shaped model on an
A100, showing why prefill is compute-bound and decode is memory-bound - and
exactly where batching's benefit runs into a hard ceiling.

Companion to [`curriculum/01-fundamentals.md`](../../curriculum/01-fundamentals.md).

## Run it

```bash
python3 cost_model.py
```

No dependencies - pure Python 3 stdlib. Runs in under a second.

## What to look for

- The prefill table's `bound` column flipping from `memory` to `compute` as
  prompt length grows - at `prompt_len=1`, prefill and decode are
  mathematically the same operation, so this is where the whole curriculum
  starts.
- The decode table staying pinned at `FLOPs/byte = 1.0` regardless of
  context length, at batch size 1 - both the numerator and denominator grow
  identically with context, so the ratio never moves. This is why "just
  use a bigger context window" doesn't change whether decode is
  memory-bound.
- The batching table: intensity climbs with batch size, but **doesn't
  climb forever**. Read the printed explanation for why, then verify it
  yourself - change `ctx` in `main()` from 128 back to 2048 and re-run;
  the batching table should now stay memory-bound even at batch_size=2048.
  That's the real mechanism behind why long-context serving needs more
  than "just add more concurrent requests" to become efficient.
- The final assertions are the checkpoint - if you change `LLAMA2_7B` or
  `A100_80GB`'s numbers and an assertion fails, that's telling you
  something true changed, not that the script is broken.

## Where the hardware numbers come from

312 TFLOP/s and 2,039 GB/s are NVIDIA's own stated A100 80GB SXM figures
(BF16 Tensor Core, no sparsity), from
[nvidia.com/en-us/data-center/a100](https://www.nvidia.com/en-us/data-center/a100/),
fetched 2026-08-30. Real hardware varies - the point of this script is the
*shape* of the result (prefill compute-bound, decode memory-bound, a
batching ceiling exists), which holds on any accelerator, not these exact
numbers on yours.
