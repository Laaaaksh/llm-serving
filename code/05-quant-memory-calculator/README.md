# 05 — Quantization memory calculator

Computes real weight-memory and KV-cache-memory footprints across FP16,
INT8, GPTQ/AWQ 4-bit, and every common GGUF k-quant/i-quant level, and
answers the question that actually matters: does this model, at this
quantization, fit on this GPU, at this batch size and context length.

Companion to [`curriculum/04-quantization.md`](../../curriculum/04-quantization.md).

## Run it

```bash
python3 memory_calculator.py
```

No dependencies - pure Python 3 stdlib.

## What to look for

- The first table's weight-memory column shrinking from ~13GB (FP16) down
  to ~2.5GB (GGUF Q2_K) for the same 6.48B-parameter model - the raw
  compression ratio each format buys.
- The second table's `fits?` column: at the chosen batch size and context
  length, FP16 doesn't fit a 24GB GPU at all once the KV cache (computed
  from [`code/02-kv-cache-paging`](../02-kv-cache-paging)'s own model) is
  added in - several quantized formats do. Change `gpu_vram_gb`,
  `batch_size`, or `context_len` in `main()` and watch the crossover point
  move.
- Notice the KV cache is often bigger than the quantized weights
  themselves at low bit-widths - quantizing the weights and managing the
  KV cache well ([`code/02-kv-cache-paging`](../02-kv-cache-paging)) are
  solving the same memory problem from two different ends, and neither
  alone is the whole answer at scale.

## Where the bits-per-weight numbers come from

The GGUF figures are llama.cpp's own stated bits-per-weight values,
fetched from
[`tools/quantize/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
on 2026-08-30 - they include per-block scale/minimum overhead, which is
why e.g. "Q4" is really 4.89 bits/weight for Q4_K_M, not a clean 4. The
GPTQ/AWQ figure (4.25 bits/weight for group_size=128) is the standard
approximation: 4 bits of payload plus a 16-bit scale and zero-point shared
across each 128-weight group (4 + 32/128 = 4.25).
