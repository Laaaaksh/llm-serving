# 04 — GGUF quantization: build, quantize, measure

Real GGUF quantization, actually run on real hardware - not a simulation.
Build llama.cpp, convert a real Hugging Face model to GGUF, quantize it to
three levels, and measure the actual size/speed/quality trade-off
yourself.

Companion to [`curriculum/04-quantization.md`](../../curriculum/04-quantization.md).
This lab's build of llama.cpp is reused by
[`code/06-speculative-decoding`](../06-speculative-decoding) and
[`code/07-llama-server-bench`](../07-llama-server-bench) - run this one
first.

## Verified against

- **llama.cpp** commit [`c841aeeb8`](https://github.com/ggml-org/llama.cpp/commit/c841aeeb8bb2fe417038dadfa9b007cf1a9ef950)
  (2026-08-29, build tag `b10687`) - pinned in [`LLAMA_CPP_COMMIT`](LLAMA_CPP_COMMIT).
  CI builds this exact commit on every push (see
  [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)); it does
  not re-download models or re-run the benchmark (that needs real disk and
  wall-clock time unsuited to per-PR CI - see the workflow's own comment).
- **Models**: [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
  (revision `7ae55760`) and [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
  (revision `989aa798`), both last modified 2024-09-25. They share the
  Qwen2 tokenizer, which [Stage 5](../../curriculum/05-speculative-decoding.md)'s
  speculative-decoding demo requires.
- **Hardware**: Apple M5, 10 cores (4P+6E), 16GB unified memory, Metal
  backend. No NVIDIA GPU - see [`curriculum/00-environment.md`](../../curriculum/00-environment.md).
- **Measured**: 2026-08-30.

If you run this on different hardware, a newer llama.cpp commit, or a
different model, your numbers will differ - that's expected and useful;
see [`CONTRIBUTING.md`](../../CONTRIBUTING.md) for how to report yours.

## Run it

```bash
./build.sh                 # clone + build llama.cpp at the pinned commit (~2 min)
./convert_and_quantize.sh  # download 2 models, convert to GGUF, quantize (~5-8 min, ~10GB disk)
./bench.sh                 # llama-bench + full WikiText-2 perplexity (~5 min)
./quality_comparison.sh    # same prompts, all quant levels, real output text
```

Total: about 20 minutes and 11GB of disk on the machine this was verified
on (624GB free at the time - if you're on a constrained disk, know that
torch's CPU wheel plus two models plus their GGUF conversions adds up
faster than "small models" might suggest).

## What was measured

### File sizes and bits-per-weight

| File | Size | BPW |
|---|---:|---:|
| 0.5B F16 | 948 MiB | 16.00 |
| 0.5B Q8_0 | 506 MiB | 8.50 |
| 0.5B Q4_K_M | 379 MiB | 6.35 |
| 0.5B Q2_K | 323 MiB | 5.39 |
| 1.5B F16 | 2.9 GiB | 16.00 |
| 1.5B Q4_K_M | 940 MiB | 5.08 |

**Gotcha, and it's a real one**: `llama-quantize` printed `WARNING: 144 of
290 tensor(s) required fallback quantization` for both the 0.5B model's
Q4_K_M and Q2_K. K-quants need tensor dimensions divisible by 256; Qwen2.5-0.5B's
hidden size is 896 (896/256 = 3.5), so llama.cpp silently falls back to
legacy quant types (a q4_0/q5_0/q6_K mix) for the affected tensors. This is
why the measured bits-per-weight above (6.35, 5.39) don't match
[`code/05-quant-memory-calculator`](../05-quant-memory-calculator)'s
generic 4.89/3.16 figures for Q4_K_M/Q2_K - those are llama.cpp's
documented figures for a model shape the K-quant kernel actually applies
to cleanly. **"Q4_K_M" doesn't mean a fixed bit-width - it means "the best
available quantization at roughly that target," which depends on the
specific model's tensor shapes.** Small/odd-shaped models are exactly
where this bites.

### Speed (`llama-bench`, 0.5B model, all four quant levels, Metal)

| Quant | Prompt processing (tok/s) | Generation (tok/s) |
|---|---:|---:|
| F16 | 9,740.8 ± 135.7 | 116.7 ± 2.0 |
| Q8_0 | 10,303.6 ± 81.4 | 194.2 ± 8.4 |
| Q4_K_M | 9,386.2 ± 158.5 | 235.8 ± 5.8 |
| Q2_K | 10,259.2 ± 111.6 | 262.8 ± 7.1 |

Prompt processing is roughly flat across quant levels - it's compute-bound
on this small a model (matches [Stage 1](../../curriculum/01-fundamentals.md)'s
prediction: prefill is compute-bound, so shrinking the weights doesn't
change the FLOPs count). Generation speed climbs clearly as the model
shrinks - more than doubling from F16 to Q2_K - because decode is
memory-bandwidth-bound, and a smaller model means fewer bytes to stream
per token, exactly as Stage 1's cost model predicts.

### Accuracy (`llama-perplexity`, full WikiText-2 test set - 584 chunks, ~300K tokens, not a subsample)

| Quant | Perplexity | Change vs. F16 |
|---|---:|---:|
| F16 | 15.184 ± 0.114 | — |
| Q8_0 | 15.200 ± 0.114 | +0.016 (essentially lossless) |
| Q4_K_M | 15.530 ± 0.116 | +0.35 (small but real) |
| Q2_K | 19.107 ± 0.145 | +3.92 (~26% relative - a clear, visible hit) |

### What quantization actually does to output (same two prompts, all quant levels, greedy/seed=42)

A trivial factual question ("What is the capital of France?") gets the
identical correct answer at every quant level, including Q2_K.

A word-problem prompt ("A farmer has 17 sheep. All but 9 die...") is more
revealing: **all four quant levels reach the same final numeric answer
(8), and all four make the identical reasoning slip** - treating "all but
9 die" as "9 died" rather than "9 survived" (which happens to still land
on the right number for this specific problem's phrasing, by
coincidence). Q2_K's phrasing is noticeably terser and less hedged than
the other three - a mild, real style degradation that didn't break
correctness or coherence here. **This is the honest limit of a
quality-comparison of two prompts: PPL (above) caught a real 26%
degradation at Q2_K that eyeballing this specific pair of outputs would
have mostly missed.** Run `quality_comparison.sh` yourself with your own
prompts, including ones that stress reasoning harder, before trusting
either signal alone.

## Toolchain gotchas hit during this lab (as of llama.cpp b10687)

- `cmake` is not preinstalled on macOS even with Xcode CLI tools -
  `brew install cmake` first.
- Binaries have moved out of `examples/` into `tools/` for most
  user-facing programs (`llama-cli` from `tools/cli`, `llama-quantize`
  from `tools/quantize`, etc.) - a lab or tutorial written against an
  older layout will reference stale paths.
- `llama-cli` is a chat-style REPL by default now, not a raw-completion
  tool - there is no `-no-cnv` flag any more (it was removed); `-st`
  (single-turn) is what these scripts use for clean one-shot output.
- `hf download` (from the `huggingface_hub` package) has superseded
  `huggingface-cli download`, which now prints a deprecation warning.
- See [`resources/common-pitfalls.md`](../../resources/common-pitfalls.md)
  for the rest, including speculative decoding's flag changes (relevant to
  [`code/06`](../06-speculative-decoding)).
