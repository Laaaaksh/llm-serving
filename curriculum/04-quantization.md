# Stage 4 — Quantization: GGUF hands-on, and the GPTQ/AWQ/GGUF/bitsandbytes decision

**You'll be able to:** actually quantize a model yourself and see the real
accuracy/size/speed trade-off, not a claimed one - and choose among
GPTQ, AWQ, GGUF, and bitsandbytes for a stated hardware and accuracy
constraint, rather than picking whichever one a blog post mentioned first.

**Time:** 5–8 hours - this is the longest stage, mostly because Part 1
involves real downloads and a real build.

**Build:** [`code/04-gguf-quantize-bench`](../code/04-gguf-quantize-bench)
(real, hands-on) and [`code/05-quant-memory-calculator`](../code/05-quant-memory-calculator)
(the decision-support math).

## Part 1: quantize a real model yourself

Read **[llama.cpp's imatrix docs](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md)**
and the informal but widely-cited **[K-quants explainer, GitHub Discussion #2094](https://github.com/ggml-org/llama.cpp/discussions/2094)**
(2023-07 - three years old and a discussion thread, not maintained docs,
but still the most common reference for what "Q4_K_M" actually means and
why the K-quant family beats the legacy Q4_0/Q5_0 quantizers it
superseded) before you start. Then work through
[`code/04-gguf-quantize-bench`](../code/04-gguf-quantize-bench) - it walks
you through building llama.cpp, converting a real Hugging Face model to
GGUF, quantizing it to several k-quant levels, and measuring the real
file-size, speed, and output-quality difference yourself, with the exact
commands and the numbers this repo measured, so you can compare your run
to a real baseline.

For the actual empirical trade-off at each quant level, on a model close in
size to what you'll run: **["Which Quantization Should I Use? A Unified
Evaluation of llama.cpp Quantization on Llama-3.1-8B-Instruct"](https://arxiv.org/html/2601.14277v1)**
(Uygar Kurt, 2026-01-11) is a real measured benchmark (not a listicle)
across 13 llama.cpp quant configs on GSM8K, HellaSwag, and other tasks.
Its headline finding: Q5_0 gets a 65% size reduction while *slightly
improving* aggregate benchmark scores vs. FP16; Q3_K_S shows the sharpest
quality drop, concentrated in math reasoning (GSM8K), while commonsense
tasks (HellaSwag) barely move at any quant level. It does not evaluate
imatrix-based quantization - a real, currently-open gap in the published
literature, not just this curriculum.

## Part 2: the decision - GPTQ vs. AWQ vs. GGUF vs. bitsandbytes

Read, in this order:

1. **[Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"](https://arxiv.org/abs/2210.17323)**
   (2022-10-31, ICLR 2023) - one-shot weight quantization using approximate
   second-order (Hessian) information; the origin algorithm most GPU-side
   4-bit quantization still descends from. **Don't use its reference repo**
   (`IST-DASLab/gptq`) - it's had no commits since July 2023. The actively
   maintained implementation today is **[GPTQModel](https://github.com/ModelCloud/GPTQModel)**
   (ModelCloud), which Hugging Face's own docs describe as having "fully
   supplanted AutoGPTQ and AutoAWQ" for Transformers/PEFT integration -
   check its releases (as recent as Aug 2026 at the time of writing) before
   trusting any older GPTQ tutorial's install instructions.
2. **[Lin et al., "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration"](https://arxiv.org/abs/2306.00978)**
   (2023-06-01, latest revision 2026-04-25, MLSys 2024 Best Paper) -
   protects the small fraction of "salient" weight channels (found via
   activation magnitude, not weight magnitude) at higher precision. The
   [`mit-han-lab/llm-awq`](https://github.com/mit-han-lab/llm-awq) repo is
   still active (last commit 2025-07-17, checked via GitHub's API
   2026-08-30).
3. **[Hugging Face Transformers, "Quantization" docs](https://huggingface.co/docs/transformers/quantization/overview)**
   - a living comparison table across roughly 20 quantization backends,
   with hardware support (CPU/CUDA/ROCm/Metal/Intel-GPU), on-the-fly vs.
   pre-quantized support, and PEFT fine-tuning compatibility. This is the
   single best "which backends even run on my hardware" reference - check
   it directly rather than trusting a static table here, since it's
   continuously updated and this repo's isn't.
4. **["GPTQ vs AWQ vs GGUF: Which 4-Bit to Pick in 2026"](https://theaiengineer.substack.com/p/quantization-in-practice-gptq-vs)**
   (Paolo Perrone, 2026-05-16), citing a January 2026 benchmark
   (Qwen2.5-32B-Instruct, single NVIDIA H200, via vLLM): FP16/AWQ/GPTQ/GGUF
   Q4_K_M/bitsandbytes NF4 perplexity within about half a point of each
   other on Wikitext-2, but throughput varying by **10x depending on which
   inference *kernel* served the format** - AWQ with vLLM's default kernel
   measured 68 tok/s, the same AWQ weights with the Marlin kernel measured
   741 tok/s. **This is the single most useful fact in this stage's
   reading**: kernel choice, not quantization method, was the dominant
   throughput variable in this benchmark. Don't pick a quantization format
   without also checking which kernel your serving engine actually uses
   for it.

## The decision, as a table

| Format | Needs | Best for | Watch out for |
|---|---|---|---|
| **GPTQ** (via GPTQModel) | CUDA GPU + a quantization-aware kernel (e.g. Marlin) | GPU serving where the engine has a fast kernel for it | Original reference repo is dead; use GPTQModel |
| **AWQ** | CUDA GPU (also CPU/ROCm/Intel-GPU per HF's matrix) | Similar niche to GPTQ; slightly different accuracy/speed trade-off - benchmark both on your model | Throughput is dominated by kernel choice (see above), not the method itself |
| **GGUF (k-quants)** | CPU, Apple Metal, CUDA, Vulkan - llama.cpp's own backends | CPU/edge/Apple-Silicon serving, or when you want one format that runs everywhere | Naming (Q4_K_M vs. Q4_K_S vs. Q3_K_L) is genuinely confusing at first - see `common-pitfalls.md` |
| **bitsandbytes** | CUDA GPU (also broad HF-listed hardware support) | Fast to try inside the Hugging Face/Transformers ecosystem, on-the-fly, no separate quantization step | Not always the fastest at inference time - it optimizes for ease of use, not peak throughput |

## Do

1. Work through [`code/04-gguf-quantize-bench`](../code/04-gguf-quantize-bench)
   end to end - build llama.cpp, convert, quantize, and run the benchmark
   and quality-comparison scripts yourself. Compare your numbers to the
   ones in its README (same commands, your hardware).
2. Run [`code/05-quant-memory-calculator/memory_calculator.py`](../code/05-quant-memory-calculator/memory_calculator.py).
   It computes real memory footprints (weights + KV cache, from Stage 1 and
   Stage 2's own numbers) across every format in the table above, for a
   specific GPU size, batch size, and context length - and shows a
   realistic case where the *only* way to fit the workload is to quantize.

## Checkpoint

- Your model fits in FP16 on your GPU, but only at batch size 1. Is
  quantizing to Q4 primarily buying you speed, memory, or both, and for
  which of Stage 1's two regimes (prefill or decode) does that matter most?
- Why is `IST-DASLab/gptq` the wrong repo to install today, even though
  it's the original GPTQ paper's own code?
- The substack benchmark found AWQ's throughput varying 10x by kernel
  choice alone. What does that imply about trusting a quantization
  method's reputation ("AWQ is fast") without checking which serving
  engine and kernel you'd actually be using it with?
- Name one real accuracy cost GGUF's Q3_K_S showed in the Kurt benchmark,
  and one task where even aggressive quantization barely mattered.

Next: [Stage 5 — speculative decoding](05-speculative-decoding.md).
