# Stage 0 — Environment & the hardware question

**You'll be able to:** know, before you spend money or time, which parts of
LLM serving you can learn on a laptop and which genuinely need a rented or
owned GPU - and why.

**Time:** 30–60 minutes.

**Build:** nothing yet. This stage ends when you know which of the three
tiers below you're starting in.

## The honest answer, up front

This repository was built and every sample in it verified **without an
NVIDIA GPU** - on Apple Silicon (macOS, no CUDA). That shaped what's
runnable here, and you should know the shape before you start:

| Tier | What it needs | What's in this repo |
|---|---|---|
| **1. Pure Python, no dependencies** | Any computer | Stages 1, 2, 3, 5's simulators, Stage 4's memory calculator - all of `code/01`, `02`, `03`, `05`, `06`'s math |
| **2. CPU or Apple Silicon, no NVIDIA GPU needed** | ~10GB disk, a few GB RAM, patience | Stage 4's real GGUF quantization lab and Stage 6's capstone (`code/04`, `06`'s llama.cpp demo, `07`) - these build and run [llama.cpp](https://github.com/ggml-org/llama.cpp) locally and download small (0.5B-1.5B parameter) real models |
| **3. An actual NVIDIA GPU (rented or owned)** | See below | Running vLLM, TensorRT-LLM, or SGLang yourself. **Stage 6's engine decision guide covers all three in depth, with cited official docs and benchmark data - but none of the three is run inside this repository.** This repo does not have NVIDIA GPU access; if you want to reproduce vLLM/TensorRT-LLM/SGLang's own numbers or run their quickstarts, you need Tier 3 hardware. Say so plainly, don't discover it by a confusing failure. |

If you only ever work through Tiers 1 and 2, you will genuinely understand
every mechanism this curriculum teaches - the KV-cache math, the scheduling
algorithm, the quantization trade-off, the speculative-decoding math - and
you will have run real inference and real quantization, just not at
data-center scale or on a production engine. Tier 3 buys you seeing the
same mechanisms at the scale they actually matter commercially, plus
hands-on time with the engines named in nearly every job posting cited in
this repo's research. Both are legitimate stopping points depending on what
you're optimizing for.

## Tier 1 + 2 setup (do this now)

You need:

- **Python 3.9+** (`python3 --version`). No packages required for Stages 1,
  2, 3, 5, or the Stage 4 calculator - they're stdlib-only, on purpose, so
  nothing here can go stale on you via a dependency upgrade.
- **A C/C++ toolchain and CMake**, for Stage 4/6's llama.cpp build:
  - macOS: `xcode-select --install` (gives you `clang`), then
    `brew install cmake`.
  - Linux: `sudo apt install build-essential cmake` (Debian/Ubuntu) or your
    distro's equivalent.
- **~10GB free disk** for the two small models (Qwen2.5-0.5B-Instruct and
  Qwen2.5-1.5B-Instruct, used throughout Stage 4-6) in F16 and several
  quantized forms, plus the llama.cpp build itself.
- A Python **virtual environment** for the one Python dependency the GGUF
  conversion script needs (`transformers`, `torch`, `sentencepiece`,
  `gguf` - all CPU-only, no CUDA needed even on a machine that has an
  NVIDIA GPU). Exact versions and install command are pinned in
  [`code/04-gguf-quantize-bench/README.md`](../code/04-gguf-quantize-bench/README.md).

That's it - no GPU, no CUDA toolkit, no driver version fights (see
[`resources/common-pitfalls.md`](../resources/common-pitfalls.md) for what
those look like when you do reach Tier 3, so you recognize them rather than
panic).

## Tier 3: renting a GPU, if and when you want it

You do not need this for Stages 1-5 or Stage 6's decision guide itself.
You'd want it to:

- Run vLLM, TensorRT-LLM, or SGLang's own quickstart yourself instead of
  reading about them.
- Reproduce a real multi-GPU tensor-parallelism deployment.
- Benchmark a 7B+ model at production batch sizes rather than the
  0.5B-1.5B models this repo uses for laptop-friendly labs.

On-demand hourly rates for a single GPU, checked directly against each
provider's own pricing page on 2026-08-30 (prices change constantly -
re-check before committing, and treat this table as a starting point, not
a live quote):

| Provider | GPU | $/hr (on-demand) | Source |
|---|---|---|---|
| [RunPod](https://runpod.io) | L4 | $0.44 (Community) / $0.49 (Secure) | [runpod.io/pricing](https://www.runpod.io/pricing), fetched 2026-08-30 |
| [RunPod](https://runpod.io) | A100 (PCIe) | $1.39 (Community) / $1.59 (Secure) | same |
| [RunPod](https://runpod.io) | H100 (PCIe) | $1.99 (Community) / $2.89 (Secure) | same |
| [Lambda](https://lambda.ai) | A10 (24GB), 1x | $1.29 | [lambda.ai/service/gpu-cloud](https://lambda.ai/service/gpu-cloud), fetched 2026-08-30 |
| [Lambda](https://lambda.ai) | A100 (80GB), 1x | $1.99 | same |
| [Lambda](https://lambda.ai) | H100 (80GB), 1x | $4.29 | same |

An L4 or A10 (16-24GB) is enough to run a 7B model in GPTQ/AWQ/GGUF
quantized form under vLLM or llama.cpp's CUDA backend and see real
continuous-batching throughput. An A100/H100 is what you'd want to
reproduce the multi-GPU tensor-parallelism and disaggregated-serving
material in Stage 6. Neither this repo nor its author independently
verified these providers' actual provisioning experience - only the listed
prices, fetched directly from each provider's own page.

## Checkpoint

You're done with this stage when:

```bash
python3 --version   # 3.9+
cmake --version      # any recent version
```

both succeed, and you know which tier you're starting in. Next: [Stage 1 —
fundamentals](01-fundamentals.md).
