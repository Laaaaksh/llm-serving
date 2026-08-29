# Stage 6 — Choosing and running an engine

**You'll be able to:** map a real deployment constraint (hardware,
latency/throughput target, feature need) to a choice among vLLM,
TensorRT-LLM, SGLang, and llama.cpp - and you'll have actually run one of
them, end to end, as a real serving stack.

**Time:** 4–6 hours.

**Build:** [`code/07-llama-server-bench`](../code/07-llama-server-bench) -
a real OpenAI-compatible server, actually running, actually benchmarked.

## Read this before anything else: this is a decision guide, not a ranking

Every one of the four engines below is under active, well-funded
development, and which one is "best" depends entirely on your constraints,
not a leaderboard position. The clearest evidence for that framing is
**Bench360** ([arXiv:2511.16682](https://arxiv.org/abs/2511.16682), Wu et
al., submitted 2025-11-12, revised 2026-01-14) - an academic benchmark
evaluating four inference engines across three GPUs and multiple
quantization formats on real NLP tasks, measuring task quality *and*
system behavior (latency, throughput, energy, startup time) together. Its
own stated headline finding: **"there is no universal best option."** Take
that seriously before trusting any single number below, including the ones
in this document.

A decision-tree-style resource for this exact question already exists
([vizuaraai.github.io's inference-engines guide](https://vizuaraai.github.io/inference-engineering-visual-guides/visual_walkthroughs/13_inference_engines.html)) -
it's a reasonable quick reference (GPU-only vs. broader hardware, "max
performance vs. easy setup"), but has no llama.cpp branch at all and no
cited benchmark data behind its claims. The guide below tries to improve
on it specifically by including CPU/edge serving and citing where each
claim actually comes from.

## Read, in this order

1. Each engine's own docs, for what it actually is before comparing
   anything: **[vLLM docs](https://docs.vllm.ai/en/latest/)**,
   **[TensorRT-LLM docs](https://nvidia.github.io/TensorRT-LLM/overview.html)**,
   **[SGLang docs](https://docs.sglang.io/)**, **[llama.cpp's README](https://github.com/ggml-org/llama.cpp)**.
2. **[Bench360](https://arxiv.org/abs/2511.16682)** (above) for the
   "no universal best option" framing, before you read anything that tries
   to crown a winner.
3. **[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)** - not a fifth
   engine, an orchestration layer *above* the other three (or above
   TensorRT-LLM specifically, though it dispatches across engines).
   Handles disaggregated prefill/decode across separate worker pools and
   KV-cache-aware request routing at a cluster level. NVIDIA calls it
   "1.0, production-ready" as of roughly March 2026, with point releases
   still shipping quickly. Relevant once you're doing multi-node serving;
   not a factor in a single-GPU or single-node engine choice.
4. **A caution about benchmark claims you'll see repeated elsewhere:** a
   specific "SGLang beats vLLM by ~29% on Llama 3.1 8B with prefix-heavy
   traffic" figure circulates across several 2026 blog posts. Tracing it
   back, every instance leads to the same two originating vendor
   benchmarks (Particula Tech, Spheron) that this repo's research could not
   independently fetch or verify - one aggregator that repeats the figure
   explicitly states it "aggregates published benchmarks rather than
   conducting independent testing." **Treat that specific number as
   unverified secondhand marketing until you've run the comparison
   yourself or found the primary source.** This is exactly the kind of
   claim to be skeptical of in this space - benchmarks that can't name
   their own batch size, concurrency, or hardware SKU precisely aren't
   evidence.

## The decision

None of these are mutually exclusive with each other over a fleet - many
teams run more than one. As a starting point for a single deployment:

**Do you have an NVIDIA GPU at all?**
- **No** (CPU-only, Apple Silicon, edge device, or you want one binary that
  runs anywhere) → **llama.cpp**. It's the only one of the four with
  serious CPU-first and heterogeneous-hardware support as a first-class
  design goal (AVX/AVX2/AVX-512/AMX on x86, RISC-V, Metal/NEON on Apple
  Silicon, Vulkan, SYCL, and more) rather than an afterthought. This is
  also what [`code/07-llama-server-bench`](../code/07-llama-server-bench)
  in this repo actually runs, since this repo has no NVIDIA GPU.
- **Yes** → keep going.

**Do you need to run on non-NVIDIA GPUs too (AMD, TPU, etc.), or want the
broadest model/hardware portability with the largest community?**
- Lean **vLLM**. It has the broadest hardware story of the three GPU-first
  engines (NVIDIA + AMD, plus TPU/Gaudi/Apple-Silicon via plugins) and the
  largest ecosystem ("2,000+ organizations" per its own docs).

**Is squeezing maximum throughput out of NVIDIA hardware specifically -
and are you willing to trade portability and setup complexity for it -
the actual priority?**
- Lean **TensorRT-LLM**. NVIDIA-only, PyTorch-native architecture as of
  its current version, and integrates with Dynamo for disaggregated
  multi-node serving and Triton Inference Server for production model
  management. The steepest setup curve of the four.

**Do you have workloads with heavy shared-prefix traffic (long system
prompts, few-shot templates, agentic tool-use loops with a growing shared
context) or need structured-output-heavy serving?**
- Consider **SGLang** specifically for RadixAttention's prefix-sharing
  design (Stage 2) - this is the one architectural difference among the
  three GPU engines that isn't just "another PagedAttention variant."
  SGLang's own docs claim production use "across over 400,000 GPUs
  worldwide" as of this research - a real-scale deployment, not a toy.

**In every case:** whichever engine you pick, re-run Stage 4's lesson -
check which *kernel* your engine uses for your chosen quantization format
before trusting a throughput number, since kernel choice swung throughput
10x in the one rigorous benchmark this repo's research found (Stage 4).

## Do

Work through [`code/07-llama-server-bench`](../code/07-llama-server-bench):
launch a real OpenAI-compatible server (`llama-server`) with a real
quantized model, hit it with real HTTP requests, and measure real
latency/throughput. This is the one corner of the four-engine comparison
this repo can actually run without an NVIDIA GPU - see
[Stage 0](00-environment.md) for exactly what that means for the other
three. If you have GPU access (rented or owned - Stage 0 has current
pricing), the natural next step is repeating this exact exercise against
vLLM's own quickstart and comparing what changes.

## Checkpoint

- Bench360's headline finding is "no universal best option." Given that,
  what should you actually be measuring on *your* workload before
  committing to an engine, rather than trusting any single published
  benchmark - including the ones in this repo?
- Why is NVIDIA Dynamo not a fifth item in the decision tree above?
- You're deploying on a fleet of CPU-only edge devices with no GPU at all.
  Which two of Stage 4's quantization formats are even usable there, and
  why do the GPU-only formats fail outright rather than just running slower?
- Name the one specific, cited, unverified claim in this stage's reading
  that you should re-derive yourself before repeating it to someone else.

You've completed the curriculum. From here: rent the GPU time (Stage 0)
to run vLLM/TensorRT-LLM/SGLang's own quickstarts against everything
you've built, or go deeper on any single stage's "read" list - most of
them are the tip of a much longer research thread.
