# Curated resources

Every entry here was actually fetched and checked at the time of writing
(August 2026) - not recalled from memory. Dates are the resource's own
stated publish/update date where one exists. Where something has aged,
that's said plainly, along with what to read instead or alongside it.
Curriculum stages link the specific entries relevant to them; this page is
the full, browsable list with the reasoning behind each recommendation.

If a link here breaks, or you find something better, please open an issue
using the "Resource suggestion" template - see
[`CONTRIBUTING.md`](../CONTRIBUTING.md).

## Foundational papers

| Resource | What it covers | Date | Verdict |
|---|---|---|---|
| [Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention"](https://arxiv.org/abs/2309.06180) | Block-based KV-cache allocation; the paper behind vLLM | Submitted 2023-09-12, SOSP 2023 | **Canonical, required.** Not aged out - this is still the primary text for [Stage 2](../curriculum/02-memory.md). |
| [Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models"](https://www.usenix.org/system/files/osdi22-yu.pdf) | Iteration-level scheduling ("continuous batching") + selective batching | OSDI 2022 (Jul 2022) | **Canonical, required.** No arXiv version - USENIX-proceedings-only. Some automated tools get a 403 from USENIX; browsers don't. Primary text for [Stage 3](../curriculum/03-throughput.md). |
| [Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"](https://arxiv.org/abs/2305.13245) | Grouped-Query Attention, the "uptraining" recipe from an existing MHA checkpoint | Submitted 2023-05-22, v3 2023-12-23 | Current, canonical, unsupplanted. |
| [DeepSeek-V2 technical report](https://arxiv.org/abs/2405.04434) | Introduces Multi-head Latent Attention (MLA); 93.3% KV-cache reduction vs. a 67B MHA baseline | 2024-05-07 | Current. The only primary source for MLA alongside DeepSeek-V3, below. |
| [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) | Confirms V3 reuses V2's MLA design, not a new mechanism | 2024-12-27, v2 2025-02-18 | Current. |
| [Frantar et al., "GPTQ"](https://arxiv.org/abs/2210.17323) | One-shot post-training weight quantization via approximate Hessian information | 2022-10-31, ICLR 2023 | Aged-but-foundational. Cite as the origin algorithm, not current best practice - see the GPTQModel note below. |
| [Lin et al., "AWQ"](https://arxiv.org/abs/2306.00978) | Activation-aware weight quantization, protecting salient channels | 2023-06-01, latest rev 2026-04-25, MLSys 2024 Best Paper | Current and still being revised. |
| [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) | Original speculative decoding (Google); exact-distribution proof | 2022-11-30, ICML | Canonical, co-originating paper. |
| [Chen et al., "Accelerating LLM Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) | Independently-developed, concurrent speculative sampling (DeepMind) | 2023-02-02 | Canonical, co-originating paper - a genuinely distinct contribution, not a duplicate of Leviathan et al. |
| [Cai et al., "Medusa"](https://arxiv.org/abs/2401.10774) | Extra decoding heads instead of a separate draft model | 2024 | Aged-but-relevant; the predecessor to the EAGLE lineage. |
| [Li et al., "EAGLE"](https://arxiv.org/abs/2401.15077) | Feature-level (not token-level) autoregression for the draft step | 2024 | Aged-but-relevant; supplanted in SOTA terms by EAGLE-3. |
| [Li et al., "EAGLE-3"](https://arxiv.org/abs/2503.01840) | Direct token prediction + multi-layer feature fusion; current SOTA in this lineage | 2025-03-03, NeurIPS 2025 | Current. Up to 6.5x speedup, ~1.4x over EAGLE-2 (paper's own numbers). |
| [Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs"](https://arxiv.org/abs/2312.07104) | Introduces RadixAttention (radix-tree KV-cache sharing) | 2023-12-12, v2 2024-06-06 | Current, canonical for SGLang's core mechanism. |
| [Wu et al., "Bench360: Benchmarking Local LLM Inference from 360 Degrees"](https://arxiv.org/abs/2511.16682) | Cross-engine, cross-GPU, cross-quantization benchmark measuring task quality AND system metrics together | Submitted 2025-11-12, v2 2026-01-14 | Current. Headline finding - "no universal best option" - is the grounding fact behind [Stage 6](../curriculum/06-choosing-an-engine.md)'s framing. |

## Blog / practitioner explainers

| Resource | What it covers | Date | Verdict |
|---|---|---|---|
| [Kipply, "Transformer Inference Arithmetic"](https://kipp.ly/transformer-inference-arithmetic/) | Arithmetic-intensity/roofline reasoning for prefill vs. decode, from first principles | 2022-03-30 | **Aged-but-best-available.** Its A100-era hardware numbers are stale; its reasoning is not, and nothing equally rigorous and concise has replaced it. This repo's [`code/01-token-cost-model`](../code/01-token-cost-model) follows the same reasoning with numbers checked directly against NVIDIA's current A100 datasheet. |
| [Anyscale, "How continuous batching enables 23x throughput..."](https://www.anyscale.com/blog/continuous-batching-llm-inference) | Plain-language case for continuous batching over static/padded batching | 2023-06-22 | Aged but still the clearest accessible bridge from arithmetic to system design; not supplanted. |
| [vLLM, "Inside vLLM: Anatomy of a High-Throughput LLM Inference System"](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) | Full-system overview: scheduling, PagedAttention, chunked prefill, prefix caching, speculative decoding, multi-GPU | 2025-09-05 | Current, excellent, broad-not-deep by design. Author states it's the first in a planned series - check for later parts. |
| [Enrico Piovano, "vLLM Internals: A Deep Dive"](https://enricopiovano.com/blog/vllm-internals-architecture-deep-dive/) | vLLM's real scheduler internals - three-queue model, admission control, preemption policy, block-pool mechanics | 2025-12-03 | Current, independently authored, technically specific to real V1 code (not just concept) - the best "how does a real scheduler actually work" resource found. |
| [NVIDIA, "Streamlining AI Inference Performance and Deployment with TensorRT-LLM Chunked Prefill"](https://developer.nvidia.com/blog/streamlining-ai-inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-prefill/) | Prefill/decode scheduling conflict and chunked prefill as a fix | 2024-11-15 | Current, vendor-authored, narrower than "Inside vLLM" but solid on its specific topic. |
| [Sebastian Raschka, "Multi-Head Latent Attention (MLA)"](https://sebastianraschka.com/llm-architecture-gallery/mla/) | MLA vs. GQA framing; notes DeepSeek-V2's own ablations found GQA underperforming plain MHA | Undated | Current in substance; no stated publish date, flagged for that reason. |
| [Chris McCormick, "The Inner Workings of Multihead Latent Attention (MLA)"](https://mccormickml.com/2025/04/26/inner-workings-of-mla/) | Deep math treatment of MLA, concrete DeepSeek-V3 dimension numbers | 2025-04-26 | Current and technically strong. **Gap:** doesn't fully work through MLA's decoupled-RoPE trick in the main post - defers to an external notebook. This is the one honest hole in current MLA explainer material. |
| [Geens & Verhelst, "Hardware-Centric Analysis of DeepSeek's Multi-Head Latent Attention"](https://arxiv.org/abs/2506.02523) | MLA from an accelerator-design perspective; shows it shifts attention toward compute-bound | 2025-06-03 | Current, advanced/optional reading. |
| ["Which Quantization Should I Use? A Unified Evaluation of llama.cpp Quantization on Llama-3.1-8B-Instruct"](https://arxiv.org/html/2601.14277v1) | Real measured benchmark of 13 llama.cpp quant configs on GSM8K/HellaSwag/etc. | 2026-01-11 | Current and genuinely empirical, not a listicle. Does not evaluate imatrix quantization - a real gap in the literature, not just here. |
| [theaiengineer.substack.com, "GPTQ vs AWQ vs GGUF: Which 4-Bit to Pick in 2026"](https://theaiengineer.substack.com/p/quantization-in-practice-gptq-vs) | Cites a real Jan 2026 H200/Qwen2.5-32B benchmark across GPTQ/AWQ/GGUF/bitsandbytes | 2026-05-16 | Current. Key finding: inference *kernel* choice (e.g. Marlin) swung AWQ throughput 10x - more than the quantization method itself. |
| [vizuaraai.github.io, inference-engines decision guide](https://vizuaraai.github.io/inference-engineering-visual-guides/visual_walkthroughs/13_inference_engines.html) | A decision-tree for choosing vLLM/SGLang/TensorRT-LLM | Undated | A real prior attempt at this exact question - reasonable as a quick reference, but no llama.cpp branch and no cited benchmark data behind its claims. [Stage 6](../curriculum/06-choosing-an-engine.md) tries to improve on it. |

## Official documentation

| Resource | What it covers | Current as of | Verdict |
|---|---|---|---|
| [vLLM docs](https://docs.vllm.ai/en/latest/) | Full official docs: PagedAttention, continuous batching, prefix caching, quantization, speculative decoding, parallelism | Page dated 2026-04-09 | Current, actively maintained, broadest hardware-portability story of the GPU-first engines. |
| [vLLM, "Automatic Prefix Caching" design doc](https://docs.vllm.ai/en/stable/design/automatic_prefix_caching.html) | Hash-based block cache mechanics for cross-request prefix sharing | Current | Current, official, with a worked example. |
| ~~vLLM, "Paged Attention" design doc~~ | CUDA-kernel-level PagedAttention mechanics | — | **Self-flagged as historical by vLLM itself**: "no longer describes the code used in vLLM today." Useful for original design intuition only, with that caveat stated. |
| [vLLM docs, "Speculative Decoding"](https://docs.vllm.ai/en/latest/features/speculative_decoding/) | Method selection (EAGLE, MTP, n-gram, etc.), config for tuning acceptance-rate behavior | Updated 2026-08-20 | Current, the best practical method-selection resource. |
| [TensorRT-LLM docs](https://nvidia.github.io/TensorRT-LLM/overview.html) | PyTorch-native architecture, parallelism, Dynamo/Triton integration | Page dated 2026-08-27 | Current, very actively maintained, NVIDIA-only. |
| [SGLang docs](https://docs.sglang.io/) | RadixAttention, prefix caching, multi-GPU parallelism | Blog posts on the same site dated as recently as 2026-08-26 | Current. Claims production use "across over 400,000 GPUs worldwide." |
| [llama.cpp README](https://github.com/ggml-org/llama.cpp) | The broadest hardware coverage of the four engines - CPU (AVX/AVX2/AVX-512/AMX, RISC-V), Metal, CUDA, HIP, Vulkan, SYCL, and more | Active, 10,000+ commits | Current, the only one of the four with CPU/edge serving as a first-class goal rather than an afterthought. |
| [llama.cpp imatrix docs](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md) | Importance-matrix quantization: computing and using a calibration-based bias for low-bit quality | Lives on `master`, current | Current, comprehensive. |
| [llama.cpp K-quants explainer, Discussion #2094](https://github.com/ggml-org/llama.cpp/discussions/2094) | Legacy quants (Q4_0 etc.) vs. K-quants (Q4_K_M etc.) naming and a perplexity/size reference table | 2023-07 | **Aged (3 years)** but still the most common informal reference for the naming scheme - a discussion thread, not maintained docs. Pair with the Kurt 2026 benchmark above for current numbers. |
| [Hugging Face, "Quantization" docs](https://huggingface.co/docs/transformers/quantization/overview) | Living comparison table across ~20 quantization backends and their hardware support | Continuously updated | Current - the best single hardware-support matrix; check it directly rather than trusting a static table, since it changes. |
| [NVIDIA A100 datasheet](https://www.nvidia.com/en-us/data-center/a100/) | Peak FLOPs, memory bandwidth, and other spec numbers | Fetched 2026-08-30 | Source for the hardware numbers used in `code/01-token-cost-model`. |
| [NVIDIA Dynamo (ai-dynamo/dynamo)](https://github.com/ai-dynamo/dynamo) | Cluster-level orchestration layer above vLLM/TensorRT-LLM/SGLang - disaggregated prefill/decode, KV-aware routing | "1.0, production-ready" claimed ~March 2026 | Current and real, but operates a layer above the engine-choice decision in [Stage 6](../curriculum/06-choosing-an-engine.md) - relevant for multi-node serving, not a single-node engine choice. |

## Maintained tools and forks worth knowing about

| Resource | What it covers | Current as of | Verdict |
|---|---|---|---|
| [ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel) | Actively maintained GPTQ/AWQ/GGUF/FP8/QQQ quantization, CPU+GPU | Releases through Aug 2026 | **Use this, not `IST-DASLab/gptq`** (dormant since July 2023) - Hugging Face's own docs say it has "fully supplanted AutoGPTQ and AutoAWQ." |
| [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq) | Reference AWQ implementation | Last commit 2025-07-17 (`gh api repos/mit-han-lab/llm-awq/commits`, checked 2026-08-30) | Actively maintained through mid-2025; re-check the repo's own commit history if you need current precision. |

## Adjacent, closely-related repos (not duplicated here)

| Resource | Why it's not duplicated by this repo |
|---|---|
| [skyzh/tiny-llm](https://github.com/skyzh/tiny-llm) (4,500+ stars, pushed 2026-08-29) | A genuinely strong 4-week hands-on "build a tiny vLLM + Qwen" course - attention/RoPE/GQA/KV-cache/quantization/continuous-batching/paged-KV, from scratch, on Apple Silicon/MLX. Its Week 3 in particular covers continuous batching and paged KV as a real implementation exercise. This curriculum points to it ([Stage 2](../curriculum/02-memory.md), [Stage 3](../curriculum/03-throughput.md)) rather than duplicating that build - this repo's own simulators isolate the same mechanisms in far fewer lines, runnable in seconds with no MLX/model dependency, as a faster way to see the concept before or instead of a full from-scratch implementation. |
| [DeepLearning.AI, "Fast & Efficient LLM Inference with vLLM"](https://www.deeplearning.ai/courses/fast-and-efficient-llm-inference-with-vllm) (launched 2026-06-03, with the vLLM project) | Confirmed: 1h38m, 9 videos. Good orientation-level intro; confirmed to skip speculative decoding, TensorRT-LLM, and GGUF entirely. Useful as a pre-Stage-1 motivational primer, not a substitute for any stage here. |

## What this repo chose not to cite, and why

- **A widely-repeated "SGLang ~29% faster than vLLM" benchmark figure** -
  every instance traced back to the same two unverifiable secondary
  sources (see [Stage 6](../curriculum/06-choosing-an-engine.md)). Excluded
  rather than repeated as fact.
- **Several near-identical 2026 "GPTQ vs AWQ vs GGUF" listicle blogs**
  (multiple SEO-pattern domains) that repeat the same numbers without an
  identifiable primary benchmark behind them - excluded in favor of the one
  post that named its actual benchmark source (the substack post above).
- **A "same-model, same-hardware" engine comparison article** that looked
  promising by title but returned an HTTP 403 during this research and
  could not be independently verified - excluded rather than cited
  secondhand.
