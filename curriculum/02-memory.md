# Stage 2 — Memory: KV-cache math, PagedAttention, prefix caching, GQA/MLA

**You'll be able to:** compute how big a KV cache actually gets, explain
why naive contiguous allocation wastes almost all of it, explain
PagedAttention's block-based fix and prefix caching's extension of it, and
explain the two real levers (GQA, MLA) model architects use to shrink the
cache itself rather than just manage it better.

**Time:** 3–5 hours.

**Build:** [`code/02-kv-cache-paging`](../code/02-kv-cache-paging).

## Read, in this order

1. **[Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention"](https://arxiv.org/abs/2309.06180)**
   (SOSP 2023) - the full paper this time. This is *the* paper for this
   stage, not a supplementary read. It frames continuous batching's memory
   fragmentation problem and introduces block-based KV-cache allocation,
   modeled directly on OS virtual memory paging.
2. **[vLLM docs, "Automatic Prefix Caching"](https://docs.vllm.ai/en/stable/design/automatic_prefix_caching.html)**
   (vLLM Project). Extends PagedAttention's blocks with content hashing so
   requests sharing a prefix (a system prompt, a few-shot preamble) share
   the same physical blocks. Current, official, includes a worked example.
3. **[SGLang paper, "SGLang: Efficient Execution of Structured Language Model Programs"](https://arxiv.org/abs/2312.07104)**
   (Zheng et al., 2023-12-12, revised 2024-06-06) - introduces RadixAttention,
   a radix-tree-based alternative to vLLM's hash-based block cache for the
   same prefix-sharing problem. For the practical mechanics (eviction
   policies, page-aligned caching), see SGLang's own concepts page - at the
   time of writing the most complete version of it lives at a preview-style
   URL (`sgl-project-sglang-93.mintlify.app/concepts/radix-attention`) that
   may move; if it's dead, search SGLang's current docs site for
   "RadixAttention" rather than assuming the mechanism itself changed.
   Comparing this to vLLM's approach is a genuinely useful exercise: two
   different data structures solving the identical problem.
4. **[Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"](https://arxiv.org/abs/2305.13245)**
   (Google Research, 2023-05-22, v3 2023-12-23). GQA reduces the KV cache
   by cutting the *number* of key/value heads while keeping all query
   heads - an interpolation between full multi-head attention and
   multi-query attention. Includes the "uptraining" recipe for converting
   an existing MHA checkpoint using ~5% of original pretraining compute.
5. **DeepSeek-V2** ([arXiv:2405.04434](https://arxiv.org/abs/2405.04434),
   2024-05-07) **and DeepSeek-V3** ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437),
   2024-12-27) technical reports, §2 (architecture), for Multi-head Latent
   Attention (MLA) - the other lever: instead of storing fewer K/V heads
   (GQA's approach), MLA compresses what's stored per token into a low-rank
   latent vector, reconstructed at attention time. DeepSeek-V2 reports a
   93.3% KV-cache reduction versus a 67B dense MHA baseline. These are the
   only primary sources - there's no separate "MLA paper."
6. For MLA's mechanics explained rather than just stated: **[Sebastian Raschka, "Multi-Head Latent Attention (MLA)"](https://sebastianraschka.com/llm-architecture-gallery/mla/)**
   and **[Chris McCormick, "The Inner Workings of Multihead Latent Attention (MLA)"](https://mccormickml.com/2025/04/26/inner-workings-of-mla/)**
   (2025-04-26). Raschka's page notes DeepSeek-V2's own ablations found GQA
   *underperforming* plain MHA while MLA matched or beat it - and that MLA
   likely only pays off at 100B+ parameter scale, making GQA the more
   practical choice below that. McCormick's post has the clearest math, with
   one honest gap: it doesn't fully work through MLA's "decoupled RoPE"
   trick in the main post, deferring that to a linked external notebook -
   if you want that specific piece, that's where to look.

## Do

Run [`code/02-kv-cache-paging/kv_cache_sim.py`](../code/02-kv-cache-paging/kv_cache_sim.py).
It builds both allocation strategies from scratch - a naive one that
reserves `max_seq_len` per sequence up front, and a paged one that
allocates fixed-size blocks on demand - runs the same synthetic workload
through both, and reports real numbers:

- How much memory naive reservation wastes when actual sequence lengths
  vary (usually >90% waste against a generous worst-case bound).
- How many more concurrent sequences the exact same memory budget supports
  under paging.
- How much further prefix sharing saves when many sequences share a
  common system prompt.

Then change the workload (make lengths less variable, make the shared
prefix longer or shorter) and watch which number moves and by how much.

## A note on what's simulated vs. real here

This script is a from-scratch, pure-Python model of the *mechanism* -
block allocation, reference counting, prefix sharing - not vLLM's or
SGLang's actual code. It gets you to the same conclusion the papers state
(paging + sharing dramatically cuts wasted memory) with numbers you compute
yourself instead of numbers you're asked to trust. If you want a from-
scratch *implementation* exercise that goes further (a real paged-KV
attention kernel, not just the allocator), [`skyzh/tiny-llm`](https://github.com/skyzh/tiny-llm)
(Alex Chi; 4,500+ stars; pushed 2026-08-29, actively maintained) builds
exactly that in its Week 2-3 material, on Apple Silicon/MLX - a strong
next step after this stage, not a replacement for it.

## Checkpoint

- A model has 32 layers, 8 KV heads (GQA), head_dim 128, and you're
  storing K and V in FP16. How many bytes does one token's KV cache take,
  and how does that number change if you double the KV head count back to
  32 (plain MHA)?
- Why does PagedAttention bound waste to "at most one block per sequence"
  instead of eliminating it entirely? What controls the size of that bound?
- vLLM's hash-based block cache and SGLang's radix tree solve the same
  prefix-sharing problem with different data structures. Name one
  situation where the two would behave differently (hint: what happens
  when two requests share a *partial* prefix that ends mid-block?).
- GQA and MLA both shrink the KV cache, but by different mechanisms. Which
  one changes how many separate K/V projections exist, and which one
  changes what's stored per token *within* an existing projection?

Next: [Stage 3 — throughput](03-throughput.md).
