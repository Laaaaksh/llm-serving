# Stage 5 — Speculative decoding

**You'll be able to:** explain the draft/target verification mechanism,
compute the expected speedup from a stated acceptance rate and draft cost,
and identify when speculative decoding makes things *worse*, not better.

**Time:** 2–4 hours.

**Build:** [`code/06-speculative-decoding`](../code/06-speculative-decoding) -
the acceptance-rate math (pure Python) and a real demo running two actual
models via llama.cpp.

## Read, in this order

1. **[Leviathan et al., "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192)**
   (Google, 2022-11-30, ICML) and **[Chen et al., "Accelerating Large
   Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318)**
   (DeepMind, 2023-02-02) - two independently-developed, concurrent papers
   that the field credits jointly as co-originators of the technique.
   Leviathan's proof that the sampling scheme exactly preserves the
   target model's output distribution (not an approximation) is the single
   most important thing to take from either paper.
2. **[vLLM docs, "Speculative Decoding"](https://docs.vllm.ai/en/latest/features/speculative_decoding/)**
   (updated 2026-08-20). The best practical "how do I actually pick a
   method" resource: model-based approaches (EAGLE, multi-token prediction,
   external draft models) for the largest gains, versus simpler methods
   (n-gram, suffix matching) for modest gains at near-zero setup cost.
   Includes vLLM's own config knobs for tuning and testing acceptance-rate
   behavior directly.
3. **[Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774)**
   (2024) and **[Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077)**
   (2024) - the "no separate draft model" lineage: extra prediction heads
   or feature-level autoregression on the target model itself, instead of
   running a whole second smaller model. Read these after the draft-model
   approach above, as a contrast, not a replacement for understanding it.
4. **[Li et al., "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840)**
   (2025-03-03, NeurIPS 2025) - current state of the art in this lineage:
   abandons feature-level prediction for direct token prediction with
   multi-layer feature fusion, reporting up to 6.5x speedup, about 1.4x
   over EAGLE-2. Worth knowing this lineage (Medusa → EAGLE → EAGLE-2 →
   EAGLE-3) is still actively moving - vLLM shipped EAGLE-3 support and
   continues building on it, so check vLLM's own speculative-decoding docs
   (above) for what's supported *today* rather than trusting a fixed list.

## Do

1. Run [`code/06-speculative-decoding/acceptance_rate_sim.py`](../code/06-speculative-decoding/acceptance_rate_sim.py)
   first. It implements Leviathan et al.'s own speedup formula and sweeps
   acceptance rate, number of speculated tokens (k), and draft-model cost
   ratio, showing:
   - Speedup rising with acceptance rate, as expected.
   - Speedup peaking at a moderate k and then *declining* for larger k -
     work out why before reading the code's comment explaining it.
   - A case where an expensive-but-convenient draft model makes decoding
     **slower** than not speculating at all - the real-world failure mode
     of picking a draft model for convenience (same family, easy to run)
     over actual cost/alignment.
2. Then follow [`code/06-speculative-decoding/README.md`](../code/06-speculative-decoding/README.md)
   for a real demo: two actual models (a small draft, a larger target from
   the same family, sharing a tokenizer) running together through
   llama.cpp, with measured tokens/sec and (if the tool reports it) a real
   acceptance rate - compare it to what the formula predicts.

## Checkpoint

- Why does speculative decoding preserve the target model's exact output
  distribution instead of just approximating it? (This is the one fact
  from the papers that, if you don't have it, means you don't yet
  understand the technique - re-read Leviathan et al.'s proof sketch if
  you can't answer this cold.)
- Your draft model has a 90% acceptance rate but costs 60% of the target
  model's decode cost. Is speculating worth it? At what draft-cost-ratio
  does it stop being worth it, holding acceptance rate fixed?
- Medusa and EAGLE both avoid running a separate small model. What do they
  need instead that a draft-model approach doesn't?
- Stage 1 established that verifying k+1 token positions in one batched
  forward pass costs "barely more" than verifying one, because decode is
  memory-bound. At what point (how large a k) would that stop being true?

Next: [Stage 6 — choosing and running an engine](06-choosing-an-engine.md).
