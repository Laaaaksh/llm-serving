#!/usr/bin/env python3
"""
Speculative decoding's speedup math: why it works, and exactly the
conditions under which it stops working.

No dependencies - pure Python 3, stdlib only. Run it:

    python3 acceptance_rate_sim.py

Companion to curriculum/05-speculative-decoding.md. See
code/06-speculative-decoding/README.md for a REAL demo of this running two
actual models via llama.cpp - this script is the math underneath it.

--------------------------------------------------------------------------
The mechanism
--------------------------------------------------------------------------
A small, cheap "draft" model proposes k tokens in a row, autoregressively.
Then the big "target" model checks all k proposed tokens in a SINGLE
forward pass (it can, because checking token i only needs tokens 0..i-1,
which are all already fixed by the draft - no need to run the target
model k separate times). Tokens are accepted left-to-right via rejection
sampling until the first one the target model would not have produced
itself; that token is discarded and replaced with a fresh sample from the
target's own (corrected) distribution, and generation continues from there.
This is provably equivalent in output distribution to running the target
model alone, token by token - it's a sampling trick, not an approximation.
See Leviathan et al., "Fast Inference from Transformers via Speculative
Decoding" (arXiv:2211.17192) for the proof; resources/curated-resources.md
has the link.

The speedup comes from one fact you already derived in
code/01-token-cost-model: a single decode step is memory-bandwidth-bound,
which means verifying k+1 token positions in one batched forward pass costs
barely more than verifying 1 - the weight read dominates either way. So one
target-model call, that used to produce exactly 1 token, now produces
(on average) more than 1 token, for close to the same cost.

--------------------------------------------------------------------------
The formula
--------------------------------------------------------------------------
Let alpha = the probability the draft model's token matches what the target
model would have sampled (the "acceptance rate" - depends entirely on how
similar the draft and target's output distributions are, which is a
property of how the draft model was trained/chosen, not a tunable knob).
Let k = number of tokens speculated per round.

Expected tokens produced per round (accepted tokens + 1 bonus corrective
token; Leviathan et al., Theorem 1):

    E[tokens] = (1 - alpha^(k+1)) / (1 - alpha)      (alpha < 1)
    E[tokens] = k + 1                                 (alpha == 1, the limit)

Cost per round, in units of "one target-model decode step":

    cost = 1 + k * draft_cost_ratio

where draft_cost_ratio is the draft model's decode-step cost relative to
the target's (roughly proportional to parameter count at batch=1, since
decode cost is memory-bound - see code/01). The "1" is the target's single
verification pass; because that pass is memory-bound, it costs about the
same as generating one token normally, REGARDLESS of k, for the k values
actually used in practice (k=3-8ish - very long k eventually stops being
free, but this script's range is realistic).

Speedup vs. plain autoregressive decoding (1 token per 1 unit of target
cost) = E[tokens] / cost.
"""

from __future__ import annotations


def expected_tokens_per_round(alpha: float, k: int) -> float:
    if alpha >= 1.0:
        return k + 1
    return (1 - alpha ** (k + 1)) / (1 - alpha)


def cost_per_round(k: int, draft_cost_ratio: float) -> float:
    return 1 + k * draft_cost_ratio


def speedup(alpha: float, k: int, draft_cost_ratio: float) -> float:
    return expected_tokens_per_round(alpha, k) / cost_per_round(k, draft_cost_ratio)


def main() -> None:
    print("-- Speedup vs. acceptance rate, at a fixed draft cost ratio --")
    print("   (draft_cost_ratio=0.15: a draft model ~1/7th the target's decode")
    print("   cost - roughly a 1B draft paired with a 7B target)")
    draft_cost_ratio = 0.15
    k = 5
    print(f"{'alpha':>8} {'E[tokens/round]':>16} {'speedup':>10}")
    for alpha in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        e_tok = expected_tokens_per_round(alpha, k)
        sp = speedup(alpha, k, draft_cost_ratio)
        print(f"{alpha:>8.2f} {e_tok:>16.2f} {sp:>9.2f}x")
    print("  Below roughly alpha=0.6-0.7 here, speculative decoding is barely")
    print("  worth it even with a very cheap draft - a mismatched or badly")
    print("  chosen draft model doesn't just help less, it can make things")
    print("  worse than not speculating at all (see the alpha=0.5 row: only")
    print("  a modest speedup despite k=5 proposed tokens per round).")
    print()

    print("-- Speedup vs. k, at a fixed realistic acceptance rate --")
    print("   (alpha=0.75: a decent but imperfect draft model)")
    alpha = 0.75
    print(f"{'k':>4} {'E[tokens/round]':>16} {'speedup':>10}")
    for k in (1, 2, 3, 5, 8, 12, 20):
        e_tok = expected_tokens_per_round(alpha, k)
        sp = speedup(alpha, k, draft_cost_ratio)
        print(f"{k:>4} {e_tok:>16.2f} {sp:>9.2f}x")
    print("  Speedup keeps climbing more slowly as k grows, then turns over -")
    print("  each additional speculated token still costs draft_cost_ratio,")
    print("  but the marginal probability that ALL of them are still being")
    print("  accepted (alpha^k) keeps shrinking. There's an optimal k for a")
    print("  given alpha, and it's usually well under 20.")
    print()

    print("-- When speculative decoding actively hurts: an expensive draft --")
    print("   (alpha=0.75, but draft_cost_ratio=0.6 - a draft model that's")
    print("   not actually much cheaper than the target)")
    expensive_draft_ratio = 0.6
    for k in (1, 3, 5, 8):
        sp = speedup(alpha, k, expensive_draft_ratio)
        verdict = "still helps" if sp > 1.0 else "SLOWER than not speculating"
        print(f"  k={k}: {sp:.2f}x -> {verdict}")
    print("  This is the real-world failure mode: picking a draft model that's")
    print("  convenient (same family, easy to run) rather than one that's both")
    print("  cheap AND well-aligned with the target's output distribution.")

    # ---- Checkpoint ----
    cheap_draft_speedup = speedup(alpha=0.75, k=5, draft_cost_ratio=0.15)
    expensive_draft_speedup = speedup(alpha=0.75, k=5, draft_cost_ratio=0.6)
    assert cheap_draft_speedup > 1.0, "a cheap, decent draft model should speed things up"
    assert expensive_draft_speedup < 1.0, "an expensive draft model should slow things down here"
    assert speedup(0.99, 5, 0.15) > speedup(0.5, 5, 0.15), \
        "higher acceptance rate should always speed things up more"
    print()
    print("CHECKPOINT PASSED: a cheap well-aligned draft speeds up decoding; "
          "an expensive draft at the same acceptance rate slows it down; "
          "speedup increases monotonically with acceptance rate.")


if __name__ == "__main__":
    main()
