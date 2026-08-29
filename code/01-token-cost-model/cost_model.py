#!/usr/bin/env python3
"""
The cost of one token: why prefill is compute-bound, decode is memory-bound,
and why that gap is the entire reason batching exists.

No dependencies - pure Python 3, stdlib only. Run it:

    python3 cost_model.py

Companion to curriculum/01-fundamentals.md.

--------------------------------------------------------------------------
The model
--------------------------------------------------------------------------
We compute FLOPs and bytes-moved for a Llama-2-7B-shaped dense transformer
(hidden=4096, 32 layers, 32 query heads, 32 KV heads - no GQA in this exact
config, see the note below) doing one forward pass, in two regimes:

  PREFILL - the whole prompt (L tokens) goes through the model in one pass.
  DECODE  - one new token goes through the model, attending to a KV cache
            of everything generated/prompted so far.

Two numbers decide whether a workload is compute-bound or memory-bound on a
given accelerator: how many FLOPs it does, and how many bytes it has to move
to do them. Their ratio is "arithmetic intensity" (FLOPs/byte). Compare that
to the hardware's own FLOPs/byte ratio (peak compute divided by peak memory
bandwidth) and whichever resource runs out first is your bottleneck. This is
the roofline model - see resources/curated-resources.md for where it comes
from.

Hardware numbers used below (A100 80GB SXM, BF16, no sparsity): 312 TFLOP/s
compute, 2039 GB/s memory bandwidth - both from NVIDIA's own A100 datasheet
(nvidia.com/en-us/data-center/a100/, fetched 2026-08-30). Real serving
hardware varies; the point is the *shape* of the result, not these exact
numbers on your GPU.
"""

from __future__ import annotations

from dataclasses import dataclass


# --------------------------------------------------------------------------
# Model + hardware configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    name: str
    hidden_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int  # equals n_heads for plain multi-head attention (no GQA)
    head_dim: int
    intermediate_size: int  # SwiGLU MLP inner dimension
    vocab_size: int
    dtype_bytes: int  # 2 for FP16/BF16

    def attn_params_per_layer(self) -> int:
        q = self.hidden_size * (self.n_heads * self.head_dim)
        k = self.hidden_size * (self.n_kv_heads * self.head_dim)
        v = self.hidden_size * (self.n_kv_heads * self.head_dim)
        o = (self.n_heads * self.head_dim) * self.hidden_size
        return q + k + v + o

    def mlp_params_per_layer(self) -> int:
        # SwiGLU: gate + up + down projections, each hidden <-> intermediate.
        return 3 * self.hidden_size * self.intermediate_size

    def params_per_layer(self) -> int:
        return self.attn_params_per_layer() + self.mlp_params_per_layer()

    def total_params(self) -> int:
        # Embedding + (usually tied) output head add ~2 * vocab * hidden,
        # but that's a one-time lookup/matmul, not repeated per layer - we
        # report it separately rather than folding it into "per-token" cost.
        return self.n_layers * self.params_per_layer()

    def embedding_params(self) -> int:
        return self.vocab_size * self.hidden_size

    def kv_cache_bytes_per_token(self) -> int:
        # Per token, per layer: K and V, each n_kv_heads * head_dim wide.
        per_layer = 2 * self.n_kv_heads * self.head_dim * self.dtype_bytes
        return self.n_layers * per_layer


@dataclass(frozen=True)
class Hardware:
    name: str
    peak_flops_per_sec: float  # dense, matching dtype_bytes above
    mem_bandwidth_bytes_per_sec: float

    def ridge_point(self) -> float:
        """Arithmetic intensity (FLOPs/byte) above which you're compute-bound."""
        return self.peak_flops_per_sec / self.mem_bandwidth_bytes_per_sec


LLAMA2_7B = ModelConfig(
    name="Llama-2-7B-shaped",
    hidden_size=4096,
    n_layers=32,
    n_heads=32,
    n_kv_heads=32,  # plain MHA - see the GQA note in curriculum/02-memory.md
    head_dim=128,
    intermediate_size=11008,
    vocab_size=32000,
    dtype_bytes=2,  # BF16
)

A100_80GB = Hardware(
    name="A100 80GB SXM (BF16, no sparsity)",
    peak_flops_per_sec=312e12,
    mem_bandwidth_bytes_per_sec=2039e9,
)


# --------------------------------------------------------------------------
# Prefill: one pass over L prompt tokens
# --------------------------------------------------------------------------

def prefill_flops(model: ModelConfig, prompt_len: int) -> float:
    """FLOPs for a prefill pass over `prompt_len` tokens.

    Two terms:
      - Weight matmuls: ~2 FLOPs (multiply-add) per parameter, per token.
        Every one of the L tokens goes through every weight once.
      - Attention scores: computing QK^T and applying it to V is
        O(L^2 * hidden) per layer, not O(L) - this is the term that makes
        very long prompts expensive even before you touch a single weight
        matrix, and the reason context-length limits exist independent of
        model size.
    """
    weight_flops = 2 * model.total_params() * prompt_len
    attn_flops = model.n_layers * 4 * (prompt_len ** 2) * model.hidden_size
    return weight_flops + attn_flops


def prefill_bytes_moved(model: ModelConfig, prompt_len: int) -> int:
    """Bytes moved from HBM for a prefill pass.

    The key fact: weights are read from memory ONCE and then reused for all
    L tokens in the batch (they sit in on-chip cache/registers across the
    matmul's inner loop) - the read cost is amortized over L tokens, not
    paid L times. That's what makes prefill compute-bound: numerator (FLOPs)
    grows with L, denominator (bytes) barely does.
    """
    weight_bytes = model.total_params() * model.dtype_bytes
    # Activations moved (a much smaller term than weights for realistic L).
    activation_bytes = prompt_len * model.hidden_size * model.dtype_bytes * 4
    return weight_bytes + activation_bytes


# --------------------------------------------------------------------------
# Decode: one new token, attending to a KV cache of length `ctx_len`
# --------------------------------------------------------------------------

def decode_step_flops(model: ModelConfig, ctx_len: int) -> float:
    weight_flops = 2 * model.total_params() * 1  # one token
    attn_flops = model.n_layers * 4 * ctx_len * model.hidden_size
    return weight_flops + attn_flops


def decode_step_bytes_moved(model: ModelConfig, ctx_len: int, batch_size: int = 1) -> int:
    """Bytes moved from HBM for one decode step.

    This is the crux of the whole curriculum. At batch_size=1, the GPU must
    stream every single weight from HBM to compute one token's worth of
    output - the FLOPs are trivial (one token) but the weights still have to
    be read in full. Nothing amortizes that read, because there's only one
    token to use it for. That is why decode is memory-bandwidth-bound, not
    compute-bound, and why decode throughput barely depends on how fast the
    GPU's compute units are.

    Batching B independent requests together changes this: all B requests
    share the SAME weight read (one pass through the weights serves all B
    sequences at once), so the weight-read cost is amortized over B tokens
    instead of 1. This is the entire mechanical reason batching increases
    throughput - see `batching_arithmetic_intensity` below.
    """
    weight_bytes = model.total_params() * model.dtype_bytes
    kv_bytes = model.kv_cache_bytes_per_token() * ctx_len * batch_size
    return weight_bytes + kv_bytes


# --------------------------------------------------------------------------
# Batching: how arithmetic intensity changes with batch size
# --------------------------------------------------------------------------

def batching_arithmetic_intensity(model: ModelConfig, ctx_len: int, batch_size: int) -> float:
    flops = batch_size * decode_step_flops(model, ctx_len)
    weight_bytes = model.total_params() * model.dtype_bytes  # read ONCE for the whole batch
    kv_bytes = model.kv_cache_bytes_per_token() * ctx_len * batch_size  # read per-sequence
    return flops / (weight_bytes + kv_bytes)


def naive_padded_batch_waste(seq_lens: list[int]) -> float:
    """Fraction of compute wasted by padding every sequence in a batch to the
    longest sequence's length - the failure mode static batching has, and
    that continuous batching (Stage 3) exists to fix.
    """
    if not seq_lens:
        return 0.0
    padded_total = max(seq_lens) * len(seq_lens)
    real_total = sum(seq_lens)
    return (padded_total - real_total) / padded_total


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def main() -> None:
    model, hw = LLAMA2_7B, A100_80GB

    print(f"Model: {model.name}")
    print(f"  non-embedding params: {model.total_params() / 1e9:.2f}B")
    print(f"  embedding params:     {model.embedding_params() / 1e9:.2f}B")
    print(f"  KV cache per token:   {model.kv_cache_bytes_per_token() / 1024:.1f} KiB "
          f"(all {model.n_layers} layers, both K and V)")
    print(f"Hardware: {hw.name}")
    print(f"  ridge point: {hw.ridge_point():.1f} FLOPs/byte "
          f"(above this = compute-bound, below = memory-bound)")
    print()

    print("-- Prefill: arithmetic intensity rises with prompt length --")
    print(f"{'prompt_len':>10} {'FLOPs':>14} {'bytes':>12} {'FLOPs/byte':>12} {'bound':>12}")
    for L in (1, 32, 512, 4096):
        flops = prefill_flops(model, L)
        moved = prefill_bytes_moved(model, L)
        intensity = flops / moved
        bound = "compute" if intensity > hw.ridge_point() else "memory"
        print(f"{L:>10} {flops:>14.3e} {moved:>12.3e} {intensity:>12.1f} {bound:>12}")
    print("  At prompt_len=1, prefill and decode are the same operation -")
    print("  the crossover in 'bound' as prompt_len grows IS the concept.")
    print()

    print("-- Decode at batch_size=1: memory-bound regardless of context length --")
    print(f"{'ctx_len':>10} {'FLOPs':>14} {'bytes':>12} {'FLOPs/byte':>12} {'bound':>12}")
    for ctx in (128, 2048, 32768):
        flops = decode_step_flops(model, ctx)
        moved = decode_step_bytes_moved(model, ctx, batch_size=1)
        intensity = flops / moved
        bound = "compute" if intensity > hw.ridge_point() else "memory"
        print(f"{ctx:>10} {flops:>14.3e} {moved:>12.3e} {intensity:>12.1f} {bound:>12}")
    print()

    print("-- Batching decode: arithmetic intensity climbs toward compute-bound --")
    print("   (short context, ctx_len=128 - see the note below on why context")
    print("   length caps how far batching alone can take you)")
    ctx = 128
    print(f"{'batch_size':>10} {'FLOPs/byte':>12} {'bound':>12}")
    for B in (1, 8, 32, 128, 512, 1024, 2048):
        intensity = batching_arithmetic_intensity(model, ctx, B)
        bound = "compute" if intensity > hw.ridge_point() else "memory"
        print(f"{B:>10} {intensity:>12.1f} {bound:>12}")
    print("  This is the entire mechanical case for batching: one weight read")
    print("  now serves B tokens' worth of compute instead of 1. But notice the")
    print("  ceiling: as batch_size -> infinity, intensity approaches")
    print("  2*params / kv_bytes_per_token(ctx_len), NOT infinity - because KV")
    print("  cache reads scale WITH the batch (each sequence has its own cache)")
    print("  while the weight read doesn't. At long context lengths that ceiling")
    print("  sits below the hardware's ridge point and no amount of batching")
    print("  alone crosses it - this is exactly why KV-cache memory (Stage 2),")
    print("  not just batch size, is the real ceiling on decode throughput.")
    print()

    print("-- Naive static batching: padding waste when request lengths vary --")
    workload = [50, 60, 55, 2000]  # three short requests, one long one
    waste = naive_padded_batch_waste(workload)
    print(f"  requested output lengths: {workload}")
    print(f"  padded-to-longest compute waste: {waste:.1%}")
    print("  One long request in a static batch forces every short request to")
    print("  pay for tokens it never needed - and to wait for the whole batch")
    print("  to finish before the next batch can start. Stage 3 fixes this.")
    print()

    # ---- Checkpoint: numerically demonstrate the two claims this script exists
    # to make, so a broken change to the model above fails loudly.
    assert batching_arithmetic_intensity(model, ctx, 1) < hw.ridge_point(), \
        "expected batch=1 decode to be memory-bound"
    assert batching_arithmetic_intensity(model, ctx, 2048) > hw.ridge_point(), \
        "expected batch=2048 decode at short context to be compute-bound"
    assert batching_arithmetic_intensity(model, 2048, 2048) < hw.ridge_point(), \
        "expected even large-batch decode at long context to stay memory-bound"
    assert prefill_flops(model, 4096) / prefill_bytes_moved(model, 4096) > hw.ridge_point(), \
        "expected long-prompt prefill to be compute-bound"
    print("CHECKPOINT PASSED: batch=1 decode is memory-bound; batch=2048 decode")
    print("at short context crosses to compute-bound; the same batch size at")
    print("long context (2048) does not; long-prompt prefill is compute-bound.")


if __name__ == "__main__":
    main()
