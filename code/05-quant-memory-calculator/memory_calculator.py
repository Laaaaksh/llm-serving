#!/usr/bin/env python3
"""
"Will this model, at this quantization, fit in this GPU's memory, at this
batch size and context length?" - the question every quantization decision
actually starts from, made concrete as arithmetic instead of a rule of
thumb.

No dependencies - pure Python 3, stdlib only. Run it:

    python3 memory_calculator.py

Companion to curriculum/04-quantization.md.

--------------------------------------------------------------------------
Bits-per-weight figures
--------------------------------------------------------------------------
GGUF k-quant/i-quant bits-per-weight values below are llama.cpp's own
stated figures (github.com/ggml-org/llama.cpp, tools/quantize/README.md,
fetched 2026-08-30) - these account for the per-block scale/min overhead,
not just the nominal bit width, which is why e.g. "Q4" is really 4.89
bits/weight for Q4_K_M, not a clean 4. GPTQ/AWQ are grouped 4-bit schemes
with their own small per-group overhead (scales + zero-points); the 4.25
bits/weight figure used here for "GPTQ/AWQ 4-bit, group_size=128" is the
standard approximation (4 bits payload + 16-bit scale/zero shared across a
128-weight group: 4 + 32/128 = 4.25) - see resources/curated-resources.md
for the papers this is drawn from.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantFormat:
    name: str
    bits_per_weight: float
    runs_on: str  # short note on hardware support
    note: str


FORMATS = [
    QuantFormat("FP16 / BF16 (no quantization)", 16.0, "any GPU", "the baseline"),
    QuantFormat("INT8 (bitsandbytes / GPTQ / AWQ, group_size=128)", 8.25, "any GPU",
                "8 bits + shared group scale/zero overhead"),
    QuantFormat("GPTQ / AWQ 4-bit, group_size=128", 4.25, "CUDA GPU (needs a quantization-aware kernel)",
                "4 bits + shared group scale/zero overhead"),
    QuantFormat("GGUF Q8_0", 8.5008, "CPU, Apple Metal, CUDA, Vulkan - llama.cpp's backends",
                "near-lossless, still meaningfully smaller than FP16"),
    QuantFormat("GGUF Q6_K", 6.5633, "same as Q8_0", "very close to Q8_0 quality, smaller"),
    QuantFormat("GGUF Q5_K_M", 5.7036, "same as Q8_0", "a common quality/size middle ground"),
    QuantFormat("GGUF Q4_K_M", 4.8944, "same as Q8_0", "the most commonly recommended default"),
    QuantFormat("GGUF Q3_K_M", 3.9960, "same as Q8_0", "visible quality loss on smaller models"),
    QuantFormat("GGUF Q2_K", 3.1593, "same as Q8_0", "aggressive - see Stage 4's real output samples"),
]


def weight_memory_bytes(total_params: int, bits_per_weight: float) -> float:
    return total_params * bits_per_weight / 8


def kv_cache_bytes(n_layers: int, n_kv_heads: int, head_dim: int, kv_dtype_bytes: int,
                    context_len: int, batch_size: int) -> int:
    per_token_per_layer = 2 * n_kv_heads * head_dim * kv_dtype_bytes  # K and V
    return n_layers * per_token_per_layer * context_len * batch_size


def main() -> None:
    # Llama-2-7B-shaped, matching code/01-token-cost-model and code/02-kv-cache-paging.
    total_params = 6_476_005_376  # from code/01-token-cost-model's total_params()
    n_layers, n_kv_heads, head_dim = 32, 32, 128

    print("Model: Llama-2-7B-shaped, 6.48B non-embedding parameters")
    print()
    print(f"{'format':<48} {'weights (GB)':>13}")
    for fmt in FORMATS:
        gb = weight_memory_bytes(total_params, fmt.bits_per_weight) / 1e9
        print(f"{fmt.name:<48} {gb:>13.2f}")
    print()

    # The question that actually matters: does weights + KV cache fit in a
    # specific GPU, at a specific batch size and context length?
    gpu_vram_gb = 24  # a single consumer/workstation-class 24GB card (e.g. RTX 4090/A10)
    batch_size = 8
    context_len = 4096
    kv_gb = kv_cache_bytes(n_layers, n_kv_heads, head_dim, kv_dtype_bytes=2,
                            context_len=context_len, batch_size=batch_size) / 1e9

    print(f"Fitting on a {gpu_vram_gb}GB GPU, batch_size={batch_size}, "
          f"context_len={context_len}:")
    print(f"  KV cache alone needs: {kv_gb:.2f} GB")
    print(f"{'format':<48} {'weights':>9} {'+ KV':>9} {'total':>9} {'fits?':>7}")
    for fmt in FORMATS:
        weights_gb = weight_memory_bytes(total_params, fmt.bits_per_weight) / 1e9
        total_gb = weights_gb + kv_gb
        fits = "yes" if total_gb < gpu_vram_gb else "NO"
        print(f"{fmt.name:<48} {weights_gb:>8.2f}G {kv_gb:>8.2f}G {total_gb:>8.2f}G {fits:>7}")
    print()
    print("Notice: at this batch size and context length, FP16 doesn't fit on a")
    print("24GB card at all - quantizing isn't an optional optimization here,")
    print("it's the difference between serving this batch size and not. That's")
    print("the decision Stage 4 is actually about: not 'is quantization good',")
    print("but 'which format's accuracy loss is worth the memory it buys back,")
    print("for YOUR model/hardware/batch size.' Also notice the KV cache alone")
    print("(computed from Stage 2's own numbers) is bigger than the quantized")
    print("model's weights at every format below FP16 - memory management")
    print("(Stage 2) and quantization (Stage 4) are solving the same problem")
    print("from two different ends.")

    # ---- Checkpoint ----
    fp16_total = weight_memory_bytes(total_params, 16.0) / 1e9 + kv_gb
    q4_total = weight_memory_bytes(total_params, 4.8944) / 1e9 + kv_gb
    assert fp16_total > gpu_vram_gb, "expected FP16 at this batch size to exceed 24GB"
    assert q4_total < gpu_vram_gb, "expected Q4_K_M at this batch size to fit in 24GB"
    print()
    print("CHECKPOINT PASSED: FP16 does not fit this batch size in 24GB; "
          "GGUF Q4_K_M does.")


if __name__ == "__main__":
    main()
