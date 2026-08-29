#!/usr/bin/env python3
"""
Why the KV cache needs its own memory manager: naive contiguous allocation
vs. block-based paging (the mechanism behind vLLM's PagedAttention), plus
prefix sharing across requests with a common system prompt.

No dependencies - pure Python 3, stdlib only. Run it:

    python3 kv_cache_sim.py

Companion to curriculum/02-memory.md.

--------------------------------------------------------------------------
The problem this simulates
--------------------------------------------------------------------------
Every sequence being served needs a KV cache that grows one token at a time,
and you don't know in advance how long it will get. Two ways to handle that:

  NAIVE (contiguous):  reserve a buffer of `max_seq_len` tokens for every
  sequence the moment it's admitted, because the buffer must be contiguous
  in memory and you can't safely predict the real length. This is what
  early serving systems did, and it's simple - and it wastes almost all of
  that memory almost all of the time, because most sequences finish well
  short of the worst case you provisioned for.

  PAGED (block-based): allocate memory in small fixed-size blocks, on
  demand, one block at a time, as the sequence actually grows - exactly
  like an OS paging virtual memory instead of requiring contiguous
  physical RAM per process. A per-sequence block table maps logical
  token positions to physical block IDs. Waste is now bounded to at most
  one partially-filled block per sequence (internal fragmentation),
  instead of the entire unused tail of a `max_seq_len` reservation.

This is the actual mechanism from Kwon et al., "Efficient Memory Management
for Large Language Model Serving with PagedAttention" (SOSP 2023) - see
resources/curated-resources.md for the paper link and vLLM's own docs on it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


BYTES_PER_TOKEN = 512 * 1024  # from code/01-token-cost-model: Llama-2-7B-shaped, all layers


# --------------------------------------------------------------------------
# Naive contiguous allocation
# --------------------------------------------------------------------------

@dataclass
class NaiveAllocator:
    max_seq_len: int

    def memory_for_batch(self, actual_lengths: list[int]) -> int:
        """Every admitted sequence reserves max_seq_len tokens' worth of KV
        cache up front, regardless of how long it actually turns out to be.
        """
        return len(actual_lengths) * self.max_seq_len * BYTES_PER_TOKEN

    def useful_bytes(self, actual_lengths: list[int]) -> int:
        return sum(actual_lengths) * BYTES_PER_TOKEN


# --------------------------------------------------------------------------
# Paged (block-based) allocation, with optional prefix sharing
# --------------------------------------------------------------------------

@dataclass
class Block:
    block_id: int
    ref_count: int = 1


@dataclass
class PagedAllocator:
    block_size: int
    _blocks: dict[int, Block] = field(default_factory=dict)
    _next_id: int = 0

    def _new_block(self) -> Block:
        b = Block(self._next_id)
        self._blocks[b.block_id] = b
        self._next_id += 1
        return b

    def allocate_sequence(self, prompt_len: int, generated_len: int,
                           shared_prefix_blocks: list[Block] | None = None) -> list[Block]:
        """Allocate blocks for one sequence of total length prompt_len +
        generated_len tokens.

        If `shared_prefix_blocks` is given, the sequence reuses those blocks
        (incrementing their ref counts) instead of allocating new ones for
        the tokens they already cover - this is prefix caching: two requests
        with the same system prompt share the same physical KV blocks for
        that shared prefix, copy-on-write once they diverge.
        """
        total_len = prompt_len + generated_len
        table: list[Block] = list(shared_prefix_blocks or [])
        for b in table:
            b.ref_count += 1
        tokens_covered = len(table) * self.block_size
        while tokens_covered < total_len:
            table.append(self._new_block())
            tokens_covered += self.block_size
        return table

    def bytes_allocated(self) -> int:
        """Physical memory actually in use - each unique block counted once,
        no matter how many sequences reference it.
        """
        return len(self._blocks) * self.block_size * BYTES_PER_TOKEN

    def internal_fragmentation_tokens(self, table: list[Block], total_len: int) -> int:
        return len(table) * self.block_size - total_len


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

def simulate_no_sharing(actual_lengths: list[int], max_seq_len: int, block_size: int) -> None:
    naive = NaiveAllocator(max_seq_len)
    naive_bytes = naive.memory_for_batch(actual_lengths)
    useful_bytes = naive.useful_bytes(actual_lengths)

    paged = PagedAllocator(block_size)
    frag_tokens = 0
    for length in actual_lengths:
        table = paged.allocate_sequence(prompt_len=length, generated_len=0)
        frag_tokens += paged.internal_fragmentation_tokens(table, length)
    paged_bytes = paged.bytes_allocated()

    print(f"Workload: {len(actual_lengths)} sequences, actual lengths "
          f"{min(actual_lengths)}-{max(actual_lengths)} tokens "
          f"(reserved worst case: {max_seq_len})")
    print(f"  naive (contiguous, reserve max_seq_len):  "
          f"{naive_bytes / 1e9:.2f} GB reserved, "
          f"{useful_bytes / naive_bytes:.1%} actually used")
    print(f"  paged (block_size={block_size}):          "
          f"{paged_bytes / 1e9:.2f} GB allocated, "
          f"{useful_bytes / paged_bytes:.1%} actually used "
          f"({frag_tokens} tokens of internal fragmentation total)")
    print(f"  memory saved by paging: {(naive_bytes - paged_bytes) / 1e9:.2f} GB "
          f"({(1 - paged_bytes / naive_bytes):.1%})")
    print(f"  -> at the same memory budget, paging fits "
          f"{naive_bytes // paged_bytes if paged_bytes else 0}x as many "
          f"sequences of this shape as naive contiguous allocation.")


def simulate_prefix_sharing(n_sequences: int, shared_prompt_len: int,
                             per_sequence_lengths: list[int], block_size: int) -> None:
    """n_sequences all share the first `shared_prompt_len` tokens (e.g. a
    system prompt / few-shot preamble), then diverge for the rest of their
    length. Compare paged allocation with and without prefix sharing.
    """
    # Without sharing: every sequence gets its own blocks for the shared part too.
    unshared = PagedAllocator(block_size)
    for length in per_sequence_lengths:
        unshared.allocate_sequence(prompt_len=length, generated_len=0)
    unshared_bytes = unshared.bytes_allocated()

    # With sharing: allocate the shared prefix's blocks once, reuse for all.
    shared = PagedAllocator(block_size)
    prefix_table = shared.allocate_sequence(prompt_len=shared_prompt_len, generated_len=0)
    for length in per_sequence_lengths:
        shared.allocate_sequence(
            prompt_len=length, generated_len=0, shared_prefix_blocks=prefix_table,
        )
    shared_bytes = shared.bytes_allocated()

    print(f"Workload: {n_sequences} sequences sharing a {shared_prompt_len}-token "
          f"prefix, diverging lengths {min(per_sequence_lengths)}-"
          f"{max(per_sequence_lengths)}")
    print(f"  paged, no prefix sharing: {unshared_bytes / 1e9:.3f} GB")
    print(f"  paged, with prefix sharing: {shared_bytes / 1e9:.3f} GB")
    print(f"  memory saved by prefix sharing: "
          f"{(1 - shared_bytes / unshared_bytes):.1%}")


def main() -> None:
    random.seed(0)  # deterministic output for a reproducible checkpoint

    print("== Naive contiguous vs. paged allocation, no sharing ==")
    workload = [random.randint(20, 400) for _ in range(64)]
    simulate_no_sharing(workload, max_seq_len=4096, block_size=16)
    print()

    print("== Prefix sharing: a common system prompt across many requests ==")
    per_seq = [random.randint(100, 500) for _ in range(32)]
    simulate_prefix_sharing(
        n_sequences=32, shared_prompt_len=800, per_sequence_lengths=per_seq, block_size=16,
    )
    print()

    # ---- Checkpoint ----
    naive = NaiveAllocator(max_seq_len=4096)
    workload_bytes_naive = naive.memory_for_batch(workload)
    paged = PagedAllocator(block_size=16)
    for length in workload:
        paged.allocate_sequence(prompt_len=length, generated_len=0)
    assert paged.bytes_allocated() < workload_bytes_naive, \
        "paging should always use no more memory than worst-case contiguous reservation"
    assert paged.bytes_allocated() < 0.3 * workload_bytes_naive, \
        "for this workload shape, paging should use well under a third of naive's memory"
    print("CHECKPOINT PASSED: paged allocation uses well under a third of the "
          "memory naive contiguous reservation requires for this workload.")


if __name__ == "__main__":
    main()
