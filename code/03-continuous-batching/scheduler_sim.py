#!/usr/bin/env python3
"""
Static (request-level) batching vs. continuous (iteration-level) batching,
simulated step-by-step so you can watch head-of-line blocking happen and
measure exactly what it costs.

No dependencies - pure Python 3, stdlib only. Run it:

    python3 scheduler_sim.py

Companion to curriculum/03-throughput.md.

--------------------------------------------------------------------------
The mechanism
--------------------------------------------------------------------------
A server has `capacity` KV-cache slots (Stage 2: this is a real, finite
number set by GPU memory, not an arbitrary knob). Requests arrive over time
and each needs some number of decode steps to finish.

STATIC BATCHING: fill up to `capacity` waiting requests into a batch, then
run that exact set of sequences together, one decode step at a time, until
every sequence in the batch has finished. A sequence that finishes early
just... sits there, occupying a slot, doing nothing, because the fixed-shape
batch can't change composition mid-flight. No new request can be admitted
until the WHOLE batch retires. This is what "batching" meant before 2022.

CONTINUOUS BATCHING (Orca, OSDI 2022; what vLLM/SGLang/TensorRT-LLM all do
today): at every single decode step, any sequence that just finished is
evicted immediately, and if requests are waiting, one is admitted into the
freed slot on the very next step. The batch's membership changes every
step. No slot ever sits idle while a request waits, as long as the queue is
non-empty.

This script simulates both policies over the same synthetic arrival stream
and the same random output lengths, so the only variable is the scheduling
policy - then reports the difference in throughput, latency, and wasted
slot-steps directly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Request:
    req_id: int
    arrival_step: int
    output_len: int  # number of decode steps this request needs
    remaining: int = field(init=False)
    start_step: int | None = None
    finish_step: int | None = None

    def __post_init__(self) -> None:
        self.remaining = self.output_len


def make_workload(n_requests: int, capacity: int, seed: int) -> list[Request]:
    """A bursty arrival stream: requests arrive faster than one batch's
    worth every `capacity` steps (so a queue actually forms), with a mix of
    short replies and a few long completions - the same shape that makes
    static batching's head-of-line blocking visible in code/01's padding-
    waste demo.
    """
    rng = random.Random(seed)
    requests = []
    step = 0
    for i in range(n_requests):
        step += rng.randint(0, max(1, capacity // 4))  # bursty, not evenly spaced
        # 80% short replies, 20% long completions - a realistic chat-workload shape.
        output_len = rng.randint(5, 30) if rng.random() < 0.8 else rng.randint(150, 300)
        requests.append(Request(req_id=i, arrival_step=step, output_len=output_len))
    return requests


# --------------------------------------------------------------------------
# Static (request-level) batching
# --------------------------------------------------------------------------

def simulate_static(requests: list[Request], capacity: int) -> dict:
    queue = sorted(requests, key=lambda r: r.arrival_step)
    pending = list(queue)
    active: list[Request] = []
    step = 0
    idle_slot_steps = 0
    total_steps_run = 0

    while pending or active:
        if not active:
            # Form a new batch: take up to `capacity` requests that have arrived.
            step = max(step, pending[0].arrival_step) if pending else step
            batch, remaining_pending = [], []
            for r in pending:
                if len(batch) < capacity and r.arrival_step <= step:
                    r.start_step = step
                    batch.append(r)
                else:
                    remaining_pending.append(r)
            active = batch
            pending = remaining_pending
            if not active:
                step += 1
                continue

        # Run one decode step for every sequence in the batch, even ones
        # already finished - they occupy a slot but do no useful work.
        # THIS is head-of-line blocking made concrete: idle_slot_steps counts
        # exactly the wasted capacity this causes.
        total_steps_run += 1
        for r in active:
            if r.remaining > 0:
                r.remaining -= 1
                if r.remaining == 0:
                    r.finish_step = step + 1
            else:
                idle_slot_steps += 1  # finished, but still holding the slot
        idle_slot_steps += capacity - len(active)  # batch smaller than capacity
        step += 1

        if all(r.remaining == 0 for r in active):
            active = []  # whole batch retires together - only now can new requests join

    return summarize(requests, capacity, total_steps_run, idle_slot_steps)


# --------------------------------------------------------------------------
# Continuous (iteration-level) batching
# --------------------------------------------------------------------------

def simulate_continuous(requests: list[Request], capacity: int) -> dict:
    pending = sorted(requests, key=lambda r: r.arrival_step)
    active: list[Request] = []
    step = 0
    idle_slot_steps = 0
    total_steps_run = 0

    while pending or active:
        # Admit waiting requests into any free slot, right now, this step.
        while pending and len(active) < capacity and pending[0].arrival_step <= step:
            r = pending.pop(0)
            r.start_step = step
            active.append(r)

        if not active and pending:
            step = pending[0].arrival_step
            continue

        total_steps_run += 1
        still_active = []
        for r in active:
            r.remaining -= 1
            if r.remaining == 0:
                r.finish_step = step + 1  # evicted immediately, slot freed next line
            else:
                still_active.append(r)
        active = still_active
        idle_slot_steps += capacity - len(active)  # only idle if the QUEUE is also empty
        # (a slot below capacity with pending requests waiting gets backfilled
        # at the top of the next iteration - so this only counts genuine idle
        # capacity, not "queue was empty" which isn't the scheduler's fault)
        step += 1

    return summarize(requests, capacity, total_steps_run, idle_slot_steps)


def summarize(requests: list[Request], capacity: int, total_steps_run: int,
              idle_slot_steps: int) -> dict:
    latencies = [r.finish_step - r.arrival_step for r in requests]
    makespan = max(r.finish_step for r in requests)
    total_output_tokens = sum(r.output_len for r in requests)
    return {
        "makespan_steps": makespan,
        "throughput_tokens_per_step": total_output_tokens / makespan,
        "avg_latency_steps": sum(latencies) / len(latencies),
        "p99_latency_steps": sorted(latencies)[int(0.99 * len(latencies))],
        "slot_utilization": 1 - idle_slot_steps / (total_steps_run * capacity),
    }


def main() -> None:
    capacity = 8
    requests_static = make_workload(n_requests=200, capacity=capacity, seed=42)
    requests_continuous = [Request(r.req_id, r.arrival_step, r.output_len) for r in requests_static]

    static_result = simulate_static(requests_static, capacity)
    continuous_result = simulate_continuous(requests_continuous, capacity)

    print(f"Workload: 200 requests, capacity={capacity} slots, "
          f"80% short (5-30 steps) / 20% long (150-300 steps)")
    print()
    print(f"{'metric':<28} {'static':>12} {'continuous':>12}")
    for key, label in [
        ("makespan_steps", "makespan (steps)"),
        ("throughput_tokens_per_step", "throughput (tok/step)"),
        ("avg_latency_steps", "avg latency (steps)"),
        ("p99_latency_steps", "p99 latency (steps)"),
        ("slot_utilization", "slot utilization"),
    ]:
        s, c = static_result[key], continuous_result[key]
        if key == "slot_utilization":
            print(f"{label:<28} {s:>11.1%} {c:>11.1%}")
        else:
            print(f"{label:<28} {s:>12.1f} {c:>12.1f}")

    print()
    speedup = static_result["makespan_steps"] / continuous_result["makespan_steps"]
    latency_improvement = static_result["avg_latency_steps"] / continuous_result["avg_latency_steps"]
    print(f"Continuous batching finishes the same workload {speedup:.2f}x faster "
          f"(fewer total steps) and cuts average latency {latency_improvement:.2f}x, "
          f"using the identical hardware capacity and the identical requests - "
          f"scheduling policy alone is the entire difference.")

    # ---- Checkpoint ----
    assert continuous_result["slot_utilization"] > static_result["slot_utilization"], \
        "continuous batching should always achieve higher or equal slot utilization"
    assert continuous_result["makespan_steps"] <= static_result["makespan_steps"], \
        "continuous batching should never take longer to clear the same workload"
    assert continuous_result["avg_latency_steps"] < static_result["avg_latency_steps"], \
        "continuous batching should reduce average latency on a bursty mixed-length workload"
    print()
    print("CHECKPOINT PASSED: continuous batching strictly improves utilization, "
          "makespan, and average latency over static batching on this workload.")


if __name__ == "__main__":
    main()
