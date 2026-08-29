# Stage 3 — Throughput: continuous batching & scheduler internals

**You'll be able to:** explain iteration-level (continuous) scheduling as
opposed to request-level (static) batching, and describe what a real
scheduler actually tracks - queues, admission control, eviction/preemption
- rather than just "it batches continuously."

**Time:** 3–4 hours.

**Build:** [`code/03-continuous-batching`](../code/03-continuous-batching).

## Read, in this order

1. **[Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models"](https://www.usenix.org/system/files/osdi22-yu.pdf)**
   (OSDI 2022). The paper that invented what the industry now calls
   continuous batching, under the name "iteration-level scheduling," plus
   "selective batching" (batch only the operations that tolerate different
   sequence lengths; keep attention un-batched across them). Reported a
   36.9x throughput improvement over NVIDIA FasterTransformer on GPT-3 175B
   at matched latency. No arXiv version exists - it's OSDI-proceedings-only
   at the link above. (Some automated fetchers get blocked by USENIX with
   an HTTP 403; a browser has no such problem.)
2. **[Anyscale, "How continuous batching enables 23x throughput..."](https://www.anyscale.com/blog/continuous-batching-llm-inference)**
   (2023-06-22) - if you skipped it in Stage 1, read it now. It's the
   clearest plain-language bridge between Orca's paper and an intuitive
   mental model.
3. **[Enrico Piovano, "vLLM Internals: A Deep Dive into the Architecture of High-Performance LLM Inference"](https://enricopiovano.com/blog/vllm-internals-architecture-deep-dive/)**
   (2025-12-03). This is the resource that actually answers "how does a
   real scheduler decide what to admit and evict each step" - vLLM's
   three-queue model (waiting / running / swapped), admission-control
   thresholds for interleaving long prefills with ongoing decodes,
   preemption policy (recompute vs. swap on eviction), and the free-block
   pool's O(1) allocation. Independently authored, current, and
   specifically about real production code rather than the general concept.
4. **[vLLM, "Inside vLLM"](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)**
   (from Stage 1) - re-read the scheduling section now that you have Orca's
   vocabulary; it'll land differently.

## Do

Run [`code/03-continuous-batching/scheduler_sim.py`](../code/03-continuous-batching/scheduler_sim.py).
It implements both policies from scratch against the identical synthetic
workload (a Poisson-ish bursty arrival stream, 80% short replies / 20% long
completions) and a fixed capacity of KV-cache "slots":

- **Static (request-level) batching**: fill up to `capacity` requests, run
  them together until every one finishes, and don't admit anything new
  until the whole batch retires - even sequences that finished early just
  sit there holding a slot.
- **Continuous (iteration-level) batching**: evict a finished sequence and
  backfill a waiting one on the very next step, every step.

The script measures makespan, throughput, average and p99 latency, and
slot utilization for both, on the same requests, and prints the ratio.
Watch the slot-utilization number specifically - that's head-of-line
blocking made into an actual percentage instead of a claim.

## Going deeper: building continuous batching and paged KV yourself

This script isolates the scheduling *policy* in under 150 lines so you can
see the mechanism in seconds, without a GPU, model, or ML framework. If you
want to build a real continuous-batching engine over an actual model's
forward pass next - not a simulation of the policy, the real thing -
[`skyzh/tiny-llm`](https://github.com/skyzh/tiny-llm)'s Week 3 ("Build a
Mini vLLM") does exactly that: continuous batching and chunked admission,
then paged KV as the canonical serving layout, on Apple Silicon/MLX. It's
actively maintained (pushed the day before this repo's research was done)
and directly on-topic. This curriculum doesn't duplicate that exercise -
it gives you the conceptual scaffolding (Orca, the scheduler-internals
reading above) to get more out of it when you do it.

## Checkpoint

- In the static-batching simulation, why does slot utilization stay well
  under 100% even though the workload has plenty of waiting requests?
- A batch of 8 static-batched requests contains one request needing 300
  decode steps and seven needing 20. Roughly how many of the 8 slots are
  sitting idle-but-occupied for most of the batch's lifetime?
- What real resource does a "slot" correspond to in an actual serving
  system (tie this back to Stage 2)? What happens to continuous batching's
  admission decision when that resource runs out mid-request, rather than
  before a request is ever admitted?
- Orca's paper predates PagedAttention by about a year. What problem does
  Orca's iteration-level scheduling *not* solve on its own, that
  PagedAttention (Stage 2) had to fix separately?

Next: [Stage 4 — quantization](04-quantization.md).
