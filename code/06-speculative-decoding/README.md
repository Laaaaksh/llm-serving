# 06 — Speculative decoding: the math, then the real thing

Two parts: [`acceptance_rate_sim.py`](acceptance_rate_sim.py) implements
the speculative-decoding speedup formula from first principles (no
dependencies, run it anywhere) - then
[`run_speculative_demo.sh`](run_speculative_demo.sh) runs two real models
together and shows whether the formula's prediction holds up on real
hardware. **Spoiler, stated honestly rather than hidden: on this hardware
and model pair, it doesn't - speculative decoding measured slower than
just running the target model alone.** That's a real, useful result, not a
failed demo.

Companion to [`curriculum/05-speculative-decoding.md`](../../curriculum/05-speculative-decoding.md).

## Part 1: the math

```bash
python3 acceptance_rate_sim.py
```

See the script's own docstring and the curriculum stage for what to look
for - it sweeps acceptance rate, speculated-token count (k), and draft
cost ratio, and ends with a `CHECKPOINT PASSED` assertion block.

## Part 2: the real demo

Run [`code/04-gguf-quantize-bench`](../04-gguf-quantize-bench)'s
`build.sh` and `convert_and_quantize.sh` first - this demo reuses that
llama.cpp build and those GGUF files (Qwen2.5-0.5B-Instruct as the draft
model, Qwen2.5-1.5B-Instruct as the target - same tokenizer family, which
speculative decoding requires). Then:

```bash
./run_speculative_demo.sh
```

### What was measured

Verified 2026-08-30, llama.cpp commit `c841aeeb8` (see
[`code/04-gguf-quantize-bench/LLAMA_CPP_COMMIT`](../04-gguf-quantize-bench/LLAMA_CPP_COMMIT)),
Apple M5, Metal backend, draft = 0.5B Q8_0, target = 1.5B Q4_K_M,
`llama-speculative-simple --spec-type draft-simple --spec-draft-n-max 7`,
greedy decoding (temp=0):

| Prompt | Target alone (tok/s) | Speculative (tok/s) | Draft-accept rate |
|---|---:|---:|---:|
| Prime-checking function (200 tok) | 77.7 | 60.6 | 72.65% |
| Capital of France (short) | 81.3 | 40.1 | 42.86% |
| Three exercise benefits | 89.9 | 44.6 | 42.86% |

**Speculative decoding was slower than running the target alone in all
three trials**, despite accept rates of 43-73% that
[`acceptance_rate_sim.py`](acceptance_rate_sim.py)'s formula would predict
should give a real speedup. Why: the formula's cost model assumes the
target model's verification pass is "nearly free" because decode is
memory-bandwidth-bound (true, and demonstrated in
[`code/01-token-cost-model`](../01-token-cost-model)) - but it doesn't
account for the *coordination* overhead between two separate models
sharing one Metal command queue, plus CPU-side accept/reject bookkeeping,
on a machine where the target model alone is already fast (77-90 tok/s on
a 1.5B model on Apple Silicon Metal is not the "GPU-bound, painfully slow
decode" regime speculative decoding was designed to fix). **The technique
is real and the math is correct - it just doesn't pay off for a small
on-device model pair on unified-memory hardware where the target is
already fast.** The papers cited in
[`curriculum/05-speculative-decoding.md`](../../curriculum/05-speculative-decoding.md)
report real gains on larger, slower target models (T5-XXL, Chinchilla-70B
in distributed serving) - exactly the regime this repo doesn't have the
hardware to reproduce. If you have GPU access to a multi-billion-parameter
target model, this script gives you a template to measure it yourself.

### The lesson this demo actually teaches

**Don't assume a technique with a correct proof and a good paper behind it
is a free win on your specific hardware and model sizes - measure it.**
That's the whole point of running this yourself instead of reading a
claimed number: [`acceptance_rate_sim.py`](acceptance_rate_sim.py)'s
formula correctly predicts speedup *given* its cost assumptions; this demo
shows a real case where one of those assumptions (verification is "nearly
free") doesn't hold as written once you account for real coordination
overhead between two models.
