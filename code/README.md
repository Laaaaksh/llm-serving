# Code samples

Seven runnable samples, each isolating one concept. Five are pure Python
with zero dependencies - they run on anything. Two do real, hands-on work
with [llama.cpp](https://github.com/ggml-org/llama.cpp): building it,
quantizing a real model, and serving it - CPU/Apple-Silicon only, no
NVIDIA GPU required, but real downloads and a real build, not a
simulation.

| # | Sample | Idea | Needs |
|---|--------|------|-------|
| [01](01-token-cost-model) | `cost_model.py` | Prefill is compute-bound, decode is memory-bound - and why batching has a ceiling | Python only |
| [02](02-kv-cache-paging) | `kv_cache_sim.py` | PagedAttention's real memory savings over naive contiguous allocation | Python only |
| [03](03-continuous-batching) | `scheduler_sim.py` | Continuous batching beats static batching, measured, not asserted | Python only |
| [04](04-gguf-quantize-bench) | build/quantize/benchmark scripts | Real GGUF quantization: measured size, speed, and output quality at each level | llama.cpp build, ~10GB disk |
| [05](05-quant-memory-calculator) | `memory_calculator.py` | Does this model, at this quantization, fit on this GPU at this batch size | Python only |
| [06](06-speculative-decoding) | `acceptance_rate_sim.py` + a real demo | The speculative-decoding speedup formula, then two real models proving (or disproving) it | Python only for the math; llama.cpp for the demo |
| [07](07-llama-server-bench) | server + client scripts | A real OpenAI-compatible serving stack, actually running, actually benchmarked | llama.cpp build |

Each pairs with a stage in [`curriculum/`](../curriculum) - see the
"Companion to" link at the top of each sample's README.

## Running the pure-Python samples

```bash
cd code/0N-sample-name
python3 <script>.py
```

Each ends with a `CHECKPOINT PASSED` line that asserts the specific
numeric claim the sample exists to demonstrate - if you change the model
or workload parameters and an assertion fails, that's telling you
something true, not that the script is broken (each assertion's comment
explains what condition it depends on).

## Running the llama.cpp labs

See [`curriculum/00-environment.md`](../curriculum/00-environment.md) for
setup (a C/C++ toolchain, CMake, a Python virtual environment - no NVIDIA
GPU) and each lab's own README for exact commands. These labs actually
download real model weights and actually build and run real inference -
budget real wall-clock time and disk space, not the seconds the pure-Python
samples take.

## GPU access

**These samples were written and verified in an environment with no
NVIDIA GPU attached** - Apple Silicon macOS. The pure-Python samples don't
need one; the llama.cpp labs run real inference on CPU/Metal, also without
one. What genuinely isn't verified anywhere in this repo: running vLLM,
TensorRT-LLM, or SGLang, or any multi-GPU configuration - see
[`curriculum/06-choosing-an-engine.md`](../curriculum/06-choosing-an-engine.md)
and [`curriculum/00-environment.md`](../curriculum/00-environment.md) for
exactly what that means and what it costs to verify yourself.
