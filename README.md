<div align="center">

# llm-serving

**A sequenced path from "I can call an LLM API" to "I can reason about
continuous batching, KV-cache paging, and which quantization format to
ship" - with runnable code at every step.**

[![CI](https://github.com/Laaaaksh/llm-serving/actions/workflows/ci.yml/badge.svg)](https://github.com/Laaaaksh/llm-serving/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)](curriculum/00-environment.md)

**[Curriculum](curriculum/README.md) • [Code samples](code/README.md) • [Resources](resources/curated-resources.md) • [Common pitfalls](resources/common-pitfalls.md) • [Contributing](CONTRIBUTING.md) • [License](LICENSE)**

</div>

## What this is

LLM inference serving has no shortage of material - vLLM's own excellent
blog posts, the PagedAttention and Orca papers, a hundred "GPTQ vs AWQ vs
GGUF" listicles, vendor docs for four different serving engines. What it
lacks is a **path**: something that tells you what order to read things in,
which claims are measured and which are marketing, and what to actually
build to check a concept landed instead of just feeling like it did. An
unordered list of links is the problem this repo exists not to be.

This repo is seven sequenced stages, each with:

- **What you'll be able to do** at the end of it, stated concretely.
- **What to read**, in order - a short, deliberately curated list, with the
  full reasoning (what's covered well, how current it is, what's aged but
  still the best available) in
  [`resources/curated-resources.md`](resources/curated-resources.md).
- **A small, runnable program to build and run**, in [`code/`](code): a
  from-scratch cost model showing why decode is memory-bound and prefill
  is compute-bound, a KV-cache paging simulator that measures PagedAttention's
  real memory savings, a continuous-batching scheduler you watch beat
  static batching on the same workload, a real GGUF quantization lab you
  run on your own machine, and a speculative-decoding acceptance-rate
  calculator that shows exactly when it backfires.
- **Checkpoint questions** to answer before moving on.

Start at [`curriculum/README.md`](curriculum/README.md).

## What this doesn't cover

This is about the *serving system's* behavior - scheduling, memory
management, quantization, engine choice - not about training, fine-tuning,
or writing GPU kernels by hand. It doesn't cover parameter-efficient
fine-tuning (LoRA/QLoRA/DPO - see
[`huggingface/smol-course`](https://github.com/huggingface/smol-course) for
that), and it doesn't teach CUDA kernel programming (see this repo's sibling,
[`cuda-engineering`](https://github.com/Laaaaksh/cuda-engineering), for
that). [Stage 6](curriculum/06-choosing-an-engine.md) covers multi-GPU
serving and disaggregated prefill/decode as a decision-guide topic, but
this repo does not have the hardware to run either.

## GPU access — read this before anything else

**This repository was built and every sample in it verified without an
NVIDIA GPU** - on Apple Silicon (macOS), no CUDA. That's a real constraint,
stated plainly instead of hidden:

- Stages 1, 2, 3, and 5's simulators, and Stage 4's memory calculator, are
  pure Python with zero dependencies. They run identically on any machine,
  including yours right now, with no GPU of any kind.
- Stage 4's GGUF quantization lab and Stage 6's capstone server actually
  build [llama.cpp](https://github.com/ggml-org/llama.cpp) and run real
  small (0.5B-1.5B parameter) models, with real measured numbers - on CPU
  or Apple Silicon, still no NVIDIA GPU required.
- **Stage 6's engine decision guide covers vLLM, TensorRT-LLM, and SGLang
  in depth, with cited official docs and benchmark data - but none of the
  three was run inside this repository.** If you want to reproduce their
  own numbers, you need actual NVIDIA GPU access; [Stage
  0](curriculum/00-environment.md) has current hourly rental pricing,
  checked directly against provider pricing pages at the time of writing.

Full detail in [`curriculum/00-environment.md`](curriculum/00-environment.md).

## Repository layout

```
curriculum/   7 sequenced stages - the path itself
code/         7 runnable samples, one per concept - pure Python where possible, real llama.cpp labs where it matters
resources/    honest, dated curation of everything external cited above
```

## Contributing

Contributions are welcome - a wrong claim, a stale "current" statement, a
dead link, or real GPU numbers this repo couldn't measure itself (vLLM,
TensorRT-LLM, SGLang, multi-GPU). See [CONTRIBUTING.md](CONTRIBUTING.md).
Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) first.

## Security

Found a security issue? See [SECURITY.md](SECURITY.md) - please don't open
a public issue for it.

## License

MIT - see [LICENSE](LICENSE).
