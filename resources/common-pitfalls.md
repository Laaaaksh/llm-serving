# Common pitfalls

The specific things that stop most people starting this topic - the
misconceptions almost everyone starts with, and the toolchain pain that
wastes time before you get to write anything interesting.

## Misconceptions almost everyone starts with

### "Throughput and latency are basically the same thing"

They trade off against each other, directly, via batching. A bigger batch
raises aggregate throughput (tokens served per second across all requests)
while typically raising *per-request* latency (your one request now shares
the GPU with more neighbors). When you read a benchmark number, always ask
which one it's reporting and at what batch size/concurrency - "vLLM does
8,000 tokens/sec" is meaningless without that context. See
[Stage 1](../curriculum/01-fundamentals.md) and
[Stage 3](../curriculum/03-throughput.md).

### "Quantization is free size reduction"

It isn't - it's a size/speed-for-accuracy trade, and the trade isn't always
worth it (see [Stage 4](../curriculum/04-quantization.md)'s benchmark
citations: Q3_K_S loses real accuracy specifically on math-reasoning
tasks, while barely moving on commonsense tasks). It's also not always a
speed win even when it's a size win: [Stage 4](../curriculum/04-quantization.md)'s
research found kernel choice swinging one quantization format's throughput
10x - a smaller quantized model with the wrong (or default) inference
kernel can be *slower* than a larger one with a well-optimized kernel.
Always check what kernel your serving engine actually uses for your chosen
format before assuming "smaller = faster."

### "More batching is always better"

[Stage 1](../curriculum/01-fundamentals.md)'s cost model shows batching
raises arithmetic intensity toward compute-bound - but only up to a ceiling
set by the KV cache's own memory-bandwidth cost, which grows with the
batch. At long context lengths, that ceiling can sit below what your
hardware needs to become compute-bound, and no amount of additional
batching crosses it. Bigger batches also mean more memory reserved for KV
cache ([Stage 2](../curriculum/02-memory.md)) - past some point you run out
of memory before you run out of upside.

### "A smaller/cheaper draft model always helps speculative decoding"

[Stage 5](../curriculum/05-speculative-decoding.md)'s acceptance-rate math
shows a draft model that's cheap but poorly aligned with the target's
output distribution (low acceptance rate) can produce almost no speedup,
and a draft model that's well-aligned but not actually much cheaper than
the target can make decoding *slower* than not speculating at all. Both
acceptance rate and cost ratio matter; optimizing only one doesn't help.

### "GGUF quant names are just 'how many bits'"

`Q4_K_M` is not literally 4 bits per weight - llama.cpp's own figures put
it at 4.89 bits/weight, because of per-block scale/minimum overhead (see
[Stage 4](../curriculum/04-quantization.md)). The `_S`/`_M`/`_L` suffixes
and the K-quant vs. legacy-quant (`Q4_0`, `Q5_1`, etc.) distinction are
genuinely confusing on first contact - [`llama.cpp` Discussion #2094](https://github.com/ggml-org/llama.cpp/discussions/2094)
is the most commonly cited explainer, and even it's a three-year-old
discussion thread, not maintained documentation. If a size number in a
tutorial doesn't match what you measure, check whether you're comparing
the same exact quant variant, not just the same leading digit.

## Toolchain pain (Stage 4/6's llama.cpp labs)

- **`cmake` isn't preinstalled on macOS** even with Xcode command line
  tools present - `brew install cmake` first, or the build fails
  immediately with "command not found," not a helpful CMake error.
- **The Python conversion script (`convert_hf_to_gguf.py`) has real,
  version-pinned dependencies** (`torch`, `transformers`, `sentencepiece`,
  `gguf`, `protobuf`) - install them in a virtual environment, and use the
  exact versions llama.cpp's own `requirements.txt` at your checked-out
  commit specifies. A newer `transformers` than llama.cpp's converter
  expects is a common source of a confusing model-loading error that looks
  unrelated to the version mismatch that actually caused it.
- **Gated/licensed models need a Hugging Face token.** Some model repos
  require accepting a license on huggingface.co and authenticating
  (`huggingface-cli login` or an `HF_TOKEN` environment variable) before
  `convert_hf_to_gguf.py` can download the weights - a plain 401/403 during
  download, not an obvious "please log in" message.
- **A GGUF binary name or flag from an older tutorial may not exist
  anymore.** As of the commit this repo verified against (see
  [`code/04-gguf-quantize-bench/LLAMA_CPP_COMMIT`](../code/04-gguf-quantize-bench/LLAMA_CPP_COMMIT)),
  most user-facing binaries live under `tools/` (`llama-cli` from
  `tools/cli`, `llama-quantize` from `tools/quantize`, etc.) rather than
  `examples/` - only the speculative-decoding demos still live under
  `examples/speculative*`. If a command from a blog post or older README
  doesn't exist, check the current repo's own directory layout and
  `--help` output rather than assuming your build is broken.
- **`llama-cli` is a chat-style REPL by default now**, not the old
  raw-completion tool - it prints a splash banner and understands
  `/exit`/`/regen`/`/clear` slash commands. There is **no `-no-cnv` flag
  any more** (it was removed); use `-p "<prompt>" -st` for clean
  single-turn output instead of trying to disable conversation mode.
- **Speculative-decoding flags were renamed.** `--draft`/`--draft-n`/
  `--draft-max` no longer exist - the draft model is passed via `-md` /
  `--model-draft`, and you must pass an explicit `--spec-type` (e.g.
  `draft-simple` for a classic small-draft-model setup; other values
  target newer draft mechanisms like EAGLE-3 or MTP that don't apply to a
  plain dense draft/target pair).
- **Speculative decoding needs a matching tokenizer between draft and
  target.** Picking a draft model from a different family/vocab than your
  target either fails outright or silently produces garbage - the two
  models in this repo's demo were deliberately chosen from the same model
  family for this reason.
- **K-quants silently fall back on small/odd-shaped models.** Q4_K_M,
  Q2_K, and the other K-quant formats need tensor dimensions divisible by
  256. A model whose hidden size isn't (Qwen2.5-0.5B's is 896) gets a
  `WARNING: N of M tensor(s) required fallback quantization` and those
  tensors are quantized with legacy (non-K) formats instead - meaning
  "Q4_K_M" doesn't mean a fixed, predictable bits-per-weight for every
  model. See [`code/04-gguf-quantize-bench/README.md`](../code/04-gguf-quantize-bench/README.md)
  for a real example of this happening.
- **Speculative decoding is not a guaranteed speedup - measure it.** This
  repo's own [`code/06-speculative-decoding`](../code/06-speculative-decoding)
  demo measured it *slower* than running the target model alone, on a
  small model pair on Apple Silicon. The technique is real and the papers'
  math is correct; it just doesn't automatically pay off at every model
  size and on every piece of hardware - see that lab's README for why.

## If you're about to reach for a GPU you don't have

Re-read [Stage 0](../curriculum/00-environment.md) before assuming you're
stuck - the pure-Python simulators in Stages 1, 2, 3, and 5, and Stage 4's
GGUF lab, all run without one. The GPU-only gap in this repo is
specifically: running vLLM, TensorRT-LLM, or SGLang yourself, and
reproducing multi-GPU tensor-parallelism numbers. That's real and stated
plainly, not hidden behind a vague "hardware requirements vary."
