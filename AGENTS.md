# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- **What this repo is**: a sequenced LLM-inference-serving curriculum (`curriculum/`, 7 stages) plus runnable samples (`code/`) plus honestly-dated resource curation (`resources/`). See the root `README.md` for the pitch and `curriculum/README.md` for the stage table.
- **No NVIDIA GPU in this environment**: built and verified on Apple Silicon macOS, no CUDA. Stages 1/2/3/5's simulators and Stage 4's memory calculator are pure Python stdlib, no dependencies, verified by actually running them (each ends in a `CHECKPOINT PASSED` assertion block). Stage 4/6/7's labs actually build `llama.cpp` and run real small (0.5B-1.5B) models on CPU/Metal - also no NVIDIA GPU needed, but real downloads/build, not simulated. **vLLM, TensorRT-LLM, and SGLang are never run in this repo** - Stage 6 covers all three as a decision guide with cited docs/benchmarks only. Never claim one was run here. See `curriculum/00-environment.md` and each README's "GPU access"/"verified against" language for the exact wording this repo uses about that gap.
- **Citation discipline**: every external link in `curriculum/*.md` and `resources/*.md` must be a URL someone actually fetched and confirmed live - this repo's whole value proposition is that its curation is trustworthy, not recalled from memory. Before adding or changing a citation, fetch the URL yourself. `resources/curated-resources.md`'s closing section ("What this repo chose not to cite, and why") documents claims found during research but deliberately excluded for lacking a verifiable primary source - don't re-add them without doing the verification that was missing.
- **The llama.cpp labs are pinned to an exact commit** - see `code/04-gguf-quantize-bench/LLAMA_CPP_COMMIT` (the CI job `build-llama-cpp` in `.github/workflows/ci.yml` reads the same file). If you re-verify against a newer commit, update that file and re-measure the numbers in `code/04`, `06`, `07`'s READMEs - don't just bump the pin.
- **Adding a sample**: one script (or a documented command sequence for the llama.cpp labs) + `README.md` per directory under `code/`, following the existing numbered naming (`0N-topic-name`). Pure-Python samples are stdlib-only and end with a `CHECKPOINT PASSED` assertion block that numerically demonstrates the concept - match that pattern rather than inventing a new verification style.
- **Sibling repo**: `github.com/Laaaaksh/cuda-engineering` is the first in this series (CUDA kernel fundamentals) - same house style, same author. This repo assumes no CUDA/GPU-programming background; point there for that layer rather than teaching it here.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
