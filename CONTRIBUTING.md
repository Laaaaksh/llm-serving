# Contributing to llm-serving

Thanks for considering a contribution. This is a curriculum plus runnable
samples, open source under the MIT license.

## Getting started

```bash
git clone https://github.com/<your-username>/llm-serving.git   # your fork
cd llm-serving
```

Most of this repo is pure Python 3 with no dependencies - `code/01`, `02`,
`03`, `05`, and `06`'s math script all run with just `python3 <file>.py`.
The GGUF/llama.cpp labs (`code/04`, part of `06`, `07`) need a C/C++
toolchain, CMake, and a Python virtual environment - see
[`curriculum/00-environment.md`](curriculum/00-environment.md) and each
lab's own README for exact setup.

## Contribution workflow

1. Fork the repo, clone your fork (command above).
2. Create a descriptively named branch off `main`.
3. Make focused commits.
4. If you touched a pure-Python sample under `code/`, run it and paste the
   output in the PR - every sample prints a `CHECKPOINT PASSED` line when
   it runs correctly; if it doesn't, something's broken.
5. If you touched the GGUF/llama.cpp labs (`code/04`, `06`, `07`), you must
   have actually run it on real hardware - say what OS/chip/GPU you used,
   and paste your own measured numbers rather than assuming the repo's
   existing numbers still apply after your change.
6. If you touched `resources/curated-resources.md`, open every link you're
   adding or changing and confirm it resolves before submitting, and note
   the resource's own stated date. Never add a link you haven't personally
   opened - this repo's whole reason for existing is that its curation is
   trustworthy.
7. Open a pull request against `main`.

A PR can merge only when CI passes and review feedback is resolved.

## What contributions are useful

- Fixing a wrong claim, a stale "current" statement, or a dead link.
- A new resource or a correction to an existing one in
  `resources/curated-resources.md`, with the date and verdict this repo's
  format expects.
- Real measured numbers from GPU hardware this repo doesn't have access to
  - vLLM/TensorRT-LLM/SGLang runs, multi-GPU tensor-parallelism results, or
  a from-a-different-machine rerun of the GGUF benchmarks in `code/04`,
  `06`, `07`. This repo says plainly where it couldn't verify something
  itself (see [`curriculum/00-environment.md`](curriculum/00-environment.md))
  - closing that gap with real numbers is one of the most valuable things
  a contribution here can do.
- A new sample that isolates one concept the way the existing ones do (see
  "Adding a sample" below) - open an issue first so scope is agreed before
  you write it.
- Curriculum sequencing feedback: if a stage assumes something the
  previous stage didn't actually teach, that's a real bug in a course, not
  a nitpick.

## Adding a sample

Each directory under `code/` demonstrates exactly one idea. Follow the
existing pattern:

- One script (or a small, README-documented set of commands for the
  llama.cpp labs), one `README.md` explaining what it shows, what to look
  for, and how it connects to the curriculum stage that references it.
- Pure-Python samples: stdlib only, no dependencies, and end with a
  `CHECKPOINT PASSED` block that asserts the concept the sample exists to
  demonstrate - if the assertion can fail on a legitimate parameter change,
  say so in a comment (see `code/01-token-cost-model`'s ceiling assertion
  for an example).
- Comments explain *why* a line matters for the concept being taught, not
  what a Python builtin does.
- If it depends on real, dated numbers (a GPU spec, a bits-per-weight
  table, a benchmark result), cite the exact source and fetch date in a
  comment, matching the existing samples.
- Prefer extending an existing stage's "prove it stuck" exercise over
  adding a new curriculum stage - new stages change the sequencing for
  everyone.

## Code style

- Comments explain *why*, not *what* - the reader can read Python syntax,
  they need help with the parts that are non-obvious (why this workload
  shape, why this hardware number, why this formula).
- Match the existing dataclass/typed-function style rather than inventing
  a new one.

## Reporting issues

Open a GitHub issue before starting anything larger than a typo fix, so
scope is agreed first. Use the bug report template for something broken,
and the resource suggestion template for anything about
`resources/curated-resources.md`.
