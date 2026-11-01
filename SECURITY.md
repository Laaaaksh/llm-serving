# Security Policy

llm-serving is educational docs plus small, self-contained code samples. A
few samples in [`code/`](code) do things a typical "sample repo" doesn't:
they download real model weights from Hugging Face, build and run
[llama.cpp](https://github.com/ggml-org/llama.cpp) locally, and run a local
OpenAI-compatible HTTP server (`llama-server`) that this repo's own client
scripts talk to over `localhost`. That's a bigger surface than a pure-Python
teaching repo, so it's worth being specific about what's in and out of scope.

## What belongs in a report

Worth reporting privately:

- A script in `code/` that sends data anywhere other than the intended local
  process or a documented, named download URL (Hugging Face, GitHub).
- A `download`/`build` script that fetches and executes something without
  saying so, or that fetches from a URL that isn't the documented upstream.
- Anything that would make following this repo's instructions unsafe on a
  reader's own machine beyond the stated risk (downloading model weights,
  building third-party C++ code, running a local server bound to
  `127.0.0.1`).

Not a security issue, just a normal bug report (open a public issue
instead):

- A quantized model producing wrong or low-quality output — that's the
  subject of [Stage 4](curriculum/04-quantization.md), not a vulnerability.
- A simulator script (the pure-Python labs in `code/`) computing the wrong
  number.
- A dead or incorrect link in the curated resources.
- A version mismatch between this repo's pinned llama.cpp/model versions and
  what you have installed.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting:

> https://github.com/Laaaaksh/llm-serving/security/advisories/new

That reaches the maintainer privately so any real issue can be fixed before
it's discussed in public.

## Credits

Reporters who wish to be credited may say so in the private report;
otherwise reports are handled without attribution.
