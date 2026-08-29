#!/usr/bin/env bash
# Clone and build llama.cpp at the exact commit this lab was verified
# against (see LLAMA_CPP_COMMIT). CPU/Metal only - no NVIDIA GPU needed.
#
# Companion to curriculum/04-quantization.md.
set -euo pipefail
cd "$(dirname "$0")"

COMMIT=$(cat LLAMA_CPP_COMMIT)

if [ ! -d llama.cpp ]; then
  git clone https://github.com/ggml-org/llama.cpp.git
fi
cd llama.cpp
git checkout "$COMMIT"

# -DLLAMA_CURL=OFF: skip llama.cpp's built-in -hf on-the-fly downloader (needs
# curl/OpenSSL wired in) - this lab fetches models ahead of time via
# `hf download` instead (see convert_and_quantize.sh). Either approach works.
CMAKE_ARGS=(-B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF)
if [[ "$(uname)" == "Darwin" ]]; then
  CMAKE_ARGS+=(-DGGML_METAL=ON)
fi

cmake "${CMAKE_ARGS[@]}"
cmake --build build --config Release -j "$(sysctl -n hw.ncpu 2>/dev/null || nproc)" \
  --target llama-cli llama-quantize llama-bench llama-perplexity llama-server \
           llama-speculative-simple

echo
echo "Built at commit $(git rev-parse --short HEAD) ($(git log -1 --format=%ad --date=short))."
echo "Binaries in llama.cpp/build/bin/"
