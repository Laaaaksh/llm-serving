#!/usr/bin/env bash
# Real speed and accuracy measurements across quant levels: llama-bench
# (tokens/sec) and llama-perplexity (full WikiText-2 test set).
#
# Run build.sh and convert_and_quantize.sh first.
# Companion to curriculum/04-quantization.md.
set -euo pipefail
cd "$(dirname "$0")"
BIN=llama.cpp/build/bin

echo "== llama-bench: prompt-processing + generation tok/s, all 0.5B quant levels =="
"$BIN/llama-bench" \
  -m gguf/qwen2.5-0.5b-instruct-f16.gguf \
  -m gguf/qwen2.5-0.5b-instruct-Q8_0.gguf \
  -m gguf/qwen2.5-0.5b-instruct-Q4_K_M.gguf \
  -m gguf/qwen2.5-0.5b-instruct-Q2_K.gguf \
  -p 512 -n 128

echo
echo "== Perplexity: full WikiText-2 test set, all 0.5B quant levels =="
if [ ! -d wikitext/wikitext-2-raw ]; then
  mkdir -p wikitext
  ( cd wikitext && sh ../llama.cpp/scripts/get-wikitext-2.sh )
fi
for q in f16 Q8_0 Q4_K_M Q2_K; do
  echo "--- $q ---"
  "$BIN/llama-perplexity" -m "gguf/qwen2.5-0.5b-instruct-${q}.gguf" \
    -f wikitext/wikitext-2-raw/wiki.test.raw
done
