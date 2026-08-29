#!/usr/bin/env bash
# Run the same two prompts through every quant level and print the actual
# output text side by side - so quantization's effect on output quality is
# something you read, not something you're told.
#
# Run build.sh and convert_and_quantize.sh first.
# Companion to curriculum/04-quantization.md.
set -euo pipefail
cd "$(dirname "$0")"
BIN=llama.cpp/build/bin

PROMPT_A="What is the capital of France?"
PROMPT_B="A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left? Think step by step then give the final answer."

for q in f16 Q8_0 Q4_K_M Q2_K; do
  echo "=================== $q ==================="
  echo "--- Prompt A: $PROMPT_A ---"
  "$BIN/llama-cli" -m "gguf/qwen2.5-0.5b-instruct-${q}.gguf" \
    -p "$PROMPT_A" -n 128 --temp 0 --seed 42 -st 2>/dev/null
  echo
  echo "--- Prompt B: $PROMPT_B ---"
  "$BIN/llama-cli" -m "gguf/qwen2.5-0.5b-instruct-${q}.gguf" \
    -p "$PROMPT_B" -n 200 --temp 0 --seed 42 -st 2>/dev/null
  echo
done
