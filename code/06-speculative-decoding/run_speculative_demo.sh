#!/usr/bin/env bash
# A real speculative-decoding demo: Qwen2.5-0.5B-Instruct (draft) proposing
# tokens for Qwen2.5-1.5B-Instruct (target) to verify, via llama.cpp's
# llama-speculative-simple - compared against the target model running
# alone on the identical prompts.
#
# Run code/04-gguf-quantize-bench's build.sh and convert_and_quantize.sh
# first - this script reuses that build and those GGUF files.
# Companion to curriculum/05-speculative-decoding.md.
set -euo pipefail
cd "$(dirname "$0")"

LAB=../04-gguf-quantize-bench
BIN="$LAB/llama.cpp/build/bin"
TARGET="$LAB/gguf/qwen2.5-1.5b-instruct-Q4_K_M.gguf"
DRAFT="$LAB/gguf/qwen2.5-0.5b-instruct-Q8_0.gguf"

if [ ! -f "$TARGET" ] || [ ! -f "$DRAFT" ]; then
  echo "Missing GGUF files - run code/04-gguf-quantize-bench's build.sh and" >&2
  echo "convert_and_quantize.sh first." >&2
  exit 1
fi

run_case() {
  local prompt="$1" n="$2"
  echo "=================================================================="
  echo "Prompt: $prompt"
  echo "--- target alone (baseline) ---"
  "$BIN/llama-cli" -m "$TARGET" -p "$prompt" -n "$n" --temp 0 --seed 42 -st 2>&1 \
    | grep -E "eval time|tokens per second" || true
  echo "--- speculative decoding (draft=0.5B Q8_0, target=1.5B Q4_K_M) ---"
  "$BIN/llama-speculative-simple" -m "$TARGET" -md "$DRAFT" -p "$prompt" \
    --spec-type draft-simple --spec-draft-n-max 7 -n "$n" --temp 0 --seed 42 2>&1 \
    | grep -E "eval time|tokens per second|accept" || true
  echo
}

run_case "Write a short Python function that checks if a number is prime, with a brief explanation." 200
run_case "What is the capital of France? Answer in one sentence." 100
run_case "List three benefits of regular exercise." 150
