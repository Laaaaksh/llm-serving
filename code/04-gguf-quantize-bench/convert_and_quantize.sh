#!/usr/bin/env bash
# Download two real models, convert to GGUF F16, and quantize the 0.5B one
# to Q8_0/Q4_K_M/Q2_K (the 1.5B model is quantized to Q4_K_M only - it's
# used as the "target" model in code/06-speculative-decoding's demo and by
# code/07-llama-server-bench).
#
# Run build.sh first. Companion to curriculum/04-quantization.md.
set -euo pipefail
cd "$(dirname "$0")"
BIN=llama.cpp/build/bin

if [ ! -d venv ]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

mkdir -p models gguf
[ -d models/Qwen2.5-0.5B-Instruct ] || hf download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir models/Qwen2.5-0.5B-Instruct
[ -d models/Qwen2.5-1.5B-Instruct ] || hf download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir models/Qwen2.5-1.5B-Instruct

[ -f gguf/qwen2.5-0.5b-instruct-f16.gguf ] || python llama.cpp/convert_hf_to_gguf.py \
  models/Qwen2.5-0.5B-Instruct --outfile gguf/qwen2.5-0.5b-instruct-f16.gguf --outtype f16
[ -f gguf/qwen2.5-1.5b-instruct-f16.gguf ] || python llama.cpp/convert_hf_to_gguf.py \
  models/Qwen2.5-1.5B-Instruct --outfile gguf/qwen2.5-1.5b-instruct-f16.gguf --outtype f16

for q in Q8_0 Q4_K_M Q2_K; do
  [ -f "gguf/qwen2.5-0.5b-instruct-${q}.gguf" ] && continue
  "$BIN/llama-quantize" gguf/qwen2.5-0.5b-instruct-f16.gguf \
    "gguf/qwen2.5-0.5b-instruct-${q}.gguf" "$q"
done
[ -f gguf/qwen2.5-1.5b-instruct-Q4_K_M.gguf ] || "$BIN/llama-quantize" \
  gguf/qwen2.5-1.5b-instruct-f16.gguf gguf/qwen2.5-1.5b-instruct-Q4_K_M.gguf Q4_K_M

echo
echo "Done. GGUF files in gguf/:"
ls -lh gguf/*.gguf
