#!/usr/bin/env bash
# Launch a real OpenAI-compatible llama-server, in the background, serving
# Qwen2.5-1.5B-Instruct Q4_K_M (built and quantized by
# code/04-gguf-quantize-bench). Stop it with stop_server.sh.
#
# Companion to curriculum/06-choosing-an-engine.md.
set -euo pipefail
cd "$(dirname "$0")"

LAB=../04-gguf-quantize-bench
BIN="$LAB/llama.cpp/build/bin/llama-server"
MODEL="$LAB/gguf/qwen2.5-1.5b-instruct-Q4_K_M.gguf"
PORT=8899

if [ ! -f "$MODEL" ]; then
  echo "Missing $MODEL - run code/04-gguf-quantize-bench's build.sh and" >&2
  echo "convert_and_quantize.sh first." >&2
  exit 1
fi

"$BIN" -m "$MODEL" --port "$PORT" -c 4096 > server.log 2>&1 &
echo $! > server.pid

echo "llama-server starting on http://127.0.0.1:$PORT (PID $(cat server.pid))."
echo "Logs: server.log. Stop with ./stop_server.sh."
echo "Waiting for it to become ready..."

for _ in $(seq 1 30); do
  if curl -s -o /dev/null "http://127.0.0.1:$PORT/health"; then
    echo "Ready."
    exit 0
  fi
  sleep 1
done
echo "Still not responding after 30s - check server.log." >&2
exit 1
