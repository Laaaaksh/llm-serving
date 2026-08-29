#!/usr/bin/env bash
# Stop the llama-server started by start_server.sh.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f server.pid ]; then
  echo "No server.pid found - is it running?" >&2
  exit 1
fi

PID=$(cat server.pid)
kill "$PID" 2>/dev/null || echo "Process $PID was not running."
rm -f server.pid
echo "Stopped."
