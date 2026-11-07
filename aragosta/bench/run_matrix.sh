#!/usr/bin/env bash
# Reproducible server-level matrix: start llama-server with a profile, run the
# agentic load at several concurrencies, stop, write JSON to results/.
#
#   run_matrix.sh <label> <llama-server-binary> <model.gguf> [-- extra server args]
#   SESSIONS="1 2 4 8" PREFIX=14000 TURNS=3 PORT=8081 run_matrix.sh base ./build/bin/llama-server m.gguf -- -np 4
set -euo pipefail
LABEL=$1; BIN=$2; MODEL=$3; shift 3
[ "${1:-}" = "--" ] && shift
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HERE/../results}; mkdir -p "$OUT"
PORT=${PORT:-8081}; SESSIONS=${SESSIONS:-"1 2 4"}; PREFIX=${PREFIX:-14000}; TURNS=${TURNS:-3}

"$BIN" -m "$MODEL" --host 127.0.0.1 --port "$PORT" --jinja -ngl 99 -fa on "$@" \
    >"$OUT/$LABEL.server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true' EXIT
for _ in $(seq 240); do
    curl -sf "localhost:$PORT/health" >/dev/null && break
    kill -0 $SRV 2>/dev/null || { echo "server died:"; tail -20 "$OUT/$LABEL.server.log"; exit 1; }
    sleep 2
done
for n in $SESSIONS; do
    python3 "$HERE/agent_load.py" --url "http://127.0.0.1:$PORT" --sessions "$n" --turns "$TURNS" \
        --prefix-tokens "$PREFIX" ${LOAD_ARGS:-} --label "$LABEL" --out "$OUT/$LABEL.c$n.json" | tee -a "$OUT/$LABEL.summary.jsonl"
    curl -s -X POST "localhost:$PORT/slots/0?action=erase" >/dev/null 2>&1 || true
done
