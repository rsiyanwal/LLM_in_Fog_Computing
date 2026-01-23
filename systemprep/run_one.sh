#!/usr/bin/env bash
# Usage: ./run_one.sh <blas|pure> <model_path> <prompt_file> <threads> <ctx> <batch> <decoding_tag>
# decoding_tag: greedy OR topk40_p095_t07
set -euo pipefail
BUILD="$1"; MODEL="$2"; PROMPT="$3"; THR="$4"; CTX="$5"; BATCH="$6"; DEC="$7"
BIN="$HOME/edge-llm-bench/bin/main_${BUILD}"

TOPK=0; TOPP=1.0; TEMP=0.0
if [[ "$DEC" != "greedy" ]]; then
  TOPK=$(echo "$DEC" | sed -n 's/.*topk\([0-9]\+\).*/\1/p')
  TOPP=$(echo "$DEC" | sed -n 's/.*_p\([0-9]\+\).*/0.\1/p')
  TEMP=$(echo "$DEC" | sed -n 's/.*_t\([0-9]\+\).*/0.\1/p')
fi

# Keep to well-documented flags (ctx, threads, batch, sampling). :contentReference[oaicite:6]{index=6}
"$BIN" -m "$MODEL" -f "$PROMPT" -n 128 --ctx-size "$CTX" -t "$THR" -b "$BATCH" \
       --top-k "$TOPK" --top-p "$TOPP" --temp "$TEMP"
