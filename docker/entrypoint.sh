#!/usr/bin/env bash
set -euo pipefail

# Engine defaults (leave QWEN3_MODEL unset so Settings picks per-variant default)
: "${QWEN3_VARIANT:=base}"
: "${QWEN3_DEVICE:=auto}"
: "${QWEN3_DTYPE:=bfloat16}"
: "${QWEN3_ATTN_IMPL:=flash_attention_2}"

# Service-level defaults
: "${VOICES_DIR:=/voices}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${LOG_LEVEL:=info}"
: "${CORS_ENABLED:=false}"
: "${PYTHONPATH:=/opt/api:/opt/api/engine}"

export QWEN3_VARIANT QWEN3_DEVICE QWEN3_DTYPE QWEN3_ATTN_IMPL \
       VOICES_DIR HOST PORT LOG_LEVEL CORS_ENABLED PYTHONPATH

if [ "$#" -eq 0 ]; then
  exec uvicorn app.server:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
fi
exec "$@"
