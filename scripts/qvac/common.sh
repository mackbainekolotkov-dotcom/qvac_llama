#!/usr/bin/env bash
# shared settings for the qvac scripts, source this from every step
set -euo pipefail

QVAC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# gated on huggingface, 00_fetch_model.sh falls back to MODEL_REPO_FALLBACK
MODEL_REPO="${MODEL_REPO:-meta-llama/Llama-3.2-3B-Instruct}"
MODEL_REPO_FALLBACK="${MODEL_REPO_FALLBACK:-unsloth/Llama-3.2-3B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-main}"

MODELS_DIR="${MODELS_DIR:-$QVAC_ROOT/models/qvac}"
HF_DIR="${HF_DIR:-$MODELS_DIR/hf}"
GGUF_DIR="${GGUF_DIR:-$MODELS_DIR/gguf}"
LOGS_DIR="${LOGS_DIR:-$QVAC_ROOT/logs}"
BUILD_DIR="${BUILD_DIR:-$QVAC_ROOT/build}"
BIN_DIR="${BIN_DIR:-$BUILD_DIR/bin}"

PYTHON_BIN="${PYTHON_BIN:-$QVAC_ROOT/../.venv-convert/bin/python}"

MODEL_NAME="${MODEL_NAME:-llama-3.2-3b}"
F16_GGUF="${F16_GGUF:-$GGUF_DIR/$MODEL_NAME-f16.gguf}"

# benchmark settings, keep these identical across every compared type
BENCH_THREADS="${BENCH_THREADS:-8}"
BENCH_SEED="${BENCH_SEED:-42}"
BENCH_NGL="${BENCH_NGL:-0}"
BENCH_NPREDICT="${BENCH_NPREDICT:-100}"
BENCH_REPS="${BENCH_REPS:-3}"

mkdir -p "$MODELS_DIR" "$HF_DIR" "$GGUF_DIR" "$LOGS_DIR"

# run a command, echo it into the log first so every number is traceable
run_logged() {
    local log="$1"; shift
    mkdir -p "$(dirname "$log")"
    {
        echo "### $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "### cwd: $(pwd)"
        echo "### cmd: $*"
        echo
    } >> "$log"
    "$@" 2>&1 | tee -a "$log"
    return "${PIPESTATUS[0]}"
}
