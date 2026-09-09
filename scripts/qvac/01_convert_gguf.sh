#!/usr/bin/env bash
# convert the huggingface checkpoint to gguf f16
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

LOG="$LOGS_DIR/task1/01_convert_gguf.log"
SRC="$HF_DIR/$MODEL_NAME"

if [ -f "$F16_GGUF" ]; then
    echo "$F16_GGUF already exists, nothing to do"
    exit 0
fi

run_logged "$LOG" "$PYTHON_BIN" "$QVAC_ROOT/convert_hf_to_gguf.py" "$SRC" \
    --outtype f16 --outfile "$F16_GGUF"

ls -l "$F16_GGUF" | tee -a "$LOG"
