#!/usr/bin/env bash
# quantize the f16 gguf into every type the report compares
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

LOG="$LOGS_DIR/task1/02_quantize.log"
TYPES="${TYPES:-Q4_0 Q4_1 Q4_HQQ}"

for type in $TYPES; do
    lower="$(echo "$type" | tr 'A-Z' 'a-z')"
    out="$GGUF_DIR/$MODEL_NAME-$lower.gguf"

    if [ -f "$out" ]; then
        echo "$out already exists, skipping"
        continue
    fi

    run_logged "$LOG" "$BIN_DIR/llama-quantize" "$F16_GGUF" "$out" "$type"
done

echo "### resulting files" | tee -a "$LOG"
ls -l "$GGUF_DIR" | tee -a "$LOG"
