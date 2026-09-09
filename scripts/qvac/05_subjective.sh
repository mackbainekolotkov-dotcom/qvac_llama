#!/usr/bin/env bash
# side by side generations for the subjective quality appendix
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="$LOGS_DIR/task1/subjective"
mkdir -p "$OUT"

TYPES="${TYPES:-f16 q4_0 q4_1 q4_hqq q4_hqq_opt}"

# the two prompts from the brief plus three fixed extras
PROMPTS=(
    "What is bitcoin?"
    "Write a Python function that reverses a list."
    "Explain the difference between TCP and UDP in three sentences."
    "A farmer has 17 sheep. All but 9 run away. How many are left? Think step by step."
    "Summarize the plot of Romeo and Juliet in exactly two sentences."
)

for t in $TYPES; do
    for i in "${!PROMPTS[@]}"; do
        n=$((i + 1))
        run_logged "$OUT/${t}_p${n}.log" "$BIN_DIR/llama-cli" \
            -m "$GGUF_DIR/$MODEL_NAME-$t.gguf" -p "${PROMPTS[$i]}" \
            -n "$BENCH_NPREDICT" -t "$BENCH_THREADS" -ngl 99 \
            --seed "$BENCH_SEED" -st > /dev/null
    done
done

echo "wrote $OUT"
