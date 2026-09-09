#!/usr/bin/env bash
# perplexity on wikitext-2, same chunk count for every type
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="$LOGS_DIR/task1"
DATA_DIR="${DATA_DIR:-$MODELS_DIR/data}"
WIKI="$DATA_DIR/wikitext-2-raw/wiki.test.raw"

TYPES="${TYPES:-f16 q4_0 q4_1 q4_hqq}"
PPL_CTX="${PPL_CTX:-512}"
PPL_CHUNKS="${PPL_CHUNKS:-40}"

mkdir -p "$DATA_DIR"
if [ ! -f "$WIKI" ]; then
    (cd "$DATA_DIR" && sh "$QVAC_ROOT/scripts/get-wikitext-2.sh")
fi

for t in $TYPES; do
    run_logged "$OUT/ppl_${t}.log" "$BIN_DIR/llama-perplexity" \
        -m "$GGUF_DIR/$MODEL_NAME-$t.gguf" -f "$WIKI" \
        -c "$PPL_CTX" --chunks "$PPL_CHUNKS" -t "$BENCH_THREADS" -ngl "$BENCH_NGL"
done

echo "### summary"
grep -H 'Final estimate' "$OUT"/ppl_*.log || true
