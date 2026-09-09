#!/usr/bin/env bash
# throughput, latency, memory and sample generations for every compared type
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

OUT="$LOGS_DIR/task1"
mkdir -p "$OUT" "$LOGS_DIR/task2"

TYPES="${TYPES:-f16 q4_0 q4_1 q4_hqq}"
PROMPT_1="${PROMPT_1:-What is bitcoin?}"
PROMPT_2="${PROMPT_2:-Write a Python function that reverses a list.}"

model_path() { echo "$GGUF_DIR/$MODEL_NAME-$1.gguf"; }

# 1. throughput, identical settings for every type
BENCH_MODELS=()
for t in $TYPES; do BENCH_MODELS+=(-m "$(model_path "$t")"); done
run_logged "$OUT/bench_llama_bench.log" "$BIN_DIR/llama-bench" "${BENCH_MODELS[@]}" \
    -p 512 -n 128 -r "$BENCH_REPS" -t "$BENCH_THREADS" -ngl "$BENCH_NGL"

# 2. sample generations, fixed seed and fixed token budget
for t in $TYPES; do
    for i in 1 2; do
        eval "prompt=\$PROMPT_$i"
        run_logged "$OUT/gen_${t}_prompt${i}.log" "$BIN_DIR/llama-cli" \
            -m "$(model_path "$t")" -p "$prompt" -n "$BENCH_NPREDICT" \
            -t "$BENCH_THREADS" -ngl "$BENCH_NGL" --seed "$BENCH_SEED" -st
    done
done

# 3. peak resident set size, --no-mmap so the weights are counted in rss
for t in $TYPES; do
    log="$OUT/rss_${t}.log"
    {
        echo "### cmd: /usr/bin/time -l llama-cli -m $(model_path "$t") --no-mmap"
    } >> "$log"
    /usr/bin/time -l "$BIN_DIR/llama-cli" -m "$(model_path "$t")" \
        -p "$PROMPT_1" -n "$BENCH_NPREDICT" -t "$BENCH_THREADS" -ngl "$BENCH_NGL" \
        --seed "$BENCH_SEED" --no-mmap -st >> "$log" 2>&1
    grep -E 'maximum resident set size' "$log" | tail -1
done

# 4. kv cache matrix on the q4_hqq weights, flash attention on
KV_MODEL="$(model_path q4_hqq)"
for kv in f16 q8_0 q4_0 q4_hqq; do
    run_logged "$LOGS_DIR/task2/kv_${kv}.log" "$BIN_DIR/llama-cli" \
        -m "$KV_MODEL" -c 8192 -fa on -v \
        --cache-type-k "$kv" --cache-type-v "$kv" \
        -p "$PROMPT_1" -n "$BENCH_NPREDICT" -t "$BENCH_THREADS" -ngl "$BENCH_NGL" \
        --seed "$BENCH_SEED" -st
done

# 5. peak resident set size for the same kv cache matrix, --no-mmap so the
#    weights are counted the same way as in step 3 and the delta is the cache
for kv in f16 q8_0 q4_0 q4_hqq; do
    log="$LOGS_DIR/task2/rss_kv_${kv}.log"
    {
        echo "### cmd: /usr/bin/time -l llama-cli -m $KV_MODEL -c 8192 -fa on --cache-type-k $kv --cache-type-v $kv --no-mmap"
    } >> "$log"
    /usr/bin/time -l "$BIN_DIR/llama-cli" \
        -m "$KV_MODEL" -c 8192 -fa on \
        --cache-type-k "$kv" --cache-type-v "$kv" \
        -p "$PROMPT_1" -n "$BENCH_NPREDICT" -t "$BENCH_THREADS" -ngl "$BENCH_NGL" \
        --seed "$BENCH_SEED" --no-mmap -st >> "$log" 2>&1
    grep -E 'maximum resident set size' "$log" | tail -1
done

# 6. unified kv cache matrix for the v2 report table: one process per configuration under
#    /usr/bin/time -l, so kv self size, first token latency, generation speed and peak memory
#    all come from the same run. the q4_hqq/f16 row covers the k-only case without flash attention
for cfg in "f16 f16 on" "q8_0 q8_0 on" "q4_0 q4_0 on" "q4_hqq f16 off" "q4_hqq q4_hqq on"; do
    set -- $cfg
    k="$1"; v="$2"; fa="$3"
    log="$LOGS_DIR/task2/kvfull_${k}_${v}_fa${fa}.log"
    {
        echo "### cmd: /usr/bin/time -l llama-cli -m $KV_MODEL -c 8192 -fa $fa --cache-type-k $k --cache-type-v $v --no-mmap -v"
    } >> "$log"
    /usr/bin/time -l "$BIN_DIR/llama-cli" \
        -m "$KV_MODEL" -c 8192 -fa "$fa" -v \
        --cache-type-k "$k" --cache-type-v "$v" \
        -p "$PROMPT_1" -n "$BENCH_NPREDICT" -t "$BENCH_THREADS" -ngl "$BENCH_NGL" \
        --seed "$BENCH_SEED" --no-mmap -st >> "$log" 2>&1
    echo "== $k/$v fa=$fa"
    grep -E 'llama_kv_cache: size|prompt eval time|      eval time|maximum resident|peak memory' "$log" | tail -5
done
