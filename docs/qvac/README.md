# QVAC assignment - navigation

- [`TASK1.md`](TASK1.md) - Task 1 on its own: requirement checklist, artifacts, how to verify each
- [`TASK2.md`](TASK2.md) - Task 2 on its own: the format, the kernels, the KV cache, how to verify each
- [`TASK3.md`](TASK3.md) - Task 3 on its own: how to run `--mmproj-backend`, and the proof the model is untouched
- [`REPORT.md`](REPORT.md) - environment, filled comparison tables, verification evidence
- [`SUMMARY.md`](SUMMARY.md) - one page: what changed, issues faced, suggestions
- [`APPENDIX.md`](APPENDIX.md) - 25 side by side generations, five prompts across five types

The first commit is the unmodified upstream tree at
`0b5be7e4a25862bc2777d0c47eae18788a8c963a`, so `git diff <first-commit> HEAD` is exactly the work
submitted here and nothing else. Everything after it is one commit per logical change:

```bash
git log --oneline                      # the change set, one commit per unit of work
git diff $(git rev-list --max-parents=0 HEAD) HEAD --stat
```

Task 2 spans the type definition, the NEON kernel, the quantization pipeline, the KV cache type, the
HQQ optimizer, the Metal kernels, the tests and the gguf-py side. Task 3 is two commits, the flag and
the fallback report. Task 1 is the scripts, and the documentation and logs follow.

Scripts: `scripts/qvac/`. Logs: `logs/task1`, `logs/task2`, `logs/task3`, plus `logs/env.txt`.
Run `scripts/qvac/06_test_env.sh` once before `ctest`; it fetches the two fixtures the build does
not produce, and the suite is then 60/60.

Quick check:

```bash
./build/bin/llama-quantize model-f16.gguf model-q4hqq.gguf Q4_HQQ
./build/bin/llama-cli -m model-q4hqq.gguf -p "What is bitcoin?" -n 100 -ngl 0   # cpu
./build/bin/llama-cli -m model-q4hqq.gguf -p "What is bitcoin?" -n 100 -ngl 99  # metal
./build/bin/llama-cli -m model-q4hqq.gguf -c 8192 -fa on --cache-type-k q4_hqq --cache-type-v q4_hqq -p "..."
./build/bin/test-quantize-fns && ./build/bin/test-backend-ops -o MUL_MAT
```
