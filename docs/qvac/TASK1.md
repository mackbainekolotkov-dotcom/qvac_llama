# Task 1 - convert, quantize, benchmark

Self-contained checklist for the first assignment task. Every row names the artifact that proves it
and the command that reproduces it. Nothing here is asserted without a committed log behind it.

Single submitted tree, derived from upstream commit `0b5be7e4a25862bc2777d0c47eae18788a8c963a`.
The scripts and this documentation are the last four commits of the series.

---

## 1. Status at a glance

| # | Brief requirement | Status | Proof |
|---|---|---|---|
| 1 | Download a small HuggingFace model, LLaMA 3.2 3B | done, with one deviation | [`logs/task1/00_fetch_model.log`](../../logs/task1/00_fetch_model.log), [`model_provenance.txt`](../../logs/task1/model_provenance.txt) |
| 2 | Convert it to GGUF using our own scripts | done | [`logs/task1/01_convert_gguf.log`](../../logs/task1/01_convert_gguf.log) |
| 3 | Quantize the model to Q4_0 using llama.cpp tools | done | [`logs/task1/02_quantize.log`](../../logs/task1/02_quantize.log) |
| 4 | Run inference on the two example prompts | done | [`logs/task1/gen_*_prompt*.log`](../../logs/task1/), [`brief_*.log`](../../logs/task1/) |
| 5 | First token latency | done | section 4 below |
| 6 | Average token generation speed | done | section 4 below |
| 7 | RAM usage during inference | done | section 4 below |
| 8 | Scripts for conversion and quantization | done | [`scripts/qvac/`](../../scripts/qvac/) |
| 9 | Output logs | done | [`logs/task1/`](../../logs/task1/), 30 files |

**The one deviation:** `meta-llama/Llama-3.2-3B-Instruct` is a gated repository and no access token
was available. The scripts fall back to an ungated mirror of the same weights and record which repo
they actually used. The failure is in the log in the library's own words, not paraphrased:

```
WARNING: meta-llama/Llama-3.2-3B-Instruct is gated or unavailable (GatedRepoError),
         falling back to unsloth/Llama-3.2-3B-Instruct
source repo:     unsloth/Llama-3.2-3B-Instruct
source revision: 006f5dcd1393c3add266de40994ba96225e9689d
```

Set `HF_TOKEN` and the same script uses the original repository. Nothing else changes.

---

## 2. Reproduce from scratch

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build -j

bash scripts/qvac/00_fetch_model.sh    # HF_TOKEN optional
bash scripts/qvac/01_convert_gguf.sh
bash scripts/qvac/02_quantize.sh
bash scripts/qvac/03_bench.sh
bash scripts/qvac/04_perplexity.sh     # optional, wikitext-2
```

Each script uses `set -euo pipefail`, takes its settings from environment variables with defaults in
`common.sh`, skips work whose output already exists, and writes the exact command line it ran into
its log before running it. Threads, seed and `-ngl` are fixed in `common.sh` so every configuration
is measured the same way.

| Script | What it does |
|---|---|
| `00_fetch_model.sh` | `huggingface_hub.snapshot_download` at a pinned revision, falls back to the mirror, records repo id and commit sha |
| `01_convert_gguf.sh` | the repository's own `convert_hf_to_gguf.py --outtype f16` |
| `02_quantize.sh` | `llama-quantize` into Q4_0, Q4_1 and Q4_HQQ |
| `03_bench.sh` | `llama-bench`, the example prompts, peak RSS, and the KV cache matrix |
| `04_perplexity.sh` | `llama-perplexity` on `wiki.test.raw`, `-c 512` |

---

## 3. Artifacts

| File | Size | Origin |
|---|---|---|
| `llama-3.2-3b-f16.gguf` | 5.99 GiB | `convert_hf_to_gguf.py --outtype f16` |
| `llama-3.2-3b-q4_0.gguf` | 1.79 GiB | `llama-quantize <f16> <out> Q4_0` |
| `llama-3.2-3b-q4_1.gguf` | 1.95 GiB | reference for the Q4_HQQ comparison |
| `llama-3.2-3b-q4_hqq.gguf` | 1.95 GiB | Task 2, listed here for context |

The model files are not shipped with this tree, they are too large. They are reproducible:
re-running `01_convert_gguf.sh` rebuilds the F16 **byte for byte**, and re-running `02_quantize.sh`
rebuilds all three quantized files **byte for byte** (`cmp` clean on each).

---

## 4. Measurements

### 4.1 The brief's two prompts

One process per row, `/usr/bin/time -l`, `-t 8 -ngl 0 --seed 42 --no-mmap -n 100`, so latency, speed
and memory all come from the same run. Logs: `logs/task1/brief_<type>_prompt<n>.log`.

| Model | Prompt | First token | Gen (t/s) | Peak RSS |
|---|---|---|---|---|
| Q4_0 | "What is bitcoin?" (39 tok) | 435.6 ms (11.17 ms/tok) | 56.17 | 5590 MiB |
| Q4_0 | "Write a Python function to reverse a list." (44 tok) | 438.1 ms (9.96 ms/tok) | 56.86 | 5665 MiB |
| Q4_HQQ | "What is bitcoin?" (39 tok) | 1002.9 ms (25.71 ms/tok) | 40.03 | 5562 MiB |
| Q4_HQQ | "Write a Python function to reverse a list." (44 tok) | 1030.6 ms (23.42 ms/tok) | 41.47 | 5631 MiB |

Read the first-token column with the token count next to it: prompt length dominates that number, so
435 ms over 39 tokens and 438 ms over 44 tokens are the same speed, not a difference.

### 4.2 Comparable across types

Prompt-length effects make the per-prompt numbers useless for comparing types, so the comparison uses
a fixed 512-token prompt for every row (`llama-bench -p 512 -n 128 -r 3`, log
`logs/task1/bench_llama_bench.log`). Peak RSS is a separate `--no-mmap` pass over the full sequence,
which is why it is much higher than in the table above.

| Type | File size | bpw | First token | Prompt eval (t/s) | Gen (t/s) | Peak RSS | PPL (c=512) |
|---|---|---|---|---|---|---|---|
| F16 | 5.99 GiB | 16.00 | 2494 ms (4.87/tok) | 205.29 +/- 0.48 | 19.11 +/- 0.66 | 10937 MiB | 10.5709 +/- 0.284 |
| Q4_0 | 1.79 GiB | 4.50 | 2296 ms (4.48/tok) | 223.04 +/- 0.82 | 55.97 +/- 2.16 | 9036 MiB | 11.2183 +/- 0.303 |

Q4_0 against F16: 3.3x smaller, 2.9x faster generation, 1.9 GiB less resident memory, +0.647
perplexity. The full five-type table, including Q4_1 and both Q4_HQQ variants, is in
[`REPORT.md`](REPORT.md) section 3.1.

`bpw` is the block format rate. `llama-quantize` prints a higher whole-file average (4.75 for Q4_0)
because `token_embd` stays at `q6_K`. Both are correct and measure different things.

---

## 5. Generated output

Readable generations are in `logs/task1/gen_<type>_prompt<n>.log`, five types x two prompts. The
`brief_*` logs use `-v` for the millisecond timings, so their text is interleaved with scheduler
debug output; use the `gen_*` set to read the answers.

Q4_0, "What is bitcoin?":

```
Bitcoin is a digital or virtual currency that uses cryptography for security and is decentralized,
meaning it is not controlled by any government or financial institution.
```

Q4_HQQ, "Write a Python function that reverses a list." - the generated function was executed and
returns `[3, 2, 1]` for `[1, 2, 3]`:

```python
def reverse_list(input_list):
    return input_list[::-1]
```

Three more prompts across five types are in [`APPENDIX.md`](APPENDIX.md), with the reasoning for why
no subjective 1-5 score is reported: 25 side-by-side generations did not separate the types, and
perplexity does.

---

## 6. Verify without re-running anything

```bash
# which model, which revision
cat logs/task1/model_provenance.txt

# the gated fallback, in huggingface_hub's own words
grep -E "WARNING|source repo|source revision" logs/task1/00_fetch_model.log

# the conversion and the quantization command lines actually executed
grep "^### cmd:" logs/task1/01_convert_gguf.log logs/task1/02_quantize.log

# first token, generation speed and peak RSS for the brief's prompts
grep -E "prompt eval time|      eval time|maximum resident" logs/task1/brief_q4_0_prompt1.log

# the comparable per-type numbers
grep -E "^\| llama" logs/task1/bench_llama_bench.log

# perplexity
grep "Final estimate" logs/task1/ppl_f16.log logs/task1/ppl_q4_0.log
```

---

## 7. Environment

Recorded verbatim in [`logs/env.txt`](../../logs/env.txt) and `logs/pip-freeze-convert.txt`.
Summary: macOS 26.6.2, Apple M5 (4P + 6E), 16 GiB unified memory, Apple clang 21.0.0, CMake 4.4.2,
Metal backend, Python 3.14.6.

All CPU numbers use `-ngl 0` and `-t 8`. All runs use `--seed 42`. `llama-cli` at this commit has no
`-no-cnv` flag; `-st` is the single-turn equivalent and is what the scripts pass.
