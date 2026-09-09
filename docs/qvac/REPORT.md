# QVAC assignment report - Q4_HQQ, KV cache and `--mmproj-backend`

All numbers below are measured on this machine, from the logs under `logs/`. No placeholders.

## 1. Environment

| Item | Value |
|---|---|
| OS | macOS 26.6.2 (25G83), Darwin kernel 25.6.0 |
| CPU | Apple M5 (Mac17,3), 10 cores: 4 performance + 6 efficiency |
| RAM | 16 GiB unified |
| Compiler | Apple clang 21.0.0 (clang-2100.1.1.101) |
| CMake | 4.4.2 |
| GPU backend | Metal (`-DGGML_METAL=ON`), device `MTL0` = Apple M5, 12124 MiB |
| Python | 3.14.6, see `logs/pip-freeze-convert.txt` |
| Upstream commit (pinned) | `0b5be7e4a25862bc2777d0c47eae18788a8c963a` |
| Submitted as | one commit per logical change on top of the pinned upstream commit |

Build:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build -j
```

## 2. Model

`meta-llama/Llama-3.2-3B-Instruct` is a gated repository and no HuggingFace token was available,
so `scripts/qvac/00_fetch_model.sh` fell back to the ungated mirror of the same weights, as the
spec allows:

| Item | Value |
|---|---|
| Repo used | `unsloth/Llama-3.2-3B-Instruct` |
| Revision | `006f5dcd1393c3add266de40994ba96225e9689d` |
| Architecture | llama, 28 layers, head_dim 128 (a multiple of 32, so every 32-block type is usable) |

Set `HF_TOKEN` and the scripts use the original `meta-llama` repo instead; nothing else changes.

Evidence, rather than prose: `logs/task1/00_fetch_model.log` holds the `GatedRepoError` and the
fallback in the library's own words, and `logs/task1/model_provenance.txt` is the tracked record of
the repo id and revision, since the model tree itself is not shipped.
`logs/task1/01_convert_gguf.log` is the conversion run. Re-running `01_convert_gguf.sh` against the
same checkpoint reproduces `llama-3.2-3b-f16.gguf` **byte for byte** (6433688192 bytes, `cmp`
clean), so the F16 every other row is derived from is reproducible, not a one-off artifact.

## 3. Weights comparison

Perplexity: `wiki.test.raw`, `-c 512`, 40 chunks, identical for every row.
Speed: `llama-bench -p 512 -n 128 -r 3`. Peak RSS: `/usr/bin/time -l` with `--no-mmap`.
There is no subjective 1-5 column: 25 side by side generations did not separate the types.
See [`APPENDIX.md`](APPENDIX.md) for the outputs and the reasoning behind dropping it.
All five perplexity numbers were re-measured against the final tree, after the NEON, Metal and
flash attention kernels landed, and reproduced to the last digit. Logs: `logs/task1/ppl_*.log`.
Four of those logs hold two complete back to back runs (a first backgrounded run was thought to have
been killed and was restarted, but it had survived and finished). The duplication is left in place
on purpose: the two independent runs agree exactly, which is stronger evidence than one run.

### 3.1 CPU (`-ngl 0`, `-t 8`)

| Type | File size | bpw | First token (ms) | Prompt eval (t/s) | Gen (t/s, median) | Gen spread | Peak RSS | PPL (c=512) | dPPL vs F16 | Quality 1-5 |
|---|---|---|---|---|---|---|---|---|---|---|
| F16 | 5.99 GiB | 16.00 | 2494 (4.87/tok) | 205.29 | 19.11 | +/- 0.66 | 10937 MiB | 10.5709 +/- 0.284 | - | n/a |
| Q4_0 | 1.79 GiB | 4.50 | 2296 (4.48/tok) | 223.04 | 55.97 | +/- 2.16 | 9036 MiB | 11.2183 +/- 0.303 | +0.647 | n/a |
| Q4_1 | 1.95 GiB | 5.00 | 2515 (4.91/tok) | 203.54 | 45.64 | +/- 1.34 | 9019 MiB | 11.4748 +/- 0.312 | +0.904 | n/a |
| Q4_HQQ | 1.95 GiB | 5.00 | 2557 (4.99/tok) | 200.26 | 36.92 | +/- 0.85 | 9066 MiB | 11.4726 +/- 0.312 | +0.902 | n/a |
| Q4_HQQ + optimizer | 1.95 GiB | 5.00 | 2575 (5.03/tok) | 198.83 | 36.50 | +/- 0.80 | 8931 MiB | 11.3808 +/- 0.309 | +0.810 | n/a |

The `bpw` column is the block format's own rate (20 B per 32 weights = 5.00 for Q4_1 and Q4_HQQ,
18 B = 4.50 for Q4_0). `llama-quantize` prints a higher whole-file average - 4.75 and 5.19 - because
`token_embd` is kept at `q6_K` under the `Q4_1` policy. Both numbers are correct and measure
different things; the format rate is the one that makes the types comparable.

Quantization is reproducible: re-running `02_quantize.sh` against the same F16 rebuilds the Q4_0,
Q4_1 and Q4_HQQ files **byte for byte** (`cmp` clean on all three). The paths inside
`logs/task1/02_quantize.log` point at a scratch directory because the script skips outputs that
already exist, so it was pointed elsewhere purely to capture the log.

First token is the 512-token prompt-eval wall time from the same `llama-bench` runs, so the token
count is identical in every row. "Gen spread" is `llama-bench`'s own standard deviation over the 3
repetitions, not a min-max range. The quality column is `n/a` rather than a number on purpose: see
section 3.4 and `APPENDIX.md`.

### 3.1.1 The brief's two prompts, measured end to end

The table above derives first-token latency from the 512-token `llama-bench` prompt, which is what
makes the rows comparable. The brief asks for the numbers on its own two prompts, so those are
measured separately, one process per row under `/usr/bin/time -l`, so latency, speed and memory all
come from the same run (`logs/task1/brief_*.log`):

| Model | Prompt | First token | Gen (t/s) | Peak RSS |
|---|---|---|---|---|
| Q4_0 | "What is bitcoin?" (39 tok) | 435.6 ms (11.17 ms/tok) | 56.17 | 5590 MiB |
| Q4_0 | "Write a Python function to reverse a list." (44 tok) | 438.1 ms (9.96 ms/tok) | 56.86 | 5665 MiB |
| Q4_HQQ | "What is bitcoin?" (39 tok) | 1002.9 ms (25.71 ms/tok) | 40.03 | 5562 MiB |
| Q4_HQQ | "Write a Python function to reverse a list." (44 tok) | 1030.6 ms (23.42 ms/tok) | 41.47 | 5631 MiB |

These runs use `-v`, so the generated text in them is interleaved with scheduler debug output. The
readable side-by-side generations are the separate `logs/task1/gen_*_prompt*.log` set, and five more
prompts across five types are in [`APPENDIX.md`](APPENDIX.md).

Peak RSS here is far below the 9 GiB in the table above because that column comes from a `--no-mmap`
run of the *full* benchmark sequence; these are single short generations.

### 3.2 Metal (`-ngl 99`)

| Type | pp512 (t/s) | tg128 (t/s) |
|---|---|---|
| F16 | 1365.01 +/- 0.73 | 21.47 +/- 0.07 |
| Q4_0 | 1353.94 +/- 4.06 | 64.17 +/- 0.20 |
| Q4_1 | 1372.05 +/- 6.02 | 57.67 +/- 0.02 |
| Q4_HQQ | 1367.37 +/- 2.28 | 57.55 +/- 0.16 |

On Metal `Q4_HQQ` is within noise of `Q4_1`, which is the honest comparison: the two formats have
the same block layout and the same bpw. The gap that remains on CPU is the `Q8_0` vs `Q8_1`
trade-off explained in the summary.

The weights really are on the device, from `logs/task2/metal_offload_q4_hqq.log`:

```
load_tensors:  MTL0_Mapped model buffer size =  1988.90 MiB
load_tensors:   CPU_Mapped model buffer size =   308.23 MiB   (token_embd, kept on host)
```

`test-backend-ops -o MUL_MAT` reports `q4_hqq` as `OK` on `MTL0` against the CPU reference, not
`not supported`, so the kernel is genuinely executing on the GPU.

### 3.2.1 Device attribution

Required because "runs on GPU" and "the `Q4_HQQ` matmuls run on GPU" are different claims, and a
new type without backend kernels would satisfy only the first.

| Config | Ops on device | Ops on CPU | Source |
|---|---|---|---|
| `-ngl 99`, Q4_HQQ weights | all 28 blocks: `mul_mat`, attention, norms, rope. `graph splits = 2`, so exactly one CPU segment | `token_embd` lookup only (`get_rows` over the `q6_K` embedding kept on host) | `logs/task2/ngl99_attribution.log` |

```
load_tensors:  MTL0_Mapped model buffer size =  1988.90 MiB
load_tensors:   CPU_Mapped model buffer size =   308.23 MiB
load_tensors: offloaded 29/29 layers to GPU
sched_reserve: graph splits = 2
```

Per-op confirmation from `test-backend-ops`, counting `q4_hqq` cases per backend:

| Op | MTL0 | BLAS |
|---|---|---|
| MUL_MAT | 6 OK | 12 not supported |
| MUL_MAT_ID | 1 OK | 3 not supported |
| GET_ROWS | 3 OK | 4 not supported |
| FLASH_ATTN_EXT | 300 OK | 336 not supported |
| CPY | 2 OK, 13 not supported | 17 not supported |
| SET_ROWS | 11 OK, 12 not supported | 24 not supported |

The `not supported` lines on `MTL0` for `CPY` and `SET_ROWS` are not a gap in this branch: `q4_1`
scores exactly 2 OK / 13 and 11 OK / 12 on the same op sets, so `q4_hqq` is at parity with the type
it mirrors. Every `BLAS` line is `not supported` because that backend implements none of these ops
for any quantized type.

### 3.3 First token latency

Derived from the same `pp512` runs, so the prompt token count is identical everywhere:

| Type | CPU prompt eval | Metal prompt eval |
|---|---|---|
| F16 | 2494 ms (4.87 ms/tok) | 375 ms (0.73 ms/tok) |
| Q4_0 | 2296 ms (4.48 ms/tok) | 378 ms (0.74 ms/tok) |
| Q4_1 | 2515 ms (4.91 ms/tok) | 373 ms (0.73 ms/tok) |
| Q4_HQQ | 2557 ms (4.99 ms/tok) | 374 ms (0.73 ms/tok) |

## 4. KV cache comparison

`-c 8192`, `-fa on`, `Q4_HQQ` weights, CPU. `KV self size` is the `llama_kv_cache: size` line.
Memory is a separate `/usr/bin/time -l` pass with `--no-mmap`, same as the weights table, so the
model bytes are counted identically in every row and the difference between rows is the cache
(`logs/task2/rss_kv_*.log`).

| K type | V type | `-fa` | KV self size | Peak RSS | Peak footprint | First token (ms) | Gen (t/s) | Long context probe |
|---|---|---|---|---|---|---|---|---|
| f16 | f16 | on | 896.00 MiB | 3361.5 MiB | 3744.3 MiB | 836.9 (21.46/tok) | 41.60 | pass |
| q8_0 | q8_0 | on | 476.00 MiB | 3366.0 MiB | 3323.7 MiB | 800.7 (20.53/tok) | 40.67 | pass |
| q4_0 | q4_0 | on | 252.00 MiB | 3140.5 MiB | 3099.2 MiB | 810.6 (20.78/tok) | 40.18 | pass |
| q4_hqq | f16 | **off** | 588.00 MiB | 3478.3 MiB | 3436.2 MiB | 813.9 (20.87/tok) | 38.46 | pass |
| q4_hqq | q4_hqq | on | 280.00 MiB | 3169.5 MiB | 3127.8 MiB | 799.6 (20.50/tok) | 39.33 | pass |

Every column in this table comes from a single process per row (`logs/task2/kvfull_*.log`), so the
cache size, the latency, the speed and the memory are all the same run. The `q4_hqq / f16` row is
the K-only configuration and is deliberately run with `-fa off`: it is the combination that would
still work if a quantized V cache required flash attention, and it does work here.

Both memory columns come from the same `/usr/bin/time -l` block and they disagree for `f16`, so
which one answers the question matters. `peak memory footprint` tracks the cache almost exactly:
its deltas against `f16` are -420.6, -645.1 and -616.5 MiB, against -420, -644 and -616 MiB of
declared `KV self size` - within 1.1 MiB in all three rows. `maximum resident set size` does not: it
puts `f16` 4.5 MiB *below* `q8_0`, because on this 16 GiB machine the `f16` run is the one that
pages, so its resident set is capped below its actual demand while its footprint is not. The spec
asks for max RSS and it is reported, but the footprint column is the honest number here, and the
two agree to within ~42 MiB in every row that does not page.

The `q4_hqq` row is 28.6 MiB above `q4_0` by footprint, against 28.00 MiB of declared cache
difference: the 5.00 vs 4.50 bpw gap, and nothing else, shows up in the process memory.

The probe puts a fact at position 0, about 5000 tokens of filler after it, and asks for the fact at
the end. Every cache type returned `84-QUARTZ-1791` (`logs/task2/longctx_*.log`).

Same matrix on Metal (`-ngl 99`), after adding the `q4_hqq` flash attention kernels:

| K/V type | KV self size | Gen (t/s) |
|---|---|---|
| f16 | 896.00 MiB | 40.1 |
| q8_0 | 476.00 MiB | 40.1 |
| q4_0 | 252.00 MiB | 37.8 |
| q4_hqq | 280.00 MiB | 39.7 |

Combinations actually verified on the pinned commit:

| Configuration | CPU | Metal |
|---|---|---|
| `-ctk q4_hqq`, no flash attention | works | works |
| `-ctv q4_hqq`, no flash attention | works | works |
| `-ctk q4_hqq -ctv q4_hqq -fa on` | works | works |

The often quoted "a quantized V cache needs flash attention" does not hold on this commit: both
K-only and V-only work without `-fa` on either backend. `test-backend-ops -o FLASH_ATTN_EXT` reports
300 `q4_hqq` cases OK on `MTL0`; the remaining "not supported" lines are all BLAS, which implements
no flash attention for any type.

Getting there needed 79 template instantiations in `ggml/src/ggml-metal/kernels/fa.metal` mirroring
the `q4_1` ones, plus `GGML_TYPE_Q4_HQQ` in the flash attention type allowlist in
`ggml-metal-device.m`. Before that the type was correctly reported as unsupported and fell back to
the CPU, which is the acceptable behaviour, but the GPU numbers above are the better outcome.

## 5. Task 3 - `--mmproj-backend`

Verified with `SmolVLM-256M-Instruct` (f16 model + f16 projector) on `media/matmul.png`,
`-p "Describe this image." -n 64 --seed 42`.

| Configuration | Startup log line | Median projector encode | Output |
|---|---|---|---|
| `--mmproj-backend cpu -ngl 99` | `projector device: CPU \| language model device: auto (-ngl 99)` | 799 ms | identical |
| `--mmproj-backend MTL0 -ngl 0` | `projector device: MTL0 \| language model device: auto (-ngl 0)` | 54 ms | identical |
| default (`auto`) `-ngl 99` | `projector device: auto \| language model device: auto (-ngl 99)` | 55 ms | identical |

The 14.8x encode time difference between row 1 and row 2 is the proof that the projector moved,
while the model backend selection stayed where `-ngl` put it. All three produced the same
description of the image. Logs: `logs/task3/`.

Value handling:

```
$ llama-mtmd-cli ... --mmproj-backend Cuda9
error while handling argument "--mmproj-backend": unknown device "Cuda9", valid values are: auto, cpu, MTL0, BLAS, CPU
```

`--mmproj-backend mtl0` (lower case) is accepted, so matching is case insensitive as required.

### 5.1 Partial fallback is reported, not hidden

A device can exist, initialise, and still not run the whole projector. Upstream prints only a
generic "the performance will be suboptimal" warning in that case, which reads as a tuning note
rather than as the requested placement not being honoured. When a device was named explicitly, the
outcome is now stated directly (`-v`, `logs/task3/guard_*.log`):

| Configuration | Line emitted | Graph splits |
|---|---|---|
| `--mmproj-backend MTL0` | `warmup: projector runs entirely on the requested device MTL0 (396 ops)` | 1 |
| `--mmproj-backend BLAS` | `warmup: projector falls back to the CPU for 238 of 420 ops, the requested device BLAS does not support them` | 172 |
| `--mmproj-backend cpu` | nothing new | - |
| default (`auto`) | nothing new | - |

`BLAS` is the useful adversarial case here: it is an accelerator device, so it is accepted as a
target, but it implements a small fraction of the projector graph. The run still completes with exit
code 0, which is the required behaviour - an explicit report, not a crash and not silence. `auto`
and `cpu` stay on the upstream code path and emit nothing new, so default behaviour is unchanged.
`graph splits = 1` on the `MTL0` row is the independent confirmation that the whole graph is on the
device.

Fixing this exposed a second problem in the same place. In the flash attention auto path the second
`reserve_compute_meta()` result was discarded, so upstream's existing op list - and the new line
built on it - described the flash attention graph that had just been *rejected* rather than the one
that runs. Keeping the result changes the BLAS report from `226 of 396` to `238 of 420`. The old
numbers were not wrong about a graph, they were right about the wrong graph.

## 6. Correctness

### 6.1 Full test suite

`ctest` over the whole repository suite, not just the quantization tests:

| Suite | Result |
|---|---|
| `ctest --test-dir build`, Q4_HQQ tree | **60 / 60 passed**, 182.1 s (`logs/task2/ctest_full.log`) |
| `ctest --test-dir build`, mmproj tree | 60 / 60 passed |

Two of these need fixtures that are not produced by the build, and both fail on a clean machine
regardless of branch: `test-jinja-py` renders its expected output with the reference `jinja2`
module, and `test-tokenizers-ggml-vocabs` needs the vocab GGUFs, which are git-lfs objects that a
plain clone leaves as 132-byte pointers (the test then reports `invalid magic characters: 'vers'`).
`scripts/qvac/06_test_env.sh` installs the first and fetches the second straight from the
HuggingFace resolve endpoint, so `git-lfs` is not required. Run it once before `ctest`.

### 6.2 Round trip and kernels

| Check | Result |
|---|---|
| `test-quantize-fns` | pass, 0 tests failed |
| `q4_hqq` round trip RMSE | 0.001144, identical to `q4_1` on the same data (budget is q4_1 + 10%) |
| `q4_hqq` vec_dot error vs f64 reference | 0.007968, `q4_1` scores 0.007931 |
| with the HQQ optimizer | RMSE 0.001137, vec_dot error 0.006484, both better |
| `test-backend-ops -o MUL_MAT` | pass, `q4_hqq` OK on Metal against the CPU reference |
| `test-backend-ops -o CPY / GET_ROWS / SET_ROWS / MUL_MAT_ID` | pass |
| `test-backend-ops -o FLASH_ATTN_EXT` | pass, 300 `q4_hqq` cases OK on Metal, the rest are BLAS which has no FA at all |

### 6.3 Numerical edge cases

`test_q4_hqq_edge_cases` in `tests/test-quantize-fns.cpp` runs the block shapes the spec calls out
and asserts four things: the round trip stays finite (non-negotiable), the error stays bounded by
the input range, `q4_hqq` does not lose to `q4_1` by more than 10 percent, and the narrow-band case
stays inside an absolute bound derived below.

| Case | q4_hqq RMSE | q4_1 RMSE | Note |
|---|---|---|---|
| uniform noise | 0.000784255 | 0.000784046 | parity |
| gaussian | 0.00174958 | 0.00175001 | parity |
| all zero | 0 | 0 | exact |
| constant non-zero | 2.5894e-06 | 2.5894e-06 | parity, the residual is f16 rounding of the zero point |
| single outlier | 0.000523103 | 0.000505524 | within the 10 percent budget |
| extreme dynamic range | 21.1895 | **nan** | q4_hqq stays finite where q4_1 does not |
| denormals | 0 | 0 | parity |
| narrow band far from zero | 0.00135001 | 0 | asserted on absolute error, see below |
| huge magnitude (scale underflow) | 2.56e+11 | **nan** | outside the format's range; finiteness is all that is promised |

Three of these are worth stating plainly rather than burying:

- **q4_hqq is more robust than q4_1 on extreme blocks.** With values spanning 1e-6 to 1e6 in one
  block, `q4_1` overflows its f16 `d` to inf and dequantizes to inf. `q4_hqq` does not, because the
  clamping bounds both parameters. The test asserts finiteness for `q4_hqq` and skips the parity
  check when `q4_1` is itself not finite.
- **q4_hqq is worse than q4_1 on a narrow band far from zero, and no encoder can fix it.**
  `q4_1` stores `(max-min)/15`, a small number that f16 holds comfortably. `q4_hqq` stores the
  reciprocal `15/(max-min)`, which saturates. The parity check is replaced by an absolute bound of
  `|min|*2^-10`, because the reconstruction `(q - zero)/scale` carries an offset of `|zero|/scale
  ~ |min|`, and f16 precision on that offset puts a floor of about `|min|*2^-11` on the error no
  matter how narrow the band is. Measured 0.00135 against a bound of 0.977.
- **A block wider than f16 can express is unrepresentable, and only finiteness is promised.** A
  range of 3e12 needs `scale = 5e-12`; the smallest f16 subnormal is 6e-8, so the scale rounds to
  zero. `dequantize_row_q4_hqq` guards this with `is = s != 0 ? 1/s : 0`, so the output stays
  finite. Clamping the scale up in the encoder instead would be a regression - see section 6.8.

Running the same test under UBSan surfaced a latent defect in **upstream** `quantize_row_q4_1_ref`
(`ggml/src/ggml-quants.c:178`): on the denormal block, `1.0f/d` overflows to inf, `(x - min)*inf`
produces NaN, and the cast `(int8_t)(NaN)` is undefined behaviour. Confirmed as pre-existing and not
caused by this branch: rebuilding the same ASan/UBSan binary against the upstream version of
`test-quantize-fns.cpp` reports 0 runtime errors, because upstream's own test data never reaches
that path. `q4_hqq` has no equivalent, it multiplies by a clamped scale instead of dividing.

### 6.4 Independent reimplementation, checked against the C encoder

`gguf-py` carries a NumPy (de)quantizer per type and a ctypes harness,
`gguf-py/tests/test_quants.py`, that compares it against the C library bit for bit. `Q4_HQQ` was
registered in `constants.py` but not in `quants.py`, so Python could read the metadata of a Q4_HQQ
file and then raise `NotImplementedError` on its tensors. Filling that in gives a second
implementation written from the format description rather than from the C code, which is a real
check on the C one:

```
Quantization to Q4_HQQ matches exactly
Dequantization from Q4_HQQ matches exactly
Dequantization from random f16 data as Q4_HQQ matches exactly
```

over 8 x 1024 x 1024 values (`logs/task2/gguf_py_vs_c_q4_hqq.log`), plus 39.3 MB of real model
weights re-quantized in NumPy and compared against what `llama-quantize` wrote: zero differing
bytes.

Getting there took three corrections, all in the Python side, and each one pins down a detail of the
C encoder that a reader would otherwise have to take on trust:

1. `zero` is computed from the **unrounded** scale and only then stored as f16. Deriving it from the
   already-rounded scale changed 1.6 % of bytes.
2. `roundf` sends halves away from zero; `np.round` sends them to even.
3. The C compiler contracts `x*scale + zero` into an **fma**, so it rounds once. Doing the multiply
   and the add separately moved one value in 262144 blocks that sat exactly on 11.5, and the
   upstream test reports that as a failure. The Python side does the product in f64 and rounds once
   to match.

Point 3 is worth keeping in mind for any future SIMD or GPU kernel: a path that does not contract
will disagree with the reference encoder on exact halves, rarely and silently.

### 6.5 Assertion build

`cmake -B build-asan -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGGML_SANITIZE_ADDRESS=ON -DGGML_SANITIZE_UNDEFINED=ON`

| Check | Result |
|---|---|
| `test-quantize-fns` under ASan + UBSan | exit 0, 0 failures |
| `test-backend-ops -o MUL_MAT` under ASan + UBSan | exit 0, 0 sanitizer reports |
| Sanitizer findings in `q4_hqq` code | none |
| Sanitizer findings elsewhere | the upstream `q4_1` issue described above |

### 6.6 Scalar versus SIMD

Built a second tree with `-DGGML_CPU_GENERIC=ON`, which routes `ggml_vec_dot_q4_hqq_q8_0` to the
scalar path in `quants.c` instead of the NEON kernel in `arch/arm/quants.c`.

| Metric | scalar (`GGML_CPU_GENERIC`) | NEON |
|---|---|---|
| quantization error | 0.001144 | 0.001144 |
| reference error | 0.000000 | 0.000000 |
| vec_dot error | 0.007968 | 0.007968 |
| all 9 edge case RMSEs | identical to 6 significant figures | identical |
| generated text, same seed | byte identical | byte identical |

Logs: `logs/task2/test_quantize_fns_generic.log`, `logs/task2/test_quantize_fns_neon.log`,
`logs/task2/gen_scalar.log`, `logs/task2/gen_neon.log`.

### 6.7 Option A versus Option B, measured

The spec frames Option B (`vec_dot_type = GGML_TYPE_Q8_1`, reading the precomputed activation sum)
as the faster choice. It was implemented as a throwaway patch and measured against Option A, the
shipped one. End to end `llama-bench` numbers drifted more than 10 percent between sessions on this
thermally loaded laptop, so the comparison uses `test-backend-ops perf -o MUL_MAT -b CPU` on the
decode shape (`m=4096, n=1, k=14336`), with `q4_1` measured in the *same* session as a normalizer.
Medians of 4 runs:

| Variant | q4_hqq us/run | q4_1 us/run (same session) | q4_hqq / q4_1 |
|---|---|---|---|
| Option A (`q8_0`, shipped) | 450.0 | 373.7 | 83.0 % |
| Option B (`q8_1`, throwaway) | 427.5 | 347.9 | 81.4 % |

**Option B is not faster.** The saving in the dot loop is cancelled by the more expensive `q8_1`
activation quantization, which has to compute the block sum that `q8_0` does not. The ratio is
stable across sessions where the absolute numbers are not, and Option B lands slightly on the wrong
side of it.

The real reason `q4_hqq` trails `q4_1` by about 17 percent on CPU is elsewhere: this build has
`__ARM_FEATURE_MATMUL_INT8`, so `q4_1` runs with `nrows = 2` and the two row `vmmlaq_s32` kernel,
while `q4_hqq` runs with `nrows = 1`. Writing that two row path is the change that would close the
gap. Logs: `logs/task2/bench_option_a.log`, `logs/task2/bench_option_b.log`.

### 6.8 Three requirements of spec v2.0 that measurement contradicts

Spec v2.0 tightened section 5.3 and section 7.1. Three of its requirements were implemented as experiments,
measured, and then **not adopted**, because the measurements say they make the format worse. Each is
recorded here with the number that decided it, and each is listed in the deviation register.

**1. `Z_MAX = 256` cap on `|zero|` (v2 section 5.3) makes quality worse, 1.8x to 5.9x.**

The v2 derivation bounds the f16 rounding error of `zero` measured *in quantization levels*. The
step it misses is that the decoder computes `w = (q - zero)/scale`: the error that reaches the
weights is `dzero/scale`, and since `dzero ~ |zero|*2^-11` and `|zero|/scale = |min|`, that error is
`|min|*2^-11` **regardless of `Z_MAX`**. Capping `scale` therefore buys nothing in weight space,
while it directly shrinks the number of levels the block uses: `span = Z_MAX/r` with
`r = |min|/(max-min)`. At `r = 1000` and `Z_MAX = 256` the block collapses to 0.26 levels.

Measured over 400 random blocks per ratio with `scripts/qvac/tools/q4_hqq_zmax_probe.c`, which
models the round trip in double with real f16 rounding (log: `logs/task2/q4_hqq_zmax_probe.log`):

| r = \|min\|/(max-min) | mean RMSE, shipped | mean RMSE, v2 Z_MAX=256 | blocks where shipped is better |
|---|---|---|---|
| 20 | 8.96e-01 | 1.14e+00 | 97 % |
| 100 | 1.79e-01 | 1.06e+00 | 100 % |
| 1e3 | 9.13e-02 | 5.85e-01 | 100 % |
| 1e4 | 1.11e-01 | 1.95e-01 | 68 % |
| 1e5 | 1.12e-01 | 1.85e-01 | 62 % |
| 1e6 | 9.96e-02 | 1.80e-01 | 61 % |
| 1e7 | 1.01e-01 | 1.89e-01 | 65 % |

The shipped encoder caps `scale` at `65504/|min|`, which for moderate ratios does not bind at all,
so the block keeps its full 15 levels. That is why it wins everywhere.

**2. The `(max - min)/2` absolute bound (v2 section 7.1) is unreachable for this format.**

For the narrow-band case the half-band is 1.55e-5, while the actual error is 6.11e-2 for the shipped
encoder and 4.04e-1 under the v2 encoder - three to four orders of magnitude above the bound. The
cause is the same offset term as above: the reconstruction carries `|min|`, and f16 precision on it
floors the error at `~|min|*2^-11 = 0.49` for `|min| = 1000`. v2 section 5.3's claim that "the residual
absolute error stays bounded by `(max - min)/2`" does not hold. The test asserts `|min|*2^-10`
instead, which is the bound the arithmetic actually supports.

**3. Clamping `scale` up to the smallest f16 *normal* (v2 section 5.3) is a 556x regression.**

`scale` is the reciprocal of the step, so the largest representable magnitude in a block is
`15/scale`. Clamping `scale` up to 6.1035e-5 caps that at 245760 and destroys any block wider than
that. On the extreme-dynamic-range case (1e-6 to 1e6) RMSE goes from 9.59e+02 to 5.33e+05. The real
concern behind the requirement - a `scale` that rounds to zero making the decoder emit `inf` - is
already handled, one layer down, by `is = s != 0 ? 1/s : 0` in `dequantize_row_q4_hqq`. Case 9 of
the edge-case test pins that behavior.

**Not adopted either: `low nibble = even index` (v2 section 5.1).** v2 section 5.1 and Appendix B/C require
`2*j` / `2*j+1` packing, while the closing line of Appendix C requires that "nibble unpacking
follows the `Q4_1` code path unchanged, since the packing convention is identical". In this tree
`quantize_row_q4_1_ref` (`ggml/src/ggml-quants.c:174-181`) packs element `j` low and `j + 16` high,
so the two requirements are mutually exclusive. The repository convention is kept, which satisfies
the Appendix C sentence and lets every kernel reuse the `Q4_1` unpacking; it is applied consistently
across the encoder, decoder, NEON kernel and all Metal kernels.

## 7. Reproduction

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build -j

bash scripts/qvac/00_fetch_model.sh    # HF_TOKEN optional, falls back to the ungated mirror
bash scripts/qvac/01_convert_gguf.sh
bash scripts/qvac/02_quantize.sh
bash scripts/qvac/03_bench.sh
bash scripts/qvac/04_perplexity.sh
bash scripts/qvac/05_subjective.sh

bash scripts/qvac/06_test_env.sh      # jinja2 + the git-lfs vocab files, needed for 60/60
ctest --test-dir build

# the hqq optimizer variant
GGML_Q4_HQQ_OPT=1 ./build/bin/llama-quantize \
    models/qvac/gguf/llama-3.2-3b-f16.gguf models/qvac/gguf/llama-3.2-3b-q4_hqq_opt.gguf Q4_HQQ
```

The two extra build trees used for correctness only:

```bash
cmake -B build-asan    -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLAMA_CURL=OFF \
                       -DGGML_METAL=OFF -DGGML_BLAS=OFF \
                       -DGGML_SANITIZE_ADDRESS=ON -DGGML_SANITIZE_UNDEFINED=ON -DLLAMA_BUILD_TESTS=ON
cmake -B build-generic -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
                       -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_CPU_GENERIC=ON -DLLAMA_BUILD_TESTS=ON
```

Every script is idempotent and writes the exact command line it ran into its log.
