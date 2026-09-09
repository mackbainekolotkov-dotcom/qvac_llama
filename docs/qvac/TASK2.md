# Task 2 - a new quantization format, Q4_HQQ

Self-contained checklist for the second assignment task. Every row names the code that implements it
and the evidence that it works. Nothing is asserted without a committed log or a test behind it.

Single submitted tree, derived from upstream commit `0b5be7e4a25862bc2777d0c47eae18788a8c963a`.
Eight commits, from the type definition to the gguf-py implementation, 30 files.

---

## 1. Status at a glance

| # | Requirement | Status | Where |
|---|---|---|---|
| 1 | `block_q4_hqq`, 20 B per 32 weights, static assert | done | `ggml/src/ggml-common.h:216-222` |
| 2 | Enum entry, appended at the end | done | `ggml/include/ggml.h:433` (`= 43`) |
| 3 | `type_traits[]` and `type_traits_cpu[]` | done | `ggml/src/ggml.c:708`, `ggml-cpu/ggml-cpu.c:259` |
| 4 | `quantize_row_q4_hqq_ref`, `dequantize_row_q4_hqq`, `quantize_q4_hqq` | done | `ggml/src/ggml-quants.h:21,51,103` |
| 5 | `ggml_vec_dot_q4_hqq_q8_0`, scalar then SIMD | done | `ggml-cpu/quants.c:371`, `ggml-cpu/arch/arm/quants.c:923` |
| 6 | ftype chain and human-readable name | done | `include/llama.h:159`, `src/llama-model-loader.cpp:44` |
| 7 | Quantization driver and CLI option | done | `src/llama-quant.cpp:832`, `tools/quantize/quantize.cpp:39` |
| 8 | `ggml_quantize_chunk` dispatch | done | `ggml/src/ggml.c:7995` |
| 9 | KV cache type `q4_hqq` for K and V | done | `common/arg.cpp:311` |
| 10 | Picked up by the existing tests | done | `tests/test-quantize-fns.cpp`, `tests/test-backend-ops.cpp` |
| 11 | **Bonus** SIMD kernel | done, ARM NEON | `ggml-cpu/arch/arm/quants.c` |
| 12 | **Bonus** GPU kernels | done, Metal | `ggml/src/ggml-metal/kernels/` |
| 13 | **Bonus** perplexity benchmark | done | `logs/task1/ppl_*.log` |
| 14 | **Bonus** real HQQ optimizer | done, `GGML_Q4_HQQ_OPT=1` | `ggml/src/ggml-quants.c:198-241` |
| 15 | Python side (`gguf-py`) | done | `gguf-py/gguf/constants.py`, `gguf-py/gguf/quants.py` |

---

## 2. The format

```c
#define QK4_HQQ 32
typedef struct {
    ggml_half scale;            // 15 / (max - min)
    ggml_half zero;             // -min * scale
    uint8_t   qs[QK4_HQQ / 2];  // 4-bit, 2 per byte
} block_q4_hqq;                 // 2 + 2 + 16 = 20 B per 32 weights = 5.00 bpw
```

`q = round(x*scale + zero)` clamped to `[0, 15]`, `x = (q - zero)/scale`.

**The one insight that shapes the whole task:** this is `Q4_1` with the two parameters moved into
quantized space, `d = 1/scale` and `m = -zero/scale`. So `Q4_1`, not `Q4_0`, is the reference for the
arithmetic and for every kernel; `Q4_0` only shows where to hook into the codebase. Nibble packing
follows the repository convention: element `j` low, element `j + 16` high, so every kernel reuses the
`Q4_1` unpacking unchanged.

The inversion is not cosmetic. Storing `scale` rather than the step puts the f16 rounding error into
a reciprocal, which is where every edge case below comes from.

### Numerical edge cases

| Case | Handling |
|---|---|
| `max == min` | `scale = 1.0`, `zero = -min`, all `q` deterministic |
| `scale` overflows f16 | clamped before `q` is computed, so encoder and decoder agree |
| `zero` overflows f16 | `scale` capped at `65504/\|min\|` first, which is a refinement on the brief: clamping the scale alone still sends `zero` to inf |
| `scale` underflows to 0 | `dequantize_row_q4_hqq` guards with `is = s != 0 ? 1/s : 0`, so output stays finite |
| NaN/Inf in the source | hard error in `llama-quantize`, naming the tensor and the index |

The encoder stores both parameters as f16 and **reads them back** before computing `q`, so it
quantizes against exactly the values the decoder will see.

---

## 3. The `Q8_0` versus `Q8_1` trade-off

With `w = (q - zero)/scale` and `a = d8*s`:

```
sum(w*a) = (d8/scale) * ( sum(q*s) - zero*sum(s) )
```

The second term needs `sum(s)`, which `block_q8_0` does not store and `block_q8_1` does. The brief's
signature `ggml_vec_dot_q4_hqq_q8_0` implies `Q8_0`, so **Option A** is what ships: the kernel
computes `sum(s)` itself, with a dot against a vector of ones.

Option B was not estimated, it was built as a throwaway patch and measured:

| Variant | q4_hqq us/run | q4_1 us/run (same session) | ratio |
|---|---|---|---|
| Option A (`q8_0`, shipped) | 450.0 | 373.7 | 83.0 % |
| Option B (`q8_1`, throwaway) | 427.5 | 347.9 | 81.4 % |

**Option B is not faster.** The saving in the dot loop is cancelled by the more expensive `q8_1`
activation quantization, which has to compute the very sum that makes it attractive. Logs:
`logs/task2/bench_option_a.log`, `logs/task2/bench_option_b.log`.

The real ~17 % CPU gap to `q4_1` is elsewhere: this build has `__ARM_FEATURE_MATMUL_INT8`, so `q4_1`
runs with `nrows = 2` and the two-row `vmmlaq_s32` kernel while `q4_hqq` runs with `nrows = 1`.
That, not the `q8_1` switch, is what would close it.

---

## 4. Correctness

| Check | Result | Evidence |
|---|---|---|
| `test-quantize-fns` | pass | round-trip RMSE 0.001144, identical to `q4_1`; `vec_dot` error 0.007968 vs f64 reference |
| 9 numerical edge cases | pass | section 6.3 of [`REPORT.md`](REPORT.md) |
| `test-backend-ops -o MUL_MAT` | pass, 3/3 backends | `q4_hqq` `OK` on `MTL0`, not `not supported` |
| `test-backend-ops` full | pass, 157.6 s | `logs/task2/ctest_full.log` |
| ASan + UBSan | 0 findings in `q4_hqq` code | `logs/task2/asan_*.log` |
| Scalar vs NEON | identical to 6 significant figures, byte-identical generation | `logs/task2/test_quantize_fns_*.log`, `gen_scalar.log`, `gen_neon.log` |
| NumPy vs C, bit for bit | exact match | `logs/task2/gguf_py_vs_c_q4_hqq.log` |

The last row is the strongest one. `gguf-py` carries a NumPy implementation per type and a ctypes
harness that compares it against `libggml`. Writing `Q4_HQQ` there from the format description, not
from the C source, gives a second implementation that checks the first:

```
Quantization to Q4_HQQ matches exactly
Dequantization from Q4_HQQ matches exactly
Dequantization from random f16 data as Q4_HQQ matches exactly
```

over 8 x 1024 x 1024 values, plus 39.3 MB of real model weights re-quantized in NumPy with zero
differing bytes against what `llama-quantize` wrote. Reaching that pinned down three properties of
the C encoder - `zero` derives from the *unrounded* scale, `roundf` sends halves away from zero, and
the compiler contracts `x*scale + zero` into an fma so it rounds once. See section 6.4 of
[`REPORT.md`](REPORT.md); the fma point matters for any future kernel.

---

## 5. Quality and speed

CPU, `-ngl 0 -t 8`, perplexity on `wiki.test.raw` at `-c 512`, 40 chunks, identical for every row.

| Type | File size | bpw | PPL | dPPL vs F16 | Gen (t/s) |
|---|---|---|---|---|---|
| F16 | 5.99 GiB | 16.00 | 10.5709 | - | 19.11 |
| Q4_0 | 1.79 GiB | 4.50 | 11.2183 | +0.647 | 55.97 |
| Q4_1 | 1.95 GiB | 5.00 | 11.4748 | +0.904 | 45.64 |
| Q4_HQQ | 1.95 GiB | 5.00 | 11.4726 | +0.902 | 36.92 |
| Q4_HQQ + optimizer | 1.95 GiB | 5.00 | 11.3808 | +0.810 | 36.50 |

`Q4_HQQ` beating `Q4_0` on perplexity is expected and uninformative: it is 11 % larger. The honest
comparison is against `Q4_1` at the same block size, and there the two are within noise of each
other - 11.4726 against 11.4748 - which is exactly what the format equivalence predicts.

**The brief's formula is not the HQQ algorithm.** It is plain affine min-max quantization carrying
the HQQ name. Real HQQ holds the scale fixed and refines the zero-point under an lp loss with p < 1.
That is implemented too, encoder-only, behind `GGML_Q4_HQQ_OPT=1`: same layout, same decoder, same
kernels, same file size, same speed. It moves perplexity from 11.4726 to 11.3808 and raises
quantization time from 4.8 s to 37 s.

On Metal `Q4_HQQ` is within noise of `Q4_1` (57.55 against 57.67 t/s), which is the honest
comparison; the CPU gap is the `nrows` issue above.

---

## 6. KV cache

`-c 8192`, Q4_HQQ weights, CPU. Every column of a row comes from one process under
`/usr/bin/time -l` (`logs/task2/kvfull_*.log`).

| K | V | `-fa` | KV self size | Peak footprint | First token | Gen (t/s) | Long-context probe |
|---|---|---|---|---|---|---|---|
| f16 | f16 | on | 896.00 MiB | 3744.3 MiB | 836.9 ms | 41.60 | pass |
| q8_0 | q8_0 | on | 476.00 MiB | 3323.7 MiB | 800.7 ms | 40.67 | pass |
| q4_0 | q4_0 | on | 252.00 MiB | 3099.2 MiB | 810.6 ms | 40.18 | pass |
| q4_hqq | f16 | **off** | 588.00 MiB | 3436.2 MiB | 813.9 ms | 38.46 | pass |
| q4_hqq | q4_hqq | on | 280.00 MiB | 3127.8 MiB | 799.6 ms | 39.33 | pass |

The probe puts a fact at position 0, about 5000 tokens of filler after it, and asks for the fact at
the end. Every cache type returned `84-QUARTZ-1791`.

**The flash-attention constraint was tested, not assumed.** The often quoted "a quantized V cache
needs `-fa on`" does not hold at this commit: K-only and V-only both work without it, on CPU and on
Metal. Verified combinations:

| Configuration | CPU | Metal |
|---|---|---|
| `-ctk q4_hqq`, no flash attention | works | works |
| `-ctv q4_hqq`, no flash attention | works | works |
| `-ctk q4_hqq -ctv q4_hqq -fa on` | works | works |

`head_dim` for Llama 3.2 3B is 128, a multiple of 32, so the type is usable; a model whose head_dim
is not a multiple of `QK4_HQQ` could not use it.

---

## 7. GPU kernels (bonus) and device attribution

Metal: `get_rows`, `mul_mv`, `mul_mv_ext`, `mul_mm`, `mul_mm_id`, `cpy`, `set_rows`, and flash
attention. The last one needed 79 template instantiations in `fa.metal` mirroring the `q4_1` ones,
plus `GGML_TYPE_Q4_HQQ` in the flash-attention type allowlist in `ggml-metal-device.m`.

A GPU kernel that silently falls back to CPU is a failed deliverable, so the claim is made per op:

| Op | MTL0 | BLAS |
|---|---|---|
| MUL_MAT | 6 OK | 12 not supported |
| MUL_MAT_ID | 1 OK | 3 not supported |
| GET_ROWS | 3 OK | 4 not supported |
| FLASH_ATTN_EXT | 300 OK | 336 not supported |
| CPY | 2 OK, 13 not supported | 17 not supported |
| SET_ROWS | 11 OK, 12 not supported | 24 not supported |

The `not supported` lines on `MTL0` for `CPY` and `SET_ROWS` are not a gap: `q4_1` scores exactly
2 OK / 13 and 11 OK / 12 on the same op sets, so `q4_hqq` is at parity with the type it mirrors.
Every `BLAS` line is `not supported` because that backend implements none of these ops for any
quantized type.

At `-ngl 99` the graph has `graph splits = 2`, so exactly one CPU segment - the `token_embd` lookup,
which stays on host as `q6_K`. Everything else runs on device
(`logs/task2/ngl99_attribution.log`):

```
load_tensors:  MTL0_Mapped model buffer size =  1988.90 MiB
load_tensors:   CPU_Mapped model buffer size =   308.23 MiB
load_tensors: offloaded 29/29 layers to GPU
sched_reserve: graph splits = 2
```

---

## 8. Reproduce and verify

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build -j

# acceptance criteria, in order
./build/bin/llama-quantize models/qvac/gguf/llama-3.2-3b-f16.gguf /tmp/q4hqq.gguf Q4_HQQ
./build/bin/llama-cli -m /tmp/q4hqq.gguf -p "What is bitcoin?" -n 100 -t 8 -ngl 0 --seed 42 -st
./build/bin/llama-cli -m /tmp/q4hqq.gguf -p "What is bitcoin?" -n 100 -ngl 99 --seed 42 -st
./build/bin/llama-cli -m /tmp/q4hqq.gguf -c 8192 -fa on \
    --cache-type-k q4_hqq --cache-type-v q4_hqq -p "What is bitcoin?" -n 100 -st

# tests
./build/bin/test-quantize-fns
./build/bin/test-backend-ops -o MUL_MAT

# python against C, bit for bit
python gguf-py/tests/test_quants.py --libggml build/bin/libggml.dylib --type Q4_HQQ

# the hqq optimizer variant
GGML_Q4_HQQ_OPT=1 ./build/bin/llama-quantize <f16> <out> Q4_HQQ
```

Read the recorded results without running anything:

```bash
grep -E "q4_hqq" logs/task2/test_quantize_fns_neon.log     # round trip and vec_dot error
grep -E "INFO|ERROR" logs/task2/gguf_py_vs_c_q4_hqq.log    # numpy vs C
grep "llama_kv_cache: size" logs/task2/kvfull_*.log        # cache sizes per type
grep "84-QUARTZ" logs/task2/longctx_*.log                  # long-context probe
grep -E "graph splits|offloaded" logs/task2/ngl99_attribution.log
```

---

## 9. Known limits, stated rather than hidden

- **Q4_HQQ is worse than Q4_1 on a narrow band far from zero,** and no encoder can fix it. `Q4_1`
  stores the step, a small number f16 holds comfortably; `Q4_HQQ` stores its reciprocal, which
  saturates. The error stays finite and bounded, but it is a property of the layout.
- **Q4_HQQ is more robust than Q4_1 on extreme-range blocks,** where `Q4_1` overflows its `d` to inf
  and dequantizes to inf. Testing this exposed a latent undefined behaviour in upstream
  `quantize_row_q4_1_ref` on denormals, confirmed pre-existing.
- **CPU generation trails `q4_1` by ~17 %** for the `nrows` reason in section 3, not for a reason
  the format imposes. On Metal the two are within noise.
- **No x86 kernel.** Only scalar, ARM NEON and Metal exist.
- **Three requirements of spec v2.0 were measured and rejected** as making the format worse; the
  measurements and the harness are in section 6.8 of [`REPORT.md`](REPORT.md) and
  `scripts/qvac/tools/q4_hqq_zmax_probe.c`.
- **The type id is fork-local.** `43` may later be claimed by a different upstream type. Older
  binaries reject the file with a clean "unknown type" error, which is the expected behaviour. No
  GGUF magic or version change.
