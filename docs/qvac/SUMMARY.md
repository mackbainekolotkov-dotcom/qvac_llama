# Summary

## What was modified

**Task 1.** `scripts/qvac/*.sh` for fetch, convert, quantize, bench, perplexity and the subjective
comparison. Each is `set -euo pipefail`, env parameterised, idempotent, and echoes its exact command
line into `logs/`.

**Task 2.** `GGML_TYPE_Q4_HQQ = 43` appended at the end of `ggml_type`, `block_q4_hqq` with the
static assert, reference encoder/decoder, row validation, the ftype chain up to
`LLAMA_FTYPE_MOSTLY_Q4_HQQ = 42`, the `llama-quantize` options table, `gguf-py` constants, and
`q4_hqq` in the KV cache type list. Kernels: `ggml_vec_dot_q4_hqq_q8_0` (scalar plus ARM NEON) and
Metal `mul_mv`, `mul_mv_ext`, `mul_mm`, `mul_mm_id`, `get_rows`, `cpy`, `set_rows` and flash
attention. Bonus: an optional real HQQ zero point optimizer, encoder only, behind `GGML_Q4_HQQ_OPT=1`.

**Task 3.** `--mmproj-backend {auto|cpu|<device>}` plus a startup log line naming the projector
device and the model device, and, when a device is named explicitly, a line stating whether the
whole projector graph ended up on it or how many ops fell back to the CPU.

## Deviation register

Every point where this branch does not follow the brief or spec v2.0 literally.

| # | Required | Delivered | Why |
|---|---|---|---|
| D1 | `quantize_row_q4_hqq` | `quantize_row_q4_hqq_ref` + `quantize_q4_hqq` driver | Tree convention: `_ref` is the scalar encoder, the unsuffixed name is the driver entry called from `ggml_quantize_chunk`. |
| D2 | `src/llama.cpp`, `src/llama-quantize.cpp`, vec_dot in `ggml-quants.c` | Pinned-commit paths (`src/llama-quant.cpp`, `ggml-cpu/arch/*/quants.c`, ftype name in `llama-model-loader.cpp`) | Upstream refactored after the brief. Row-by-row map in REPORT. |
| D3 | `ggml_vec_dot_q4_hqq_q8_0` | Signature kept; Option B built as a throwaway and measured | Option B is *not* faster here: 81.4 % vs 83.0 % of `q4_1`. Reported, not silently substituted. |
| D4 | Format named HQQ | Affine min-max as specified, plus a real HQQ optimizer behind `GGML_Q4_HQQ_OPT=1` | The brief's formula is not the HQQ algorithm. Both are measurable on one binary. |
| D5 | GPU run required, Task 2c a bonus | Native Metal kernels + device attribution table | Exceeds the SHOULD tier: the `Q4_HQQ` matmuls really execute on device. |
| D6 | `meta-llama/Llama-3.2-3B-Instruct` | `unsloth/Llama-3.2-3B-Instruct`, rev `006f5dcd` | Gated repo, no token. Same weights; `HF_TOKEN` switches back. |
| D7 | v2 section 5.3 `Z_MAX = 256` cap on `\|zero\|` | Cap at the f16 max instead | Measured regression of 1.8x-5.9x. The v2 derivation counts the `zero` error in levels, but the decoder divides by `scale`, so the weight-space error is `\|min\|*2^-11` regardless of `Z_MAX`; the cap only removes levels. REPORT section 6.8. |
| D8 | v2 section 7.1 absolute error <= `(max-min)/2` | Asserted `\|min\|*2^-10` | The bound is unreachable by 3-4 orders of magnitude for any encoder: the reconstruction carries the offset `\|min\|`. REPORT section 6.8. |
| D9 | v2 section 5.1 `low nibble = even index` | `Q4_1` convention (`j` low, `j+16` high) | v2 contradicts itself: Appendix C also requires unpacking to follow the `Q4_1` path unchanged, and in this tree that path is not even-index. Repo convention wins; applied consistently in encoder, decoder, NEON and Metal. |
| D10 | `-no-cnv` in the run commands | `-st` | The flag does not exist at the pinned commit (`error: invalid argument: -no-cnv`); `-st` is the single-turn equivalent actually used. |

## Issues faced

**The brief's paths are from an older tree.** `src/llama-quantize.cpp` is now `src/llama-quant.cpp`
and the CPU kernels moved under `ggml/src/ggml-cpu/arch/*/` behind the `_generic` renaming trick in
`arch-fallback.h`. The real integration set came from `rg -n 'Q4_1|GGML_TYPE_Q4_1'`.

**Q4_HQQ is Q4_1 with the parameters moved.** `d = 1/scale` and `m = -zero/scale` maps one onto the
other exactly, so `Q4_1` - not `Q4_0` - is the reference for the arithmetic and every kernel, while
`Q4_0` only shows where to hook in. The measurements confirm it: PPL 11.4726 against 11.4748.

**Option B is not faster, contrary to the brief.** Measured, not estimated: Option A runs at 83.0
percent of `q4_1` at the kernel level and Option B at 81.4 (medians of 4, `q4_1` normalised
in-session). The saving in the dot loop is cancelled by the `q8_1` activation quantization, which
must compute the very sum that makes it attractive. The real ~17 percent gap to `q4_1` is that this
build has `__ARM_FEATURE_MATMUL_INT8`, so `q4_1` uses `nrows = 2` and the two-row `vmmlaq_s32`
kernel while `q4_hqq` uses `nrows = 1`.

**Affine min-max is not HQQ.** Real HQQ was implemented as well, encoder-only: 20 iterations per
block alternating a generalized soft threshold on the residual with a closed-form zero-point update.
Same layout, decoder, speed and file size; PPL 11.4726 to 11.3808, quantization 4.8 s to 37 s.

**The FP16 `zero` bound is where the format actually bites.** Capping the scale so `zero = -min*scale`
also fits f16 is a refinement on the brief, which asks only about the scale. Beyond that the
reciprocal storage sets a hard floor: the reconstruction `(q-zero)/scale` carries the offset `|min|`,
so f16 precision floors the error at `|min|*2^-11` no matter how narrow the block. This is why
Q4_HQQ is *more* robust than Q4_1 on extreme-range blocks yet *structurally worse* on a narrow band
far from zero, and why spec v2's `Z_MAX = 256` and `(max-min)/2` requirements were measured and
rejected (D7, D8; REPORT section 6.8). The same test under UBSan exposed latent undefined behaviour in
upstream `quantize_row_q4_1_ref` on denormals, confirmed pre-existing.

**No subjective score is reported.** 25 side by side generations across five types did not separate
them; all produced coherent text and working Python. Perplexity separates them, five short prompts
do not, so the 1-5 column was dropped rather than invented. See `APPENDIX.md`.

**`--mmproj-backend` largely already exists upstream.** The pinned commit has `-mmdev` and
`--mmproj-offload`, and `clip_context_params` already carries a device. The new flag is the
vocabulary the brief asks for layered onto that mechanism, with `auto` on the existing code path so
default behaviour is unchanged. What was genuinely missing is the guard: a device can initialise and
still run only part of the graph, and upstream said so only as a generic "performance will be
suboptimal" note. Adding the explicit report surfaced a bug behind it - in the flash attention auto
path the second `reserve_compute_meta()` result was discarded, so the op list described the graph
that had just been rejected rather than the one that runs.

**Peak RSS is the wrong instrument for the KV cache here.** It puts `f16` 4.5 MiB *below* `q8_0`,
because the `f16` run is the one that pages on 16 GiB. `peak memory footprint`, from the same
`/usr/bin/time -l` block, tracks the declared `KV self size` to within 1.1 MiB. Both are in the
table. Flash attention is not required for a quantized V cache at this commit, on either backend.

## Suggestions for further improvement

1. Give `q4_hqq` the `nrows = 2` I8MM path. This, not the `q8_1` switch, closes the CPU gap.
2. Add an x86 AVX2 kernel; only NEON, Metal and scalar exist today.
3. Let the HQQ optimizer refine the scale too, and expose its knobs as real CLI options.
4. Reuse the repack infrastructure that gives `Q4_0` its `CPU_REPACK` speedup.
5. If the format is ever revised, store the forward scale instead of the reciprocal. It costs
   nothing and removes the one case where Q4_HQQ is structurally worse than Q4_1.
