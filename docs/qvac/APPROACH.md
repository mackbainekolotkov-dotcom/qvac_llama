# Approach, decisions and results

This document is the reasoning behind the commits. It answers three questions: how each task was
approached, which technical decisions were made and why, and what the measurements say.

Everything below is backed by a committed log; the per-task readmes
([`TASK1.md`](TASK1.md), [`TASK2.md`](TASK2.md), [`TASK3.md`](TASK3.md)) say which one.

---

## 1. How the work was framed

The assignment names three tasks, but only the second one is really open-ended, so the effort went
there. Two rules governed the whole submission:

**Every number in the report comes from a committed log.** Nothing is quoted from memory or from a
run that was not saved. Where a claim could not be measured, it is stated as unmeasured rather than
estimated.

**Where the brief and the codebase disagree, the disagreement is documented, not silently
resolved.** The brief was written against an older tree, and one of its formulas is not what its
name says. Each such point is a numbered entry in the deviation register in
[`SUMMARY.md`](SUMMARY.md).

---

## 2. Task 1 - convert, quantize, benchmark

Straightforward, with one obstacle: `meta-llama/Llama-3.2-3B-Instruct` is gated and no access token
was available. The fetch script tries the gated repository first, falls back to an ungated mirror of
the same weights, and records which one it actually used. The `GatedRepoError` is in the log in the
library's own words, so the substitution is visible rather than asserted.

The measurement decision worth explaining is that there are **two** latency tables, not one.

- The brief asks for first-token latency on its two example prompts. Those prompts are 39 and 44
  tokens, so their prompt-eval time is dominated by length, and comparing 39-token latency for one
  quantization type against 44-token latency for another measures nothing.
- So the per-prompt numbers are reported for the brief, and a second table uses a fixed 512-token
  prompt for every type. That second table is the one that supports statements like "Q4_0 generates
  2.9x faster than F16".

Both are in [`TASK1.md`](TASK1.md), with the token counts printed next to the times so the reader
can see why they are not comparable.

**Results.** Q4_0 against F16: 3.3x smaller on disk, 2.9x faster generation, 1.9 GiB less resident
memory, +0.647 perplexity. Re-running the conversion and quantization scripts reproduces the GGUF
files **byte for byte**, so those numbers rest on a reproducible artifact rather than a one-off.

---

## 3. Task 2 - the Q4_HQQ format

### 3.1 The observation that shaped everything

`Q4_HQQ` as specified is `Q4_1` with the two block parameters moved into quantized space:

```
Q4_1:    w = d*q + m           stores (d, m)
Q4_HQQ:  w = (q - zero)/scale  stores (scale, zero),  d = 1/scale, m = -zero/scale
```

Same 20 bytes per 32 weights, same nibble layout, same arithmetic shape. So `Q4_1` - not `Q4_0` - is
the reference for the maths and for every kernel; `Q4_0` only shows *where* to hook into the
codebase. Recognising this early meant the NEON and Metal kernels could be derived from the existing
`q4_1` ones rather than written from scratch, and it gives a free correctness oracle: on the same
data the two types must produce near-identical error. They do - perplexity 11.4726 against 11.4748.

The inversion is not cosmetic, though. Storing `scale` instead of the step puts the FP16 rounding
error into a **reciprocal**, and that is the source of every edge case in the format.

### 3.2 Q8_0 versus Q8_1: measured, not assumed

The dot product expands to

```
sum(w*a) = (d8/scale) * ( sum(q*s) - zero*sum(s) )
```

The `sum(s)` term is the problem: `block_q8_0` does not store it, `block_q8_1` does. That is exactly
why `Q4_1` pairs with `Q8_1`. The brief's required signature is `ggml_vec_dot_q4_hqq_q8_0`, which
implies `Q8_0`.

The obvious reading is that the brief's signature costs performance. Rather than assume that, both
were built and measured at the kernel level, with `q4_1` normalised in the same session:

| Variant | q4_hqq us/run | q4_1 us/run | ratio |
|---|---|---|---|
| Option A (`q8_0`, shipped) | 450.0 | 373.7 | 83.0 % |
| Option B (`q8_1`, throwaway) | 427.5 | 347.9 | 81.4 % |

**Option B is not faster.** The saving inside the dot loop is cancelled by the more expensive `q8_1`
activation quantization, which has to compute the very sum that made it attractive. So the brief's
signature costs nothing, and the shipped kernel computes `sum(s)` itself with a dot against a vector
of ones.

The real ~17 % CPU gap to `q4_1` is unrelated: this build has `__ARM_FEATURE_MATMUL_INT8`, so `q4_1`
runs with `nrows = 2` and a two-row `vmmlaq_s32` kernel while `q4_hqq` runs with `nrows = 1`. That,
not the activation type, is what would close it. It is listed as the first suggested improvement.

### 3.3 The numerical edge cases, and where the format is genuinely weak

Because the stored `scale` is a reciprocal, four cases need explicit handling in the encoder, before
any kernel work:

| Case | Handling |
|---|---|
| `max == min` | `scale = 1`, `zero = -min` |
| `scale` overflows FP16 | clamp before computing `q`, so encoder and decoder agree |
| `zero` overflows FP16 | cap `scale` at `65504/abs(min)` first - a refinement on the brief, which only mentions the scale; clamping the scale alone still sends `zero` to inf |
| `scale` underflows to 0 | the decoder guards with `is = s != 0 ? 1/s : 0`, so output stays finite |

The encoder also stores both parameters as FP16 and **reads them back** before computing `q`, so it
quantizes against exactly the values the decoder will see.

Testing this produced two findings that are stated plainly rather than buried:

- **Q4_HQQ is more robust than Q4_1 on extreme-range blocks.** With values spanning 1e-6 to 1e6 in
  one block, `q4_1` overflows its `d` to inf and dequantizes to inf; `q4_hqq` stays finite. Running
  the same test under UBSan surfaced pre-existing undefined behaviour in upstream
  `quantize_row_q4_1_ref` on denormals.
- **Q4_HQQ is structurally worse than Q4_1 on a narrow band far from zero, and no encoder can fix
  it.** `q4_1` stores the step, a small number FP16 holds comfortably; `q4_hqq` stores its
  reciprocal, which saturates. This is a property of the layout. If the format were ever revised,
  storing the forward scale would remove it at no cost.

### 3.4 Affine min-max is not HQQ

The brief's formula is plain asymmetric min-max quantization carrying the HQQ name. Real HQQ holds
the scale fixed and iteratively refines the zero-point under an lp loss with p < 1.

Both are shipped. The specified formula is the default; the real optimizer is encoder-only, behind
`GGML_Q4_HQQ_OPT=1`, leaving the block layout, the decoder and every kernel untouched, so the two
can be measured on the same binary. It moves perplexity from 11.4726 to 11.3808 and raises
quantization time from 4.8 s to 37 s. Calling this out is the difference between shipping "renamed
Q4_1" and shipping HQQ.

### 3.5 How correctness was established

Tests in increasing order of strength:

1. `test-quantize-fns` round trip and `vec_dot` against a float64 reference.
2. Nine numerical edge cases, including the two adversarial ones above.
3. `test-backend-ops`, comparing the Metal path against the CPU reference.
4. ASan + UBSan: zero findings in `q4_hqq` code.
5. Scalar versus NEON: identical to six significant figures, byte-identical generation.
6. **An independent reimplementation.** `gguf-py` carries a NumPy quantizer per type and a ctypes
   harness that compares it against `libggml` bit for bit. `Q4_HQQ` was registered in `constants.py`
   but missing from `quants.py`, so Python could read a Q4_HQQ file's metadata and then fail on its
   tensors. Filling that in gives a second implementation written from the format description rather
   than from the C source, so the comparison checks the C encoder instead of restating it.

The sixth is the strongest, and getting it to match exactly pinned down three properties of the C
encoder that were otherwise undocumented:

- `zero` is derived from the **unrounded** scale, and only then stored as FP16;
- `roundf` sends halves away from zero, while `np.round` sends them to even;
- the compiler contracts `x*scale + zero` into an **fma**, so it rounds once. Doing the multiply and
  the add separately moves values that land exactly on a half - one block in 262144 on the test
  data.

The third point is worth carrying forward: any future SIMD or GPU kernel that does not contract will
disagree with the reference encoder on exact halves, rarely and silently.

### 3.6 KV cache

`q4_hqq` is accepted for both K and V. The widely repeated claim that a quantized V cache requires
flash attention was **tested rather than assumed**, and does not hold at this commit: K-only and
V-only both work without `-fa`, on CPU and on Metal.

At `-c 8192` the cache is 280 MiB against 896 MiB for F16, and a long-context retrieval probe (fact
at position 0, ~5000 tokens of filler, question at the end) returns the correct answer for every
cache type. `head_dim` for this model is 128, a multiple of 32; a model whose head_dim is not would
be unable to use the type, which is noted as a general limitation.

Measuring the memory produced its own lesson: `maximum resident set size` puts the F16 cache *below*
`q8_0`, because the F16 run is the one that pages on a 16 GiB machine. `peak memory footprint`, from
the same `/usr/bin/time -l` block, tracks the declared cache size to within 1.1 MiB. Both columns
are reported, with the reason they disagree.

### 3.7 GPU

Metal, since that is the available backend: `get_rows`, `mul_mv`, `mul_mv_ext`, `mul_mm`,
`mul_mm_id`, `cpy`, `set_rows` and flash attention.

A GPU kernel that quietly falls back to the CPU is a failed deliverable, so the claim is made per
op, not per run: `test-backend-ops` counts for `q4_hqq` on each backend, and at `-ngl 99` the graph
has `graph splits = 2` - one CPU segment, the `token_embd` lookup that stays on host as `q6_K`.
Where `MTL0` reports `not supported` for some `CPY` and `SET_ROWS` shapes, `q4_1` reports exactly
the same counts, so `q4_hqq` is at parity with the type it mirrors rather than missing something.

---

## 4. Task 3 - `--mmproj-backend`

Most of the mechanism already existed upstream: the pinned commit has `-mmdev` and
`--mmproj-offload`, and `clip_context_params` already carries a device. The honest description of
the flag is that it is the vocabulary the brief asks for, layered onto that mechanism, with `auto`
left on the existing code path so default behaviour is bit-identical.

The requirement that actually needed care is "the base model backend should not change", because it
is easy to claim and easy to get wrong. It is verified by looking at where the *model tensors* went,
not at what the projector reported:

| Run | Projector | Language model |
|---|---|---|
| `--mmproj-backend cpu -ngl 99` | `CLIP using CPU backend` | `offloaded 31/31 layers to GPU` |
| `--mmproj-backend MTL0 -ngl 0` | `CLIP using MTL0 backend` | `offloaded 0/31 layers to GPU` |

Opposite placements in the two runs, and `-ngl` alone decides the model's. Image encode time moves
by 11.4x between them, so the projector physically moved rather than being relabelled.

**What was genuinely missing upstream** is a guard. A device can exist, initialise, and still run
only part of the projector graph; upstream reported that as a generic "performance will be
suboptimal" warning, which reads as a tuning note rather than as the request not being honoured.
When a device is named explicitly, the outcome is now stated directly - either the whole graph is on
it, or how many of the graph's ops fell back. `BLAS` is the useful adversarial case: it is accepted
as an accelerator target but implements a small fraction of the graph, and the run reports `238 of
420 ops` falling back while still exiting cleanly.

Adding that guard exposed a real bug underneath it: in the flash-attention auto path the second
`reserve_compute_meta()` result was discarded, so upstream's own op list described the graph that
had just been *rejected* rather than the one that runs. Keeping the result changes the BLAS report
from `226 of 396` to `238 of 420`.

---

## 5. Results at a glance

| Type | Size | bpw | PPL (c=512) | dPPL vs F16 | Gen t/s (CPU) | Gen t/s (Metal) |
|---|---|---|---|---|---|---|
| F16 | 5.99 GiB | 16.00 | 10.5709 | - | 19.11 | 21.47 |
| Q4_0 | 1.79 GiB | 4.50 | 11.2183 | +0.647 | 55.97 | 64.17 |
| Q4_1 | 1.95 GiB | 5.00 | 11.4748 | +0.904 | 45.64 | 57.67 |
| Q4_HQQ | 1.95 GiB | 5.00 | 11.4726 | +0.902 | 36.92 | 57.55 |
| Q4_HQQ + optimizer | 1.95 GiB | 5.00 | 11.3808 | +0.810 | 36.50 | - |

`Q4_HQQ` beating `Q4_0` on perplexity is expected and uninformative - it is 11 % larger. The honest
comparison is against `Q4_1` at equal block size, where the two are within noise, which is what the
format equivalence predicts. On Metal `Q4_HQQ` matches `Q4_1`; the CPU gap is the `nrows` issue.

KV cache at `-c 8192`: 280 MiB for `q4_hqq` against 896 MiB for `f16`, long-context probe passing
for every type.

Test suite: `ctest` 60/60.

---

## 6. What is deliberately not here

- **No subjective 1-5 quality column.** 25 side-by-side generations across five types did not
  separate them; every type produced coherent text and working Python. Perplexity separates them,
  five short prompts do not, so the column was dropped rather than invented. The generations are in
  [`APPENDIX.md`](APPENDIX.md).
- **No x86 kernel.** Scalar, ARM NEON and Metal only.
- **No `nrows = 2` path**, which is the single change that would close the CPU gap.
- **Three requirements from a later revision of the spec were implemented, measured and then
  rejected** as making the format worse. The measurements and the 40-line harness that produced them
  are in [`REPORT.md`](REPORT.md) and `scripts/qvac/tools/q4_hqq_zmax_probe.c`.
