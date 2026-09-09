# Task 3 - `--mmproj-backend {value}`

Self-contained checklist for the third assignment task: how to run it, what it does, and the
evidence that the language model is left alone.

Single submitted tree, derived from upstream commit `0b5be7e4a25862bc2777d0c47eae18788a8c963a`.
Two commits, the flag and the fallback report, 6 files, +112 / -1.

---

## 1. Status at a glance

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | New CLI flag `--mmproj-backend <value>` | done | `common/arg.cpp`, `--help` output in section 3 |
| 2 | Selects the backend for the multimodal projector **only** | done | `clip_ctx: CLIP using <device> backend` moves with the flag |
| 3 | The base model backend does **not** change | done | section 5, `offloaded N/N layers to GPU` is driven by `-ngl` alone |
| 4 | Available in the multimodal tools | done | `llama-mtmd-cli` and `llama-server` share the arg table |
| 5 | Accepted values: `auto`, `cpu`, a device name, case-insensitive | done | section 4 |
| 6 | Invalid value gives a clean error, no crash, no silent fallback | done | section 4 |
| 7 | One startup line naming both devices | done | `common_log_mmproj_devices` |
| 8 | Partial fallback is reported, not hidden | done | section 6 |
| 9 | Deliverable: llama.cpp changes | done | section 2 |
| 10 | Deliverable: instructions to run | done | section 3 |
| 11 | Deliverable: example output logs | done | `logs/task3/`, 9 files |

---

## 2. What changed

| File | Change |
|---|---|
| `common/common.h` | `std::string mmproj_backend = "auto";` on `common_params` |
| `common/arg.cpp` | the flag itself, case-insensitive device lookup, the valid-value list for the error message |
| `common/common.cpp` | `common_log_mmproj_devices`, the one startup line naming both devices |
| `tools/mtmd/mtmd-cli.cpp` | calls it before the vision context is built |
| `tools/server/server-context.cpp` | same, so the server reports it too |
| `tools/mtmd/clip.cpp` | reports how much of the projector actually landed on a requested device |

`auto` stays on the existing upstream code path, so default behaviour is unchanged. The flag reuses
the device plumbing that `clip_context_params` already carries rather than adding a parallel one.

The diff applies onto a clean checkout of the pinned commit with `git apply`.

---

## 3. How to run it

The flag needs a multimodal model: a language model GGUF plus a projector (`mmproj`) GGUF. The
script fetches a small ungated one, about 500 MB for the f16 pair:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_METAL=ON
cmake --build build -j --target llama-mtmd-cli

bash scripts/qvac/07_fetch_mmproj_model.sh   # ggml-org/SmolVLM-256M-Instruct-GGUF
```

`--list-devices` prints the names the flag accepts. On this machine they are `MTL0`, `BLAS`, `CPU`.

```bash
M=models/qvac/gguf/smolvlm/SmolVLM-256M-Instruct-f16.gguf
P=models/qvac/gguf/smolvlm/mmproj-SmolVLM-256M-Instruct-f16.gguf

# projector on the CPU, language model on the GPU
./build/bin/llama-mtmd-cli -m $M --mmproj $P -ngl 99 --mmproj-backend cpu \
    --image media/matmul.png -p "Describe this image." -n 64 --seed 42

# projector on the GPU, language model on the CPU
./build/bin/llama-mtmd-cli -m $M --mmproj $P -ngl 0 --mmproj-backend MTL0 \
    --image media/matmul.png -p "Describe this image." -n 64 --seed 42

# default, unchanged from upstream
./build/bin/llama-mtmd-cli -m $M --mmproj $P -ngl 99 \
    --image media/matmul.png -p "Describe this image." -n 64 --seed 42
```

`llama-server` takes the same flag. The environment variable `LLAMA_ARG_MMPROJ_BACKEND` sets it too.
Add `-v` to see the per-device placement lines quoted below.

Help text:

```
--mmproj-backend VALUE      backend used by the multimodal projector only, the language model is
                            not affected
                            accepts auto (default), cpu, or a device name from --list-devices,
                            case insensitive
                            (env: LLAMA_ARG_MMPROJ_BACKEND)
```

---

## 4. Value handling

| Input | Result |
|---|---|
| `auto` (default) | upstream code path, behaviour bit-identical to before the change |
| `cpu` | projector forced onto the CPU backend |
| `MTL0`, `BLAS`, `CPU` | that specific device |
| `mtl0` | accepted, matching is case-insensitive |
| `Cuda9` | clean error, exit code 1, no crash |

```
$ llama-mtmd-cli ... --mmproj-backend Cuda9
error while handling argument "--mmproj-backend": unknown device "Cuda9", valid values are: auto, cpu, MTL0, BLAS, CPU
```

The error lists what is valid on the machine it runs on, so the user does not have to guess.

---

## 5. The language model is not affected

This is the requirement that is easy to claim and easy to get wrong, so it is checked directly:
which device the *model* tensors went to, not just what the projector did.

| Run | Projector | Language model |
|---|---|---|
| `--mmproj-backend cpu -ngl 99` | `CLIP using CPU backend` | `offloaded 31/31 layers to GPU` |
| `--mmproj-backend MTL0 -ngl 0` | `CLIP using MTL0 backend` | `offloaded 0/31 layers to GPU` |
| default `-ngl 99` | `CLIP using MTL0 backend` | `offloaded 31/31 layers to GPU` |

Row 1 puts the projector on the CPU while the model is fully on the GPU, row 2 does the exact
opposite. `-ngl` alone decides where the model goes; the flag never touches `params.devices` or
`n_gpu_layers`.

The startup line states both, so a reader does not have to infer it:

```
common_log_mmproj_devices: projector device: CPU | language model device: auto (-ngl 99)
```

Timing confirms the projector really moved rather than just being labelled differently. Median image
encode over the same input:

| Configuration | Median encode | Output |
|---|---|---|
| `--mmproj-backend cpu` | 1059 ms | identical |
| `--mmproj-backend MTL0` | 93 ms | identical |
| default (`auto`) | 90 ms | identical |

11.4x apart, and all three produced a byte-identical description of the image (same MD5). Logs:
`logs/task3/proj_cpu_model_gpu.log`, `proj_gpu_model_cpu.log`, `default_auto.log`.

---

## 6. Partial fallback is reported, not hidden

A device can exist, initialise, and still run only part of the projector graph. Upstream printed
only a generic "the performance will be suboptimal" warning in that case, which reads as a tuning
note rather than as the requested placement not being honoured. When a device is named explicitly,
the outcome is now stated directly:

| Configuration | Line emitted | Graph splits |
|---|---|---|
| `--mmproj-backend MTL0` | `warmup: projector runs entirely on the requested device MTL0 (396 ops)` | 1 |
| `--mmproj-backend BLAS` | `warmup: projector falls back to the CPU for 238 of 420 ops, the requested device BLAS does not support them` | 172 |
| `--mmproj-backend cpu` | nothing new | - |
| default (`auto`) | nothing new | - |

`BLAS` is the useful adversarial case: it is an accelerator device, so it is accepted as a target,
but it implements a small fraction of the projector graph. The run still completes with exit code 0,
which is the required behaviour - an explicit report, not a crash and not silence.

Fixing this exposed a second problem in the same place. In the flash-attention auto path the second
`reserve_compute_meta()` result was discarded, so upstream's own op list, and the new line built on
it, described the flash-attention graph that had just been *rejected* rather than the one that runs.
Keeping the result changes the BLAS report from `226 of 396` to `238 of 420`. The old numbers were
not wrong about a graph, they were right about the wrong graph.

Logs: `logs/task3/guard_device_mtl0.log`, `guard_fallback_blas.log`, `guard_cpu.log`, `guard_auto.log`.

---

## 7. Example logs

| File | What it shows |
|---|---|
| `proj_cpu_model_gpu.log` | projector on CPU, model on GPU, encode times |
| `proj_gpu_model_cpu.log` | projector on GPU, model on CPU |
| `default_auto.log` | unchanged default behaviour |
| `guard_device_mtl0.log` | `-v` run, whole graph on the requested device, `graph splits = 1` |
| `guard_fallback_blas.log` | `-v` run, partial fallback reported, still exits 0 |
| `guard_cpu.log`, `guard_auto.log` | the two paths that deliberately emit nothing new |
| `07_fetch_mmproj_model.log`, `mmproj_provenance.txt` | which multimodal model, which revision |

Read the key lines without re-running anything:

```bash
grep "projector device" logs/task3/proj_*.log logs/task3/default_auto.log
grep -E "runs entirely|falls back" logs/task3/guard_*.log
grep "offloaded" logs/task3/guard_device_mtl0.log
```

---

## 8. Honest notes

- **Most of the mechanism already existed upstream.** The pinned commit has `-mmdev` and
  `--mmproj-offload`, and `clip_context_params` already carries a device. The new flag is the
  vocabulary the brief asks for layered onto that mechanism. What was genuinely missing is the
  guard in section 6.
- **`ctest` on this branch is 60/60**, the same as on the other branch, so nothing was broken in
  passing. Run `scripts/qvac/06_test_env.sh` once first; two tests need fixtures the build does not
  produce.
- The three-configuration comparison uses a 256M-parameter model. It is enough to show the projector
  moving between devices and the model staying put, which is what the flag is about, but the
  absolute encode times are not representative of a large vision model.
