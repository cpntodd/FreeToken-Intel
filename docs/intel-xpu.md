# Intel XPU development

FreeToken's native Intel path uses PyTorch XPU over oneAPI/Level Zero. It keeps the
existing scheduler, model, KV-cache, and sampling architecture; it does not launch a
separate inference server. Explicit `--accelerator xpu` never falls back to CPU or CUDA.

The currently validated Arc B580 stack is:

- `torch==2.12.1+xpu`
- `triton-xpu==3.7.1`
- the oneAPI 2025.3 runtime bundled by that PyTorch wheel
- Intel compute-runtime driver `1.6.33578+15`
- Level Zero development headers (`libze-dev` on Debian/Ubuntu)

PyTorch `2.14.0+xpu` was also tested on this host. It enumerated the B580 but aborted
inside Intel's command encoder on its first submitted kernel, so it is not a supported
combination yet.

Install the XPU wheels before building FreeToken. Do not install the NVIDIA `triton`
wheel into the same environment as `triton-xpu`; both own the `triton` Python package.
The `xpu` extra deliberately excludes CUDA-only `flashlib`, CUDA Torch, and NVIDIA
Triton. The build is run without isolation so the native extension links against the
already-installed XPU Torch rather than a build-isolation CUDA Torch.

```bash
uv venv --python 3.12 .venv
# Install the matching oneAPI 2025.3 DPC++ compiler from Intel's package repository first.
export ONEAPI_ROOT=/opt/intel/oneapi
export FREETOKEN_SYCL_COMPILER_VERSION=2025.3
export CXX="$ONEAPI_ROOT/compiler/2025.3/bin/icpx"
sudo apt-get install libze-dev
FREETOKEN_ACCELERATOR=xpu uv pip install --python .venv/bin/python \
  --torch-backend xpu --no-sources --no-build-isolation -e ".[xpu]"
```

`--torch-backend xpu` resolves `torch==2.12.1+xpu` and its matching `triton-xpu` from
PyTorch's XPU index. `--no-sources` prevents the repository's CUDA-specific uv source
pin from overriding that selection. CUDA users install `freetoken[cuda]` (or the
existing full-path alias `freetoken[accel]`); these profiles must not be combined in one
environment.

Run the hardware benchmark with a supported local GGUF:

```bash
FREETOKEN_ACCELERATOR=xpu \
PYTHONPATH=python \
.venv/bin/python benchmarks/bench_xpu_gemma4.py \
  /home/oddsoul/models/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf
```

The benchmark JSON reports time-to-first-token, decode tokens/s measured from
inter-token timestamps, and end-to-end output tokens/s separately. TTFT starts after
prompt tokenization and includes the first request's scheduler and runtime work; decode
rate excludes the first token and is null when fewer than two tokens are emitted. Lazy
kernel setup on the first request is included, so treat a single cold run as diagnostic
rather than a steady-state performance result.

On 2026-09-20, the updated harness ran
`gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` on the B580 and reported
`execution_device: xpu:0`, device ID `0xE20B`, and Level-Zero V2. The 20-token prompt
reached first token in 19.88 s; 14 measured decode intervals took 4.25 s (3.29 tokens/s),
and the 15 returned tokens measured 0.62 end-to-end tokens/s. Model load took 46.14 s.
The forced-length sample repeated control tokens, so these numbers validate the live
execution path and timer, not response quality or steady-state speed. During the run,
`nvtop` identified Battlemage G21 / Arc B580 and showed 34% memory use at 2.15 GHz; its
GPU-utilisation field was unavailable.

After the GGUF USER_DEFINED-token fix, a normal four-token regression smoke also loaded
this Gemma checkpoint on the B580. The 20-token prompt returned IDs `9259, 236888`
(`Hello!`); load took 47.88 s and TTFT was 20.03 s. This confirms the shared tokenizer
change did not break this Gemma path, but it is not a throughput or model-quality result.

Use `--max-tokens 64 --force-decode-length` for a longer decode sample. That option
ignores EOS until the token limit and is intended for measurement, not normal generation.
The earlier one-token B580 smoke timing used the old whole-generation timer and is not
comparable to the separate decode metric.

The same benchmark also covers Llama 3 GGUF checkpoints. The validated
`Llama-3.2-1B-Instruct-Q4_K_M.gguf` path reconstructs the checkpoint's Llama 3 RoPE
scaling, reverses the GGUF q/k RoPE permutation, and dequantizes mixed Q4_K/Q6_K
weights to BF16 before fusing projections. On the Arc B580, the public `LLM` API
generated `Hello!` for the default chat prompt. This is a correctness baseline, not
the final packed-quantized execution path; the dense BF16 expansion increases device
memory use and load time.

On 2026-09-20, the same host reran `Llama-3.2-1B-Instruct-Q4_K_M.gguf` through the
public `LLM` API on the B580 (PCI `0xE20B`, driver `1.6.33578+15`, Level-Zero V2).
The 35-token prompt ran at 10.29 tokens/s and produced `Hello!` (token IDs 9906, 0);
load took 21.48 s and generation took 3.50 s including prefill. This is a short
functional run, not a packed-quantized performance result.

After registering GGUF USER_DEFINED tokens as special tokens, a repeat 35-token
smoke on the same B580 produced `Hello!` (token IDs 9906, 0); load took 22.68 s,
TTFT was 3.67 s, and the one decode interval took 0.055 s (18.05 decode tokens/s).
This confirms the shared tokenizer fix did not regress this path; it is not a
steady-state performance comparison.

The revised timer was exercised on the same B580 with:

```bash
FREETOKEN_ACCELERATOR=xpu PYTHONPATH=python \
  .venv/bin/python benchmarks/bench_xpu_gemma4.py \
  /home/oddsoul/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --max-tokens 32 --force-decode-length --context 128 --kv-tokens 256
```

The cold run loaded in 23.15 s and reported 5.04 s to first token, 30 decode tokens over
1.52 s (19.77 decode tokens/s), and 4.73 end-to-end output tokens/s across 31 returned
tokens; the terminal EOS is omitted from the returned list. EOS was ignored to hold the
decode sample length, producing repeated chat/control tokens, so this validates timing
instrumentation only, not output quality or steady-state performance.

Dense Qwen3.8 GGUF checkpoints use a memory-feasible native packed path rather than
expanding the 27B model to BF16. The adapter preserves each source tensor's GGML type,
supports the checkpoint's mixed K/IQ/Q8 formats, reverses llama.cpp's tiled GDN value-head
storage at the FreeToken recurrence boundary, and omits the stored MTP layer. The
`Qwen3.8-27B-UD-Q2_K_XL.gguf` checkpoint loads 9.47 GB of model state on the B580.
The active serving layers dispatch every quant format used by those layers to a direct
packed SYCL kernel; Q6_K remains confined to its omitted MTP layer. All 13 quant
formats used by active serving layers now select a four-token tiled workgroup for
batches of four or more, reusing each decoded weight across prompt tokens without a
dense-weight temporary; batches below four retain the direct path. On the Arc B580, the
historical 61-token public `LLM` benchmark improved from 0.55 prompt tokens/s on the untiled
path to 1.81 tokens/s with all active quant formats tiled (3.3x). Both runs returned
token `The` (ID 760). The latest tiled run took 39.16 s to load and 33.62 s for
generation; generation timing includes prefill. This remains experimental. The
identical-token parity warning referred to the pre-fix, 61-token prompt. After
registering GGUF USER_DEFINED tokens as special, the exact 59-token prompt matches
Prism `/tokenize`. FreeToken emitted IDs `760, 1156, 6587` (`The user wants`) on the
B580; load took 47.89 s, TTFT was 36.52 s, and two decode intervals measured 0.997
tokens/s. Prism Vulkan generated the same short text from the exact prompt. This is
prompt-token parity and short text-level agreement, not logits parity or a general
model-quality result.

A reusable first-token score probe is in
`experiments/compare_xpu_prism_logits.py`. Run `capture-xpu` with Prism stopped,
then run the same GGUF on Prism `llama-server` with `--device Vulkan0`, capture its
`/completion` top probabilities with `capture-prism`, and use `compare`. The two
engines are run sequentially to fit the B580's 12 GB memory. The probe sends the
exact FreeToken token IDs to Prism and compares pre-sampler FreeToken log-softmax
scores with Prism `n_probs` output; it does not modify serving behavior.

For a repeatable comparison, first capture FreeToken with Prism stopped:

```bash
FREETOKEN_ACCELERATOR=xpu PYTHONPATH=python:. \
  .venv/bin/python experiments/compare_xpu_prism_logits.py capture-xpu \
  /path/to/model.gguf --output /tmp/xpu-scores.json
```

Then start Prism `llama-server` on that same model with `--device Vulkan0` and
capture/compare its response:

```bash
PYTHONPATH=python:. .venv/bin/python experiments/compare_xpu_prism_logits.py \
  capture-prism --xpu-json /tmp/xpu-scores.json \
  --server-url http://127.0.0.1:8080 --reference-backend Vulkan0 \
  --output /tmp/prism-scores.json
PYTHONPATH=python:. .venv/bin/python experiments/compare_xpu_prism_logits.py \
  compare --xpu-json /tmp/xpu-scores.json --prism-json /tmp/prism-scores.json
```

On 2026-09-21, this probe ran the first token for the same Qwen Q2_K checkpoint and
59-token prompt on FreeToken XPU and Prism Vulkan0 on the B580. Both selected token
ID 760 (`The`), and the top-five IDs matched. The top-20 sets overlapped 18/20; among
shared candidates, mean absolute log-probability difference was 0.18 and the maximum
was 0.63. This establishes top-token agreement, not full-vocabulary or logits parity;
the score differences remain a numerical validation gap. `nvtop` showed B580 memory
use during each engine run; its utilization field was unavailable.

The same public XPU path completed a forced-length smoke for the local
`Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` checkpoint on this B580. Its earlier 61-token
sample (48.85 s load, 1.53 prompt tokens/s, and 0.794 decode tokens/s) used the
pre-fix prompt and is not comparable to the corrected run. After the tokenizer fix,
the exact 59-token prompt matched Prism `/tokenize`; with context 128 and a 256-token
KV cache, FreeToken emitted IDs `1596, 1144, 4087` (`We need answer`). Load took
47.67 s, TTFT was 41.71 s, and two decode intervals measured 0.829 tokens/s. Prism
Vulkan generated the same short text from the exact prompt. This is one constrained
functional sample and a text-level comparison, not logits parity, MTP validation, or
a steady-state performance claim.

For a non-system Level Zero SDK, expose its header and loader paths through `CPATH` and
`LIBRARY_PATH` before running Intel Triton for the first time. The runtime reports the
selected device, PCI ID, driver, and Level Zero platform in the benchmark JSON.

Current constraints are explicit: one XPU only, eager execution only, and the portable
`torch` attention backend for FULL/SWA models. Routed MoE checkpoints are rejected on XPU
before loading because their current host-bank, slot-cache, and expert kernels use CUDA-only
registration and stream APIs; use a dense checkpoint on XPU or CUDA for routed MoE. CUDA
graphs, CUDA device identifiers, and tensor parallel XPU launches are rejected rather than
silently falling back.

KV and recurrent-state cache rebuilds release their old slabs through the selected CUDA
or XPU runtime before allocating replacements. On the B580, an XPU MHA cache resize kept
both old and new slabs on `xpu:0`; CPU devices and unavailable accelerator APIs are
intentional no-ops rather than a CUDA fallback.

## Device capability report

Run `ft devices` to list CUDA and XPU devices, memory, driver/platform identifiers, and
runtime features such as streams, events, and graph capture. Use `ft devices --json` for
machine-readable output. The JSON object contains `devices` and `backends`; each backend
reports whether its runtime is available, unavailable, failed to probe, or only partially
enumerated. This makes missing drivers and device-property errors visible even when one
backend still works. If no accelerator is found, the command reports that directly; it does
not select a CPU inference fallback.

For an XPU device, each device record also includes an `engine` capability object. It reports
the FreeToken constraints separately from Level Zero features: single-GPU eager dense inference,
the portable `torch` attention backend, and the current lack of routed-MoE and CUDA-graph
support. The human-readable report prints the same engine limits below the device runtime data.

## Native SYCL kernels

The XPU build includes a native SYCL extension for the causal depthwise convolution
used by gated-delta-network decode. It submits to PyTorch's current XPU queue, so tensor
ownership and stream ordering remain inside the existing engine. FP32 and BF16 output
and in-place state updates are validated on the Arc B580.

The same extension includes direct packed GGUF matvec kernels for Q4_0, Q4_1, Q5_0,
Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_S, IQ2_XXS, IQ2_XS,
IQ3_S, IQ3_XXS, and IQ4_XS. Synthetic FP32/BF16 device tests cover each format, real slices from the Qwen3.8
checkpoint match the FP32 dequantized reference, and the complete packed Qwen serving
model has run through the public `LLM` API on the B580. Q4_0, Q4_1, Q5_0, and Q5_1
currently use the native SYCL kernels for single-token decode; larger XPU batches retain
the existing chunked dequantize-plus-matmul path. The supplied model set has no Q4_1-,
Q5_0-, or Q5_1-designated checkpoint, so their coverage is synthetic rather than
full-model. Warmed 15-iteration B580 microbenchmarks at `[1, 4096] x [4096, 4096]` (BF16)
measured median direct-SYCL vs XPU dequantize-plus-matmul times of 0.677 vs 2.841 ms for
Q4_1, 0.428 vs 4.128 ms for Q5_0, and 0.520 vs 4.584 ms for Q5_1; output parity passed
for all three. These are synthetic samples, not end-to-end model performance. The
decode-only dispatch avoids CPU fallback while keeping prompt batches on XPU matmul.
Other active-serving packed formats retain their small-batch and four-token tiled paths.
In the recorded Qwen3.8 checkpoint, Q6_K is
present only in the MTP layer that serving currently omits. A bounded probe now reads
16 packed rows from the real `blk.64.attn_output.weight` tensor (GGML shape
`[6144, 5120]`) and checks three input tokens against the canonical dequantized CPU
reference on B580 `xpu:0`. FP32 max absolute/relative errors were `2.50e-6` / `1.48e-5`;
BF16 outputs matched exactly. This exercises actual Q6_K weight bytes but does not run
the omitted MTP layer or a full Q6_K model. The probe copies only the requested rows:

```bash
FREETOKEN_ACCELERATOR=xpu PYTHONPATH=python .venv/bin/python \
  experiments/probe_q6_k_real_tensor.py \
  /home/oddsoul/models/Qwen3.8-27B-UD-Q2_K_XL.gguf \
  --dtype fp32 --rows 16 --tokens 3
```

The host B580's `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` contains 329 Q4_0 tensors and 338
F32 tensors, with no Q6_K tensors. The earlier note claiming this checkpoint exercised
Q6_K was incorrect: those runs used the prior Q4_0 XPU fallback. On this model and the
20-token prompt below, a forced eight-token first direct run reported 349.41 s TTFT and
2.93 decode tokens/s. A repeated direct run with the same token IDs reported 16.60 s
TTFT and 3.40 decode tokens/s. The previous chunked XPU fallback, restored by a runtime
override, reported 8.68 s TTFT and 0.50 decode tokens/s. A run using the hybrid dispatch
(XPU matmul for prompt prefill, native SYCL for single-token decode) returned the same
token IDs and reported 20.55 s TTFT and 2.93 decode tokens/s. These are single samples
with substantial timing variance: decode improved in the direct/hybrid runs, but no
TTFT or end-to-end throughput win is established. The cause of the very high first-run
TTFT has not been isolated. Forced-length output included repeated control tokens, so
this is a kernel/serving-path check, not a quality sample.

```bash
FREETOKEN_ACCELERATOR=xpu PYTHONPATH=python .venv/bin/python \
  benchmarks/bench_xpu_gemma4.py \
  /home/oddsoul/models/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
  --max-tokens 8 --force-decode-length --context 128 --kv-tokens 256
```

Build the extension with an `icpx` compiler from the same oneAPI release as the SYCL
runtime bundled by PyTorch. For the validated `torch==2.12.1+xpu` wheel, that is oneAPI
2025.3. Mixing a 2026 compiler with the wheel's `libsycl.so.8` produces an incompatible
extension even if compilation succeeds. The build checks that `CXX` belongs to the
selected oneAPI compiler directory and that its `libsycl` SONAME matches the dependency
declared by `torch_xpu`; it stops before compilation if the ABI cannot be verified.
This check requires `readelf` from binutils.

```bash
export ONEAPI_ROOT=/opt/intel/oneapi
export FREETOKEN_SYCL_COMPILER_VERSION=2025.3
export CXX="$ONEAPI_ROOT/compiler/2025.3/bin/icpx"
FREETOKEN_ACCELERATOR=xpu \
  .venv/bin/python setup.py build_ext --inplace
PYTHONPATH=python \
  .venv/bin/python -m pytest -q tests/accelerator/test_sycl_causal_conv1d.py
```

The build resolves headers from
`$ONEAPI_ROOT/compiler/$FREETOKEN_SYCL_COMPILER_VERSION/include`. The extension fails
explicitly when they are absent; it does not substitute a CPU implementation.

## OpenVINO compute islands

Install `openvino==2026.4.0` to enable the optional bounded dense compute-island
provider. `OpenVINODenseIsland` compiles only for an explicit OpenVINO `GPU` device,
checks the compiled model's `EXECUTION_DEVICES`, and rejects CPU participation. Its
current bridge stages tensors through host FP16 memory; each invocation reports input,
inference, and output-copy timings so this cost cannot be mistaken for zero-copy USM.
By default, OpenVINO import, GPU compilation, and inference failures raise to the caller.
Callers may explicitly pass `fallback="xpu"` to run the same bounded projection and
activation with PyTorch on XPU when one of those OpenVINO steps fails. The fallback
requires XPU input and an available XPU device; every result identifies its actual
`backend` and includes a `fallback_reason` when XPU was used.

Run its B580 benchmark with:

```bash
PYTHONPATH=python .venv/bin/python benchmarks/bench_openvino_island.py \
  --source xpu --tokens 32 --hidden-size 1024 --output-size 4096
```

To measure the opt-in fallback path after an OpenVINO failure, add `--fallback xpu`.
The benchmark requires `--source xpu` for that policy and reports the actual backend,
fallback reason, and backend counts rather than labeling XPU fallback timings as
OpenVINO timings.

The benchmark compares the complete OpenVINO path, including host staging, with
source-device eager execution and checks output parity. On the B580, a 32-token
`5120 -> 10240` projection measured 2.21 ms through the OpenVINO island versus
0.33 ms for eager XPU (6.6x slower); maximum absolute output error was 0.0078125.
This is evidence for the current staging cost, not a general OpenVINO-versus-XPU
performance claim. A zero-copy or larger fused island requires a separate benchmark.

OpenVINO's C++ GPU Remote Tensor API supports USM pointers, but direct ownership-safe
PyTorch XPU interoperability has not yet been proven in this Python integration. Host
staging remains intentional until that proof exists.

The OpenVINO 2026.4 Python `RemoteContext.create_tensor` binding was tested with a
live PyTorch XPU `data_ptr()` and `SHARED_MEM_TYPE=USM_USER_BUFFER`. Python integers,
`ctypes.c_void_p`, and NumPy pointer scalars were all rejected before tensor creation;
the binding does not expose a safe raw-pointer conversion. In addition, OpenVINO's
default GPU context reports `CONTEXT_TYPE=OCL`, while PyTorch XPU normally submits via
oneAPI/Level Zero. A zero-copy bridge therefore belongs in a C++ extension that can
validate native context compatibility, retain the PyTorch allocation, and synchronize
both queues. Passing an integer device address through Python is not a supported path.

`experiments/probe_openvino_usm.py` is an isolated C++/Python probe for that boundary.
It compiles a GPU-only OpenVINO multiply model, tries to wrap a live PyTorch XPU tensor
with the GPU plugin's `USM_USER_BUFFER` RemoteTensor, checks pointer identity, and
compares the result. It synchronizes PyTorch around inference, so it tests serialized
sharing only, not asynchronous queue interoperability. On the host B580 with PyTorch
2.12.1+xpu and OpenVINO 2026.4.0, PyTorch reports Level-Zero V2 and OpenVINO identifies
the same Arc B580 but exposes an OCL context. RemoteTensor creation rejects the
PyTorch allocation with `shared USM buffer has smaller size (0) than specified layout
(64)` before inference; pointer identity and output parity are therefore unproven. This
confirms that the documented OpenVINO USM-pointer API does not by itself make this
Level-Zero allocation importable into the current OCL context. Keep the host-staged
path until a compatible context or explicit export/import mechanism is proven.

Run the probe with:

```bash
FREETOKEN_ACCELERATOR=xpu PYTHONPATH=python \
  .venv/bin/python experiments/probe_openvino_usm.py
```

Exit code 2 with a JSON diagnostic means the current runtime pair rejected the pointer;
it is not a CPU fallback or a failed OpenVINO GPU compilation.

## Vulkan prototype

`experiments/vulkan` contains an isolated dense compute benchmark; it does not alter
serving dispatch. It selects only an Intel discrete Vulkan compute device (`vendorID`
`0x8086`) and fails rather than using llvmpipe. Three paths can be compared: the
original row-major FP32 shader, a transposed-weight FP32 shader, and an optional
`VK_KHR_cooperative_matrix` shader. The cooperative path is enabled only when the
runtime reports a compatible FP16-input/FP32-accumulator subgroup tuple, the required
Vulkan 1.1/1.2 features are enabled, and `glslc` successfully builds the shader.
Otherwise `best` uses the measured FP32 baseline.

Buffers prefer memory that is both `DEVICE_LOCAL` and `HOST_VISIBLE`; the current
B580 selects memory type 3, backed by its device-local heap and also host coherent. On a
device without coherent host-visible memory, the probe flushes uploads and invalidates
the mapped output. This is Vulkan-owned mapped memory, not a proven shared allocation
with PyTorch XPU. The cooperative probe pads row, input, and output dimensions to the
queried matrix tile and validates against a CPU FP32 reference using the same FP16
rounded inputs and weights.

```bash
bash experiments/vulkan/run_probe.sh both 8 256 512 20
bash experiments/vulkan/run_probe.sh best 1 1024 4096 10
```

On the host Arc B580 (device `0xE20B`, Mesa 25.0.7), the runtime reports a subgroup
cooperative-matrix tile of `8x16x16` for FP16 inputs with FP32 accumulation. For
`8x256` by `256x512` over 20 warmed iterations, median blocking dispatch was 0.335 ms
for naive FP32, 0.417 ms for transposed FP32, and 0.246 ms for cooperative FP16. For
`1x1024` by `1024x4096` over 10 iterations, medians were 0.761 ms naive, 0.867 ms
transposed, and 0.254 ms cooperative; maximum error against the FP16-rounded reference
was `5.96e-8`. These small synthetic probes show the cooperative path is promising on
this B580, not a general Vulkan-vs-XPU result. The host-visible mapped buffers avoid a
Vulkan staging copy on this memory type. `median_upload_plus_dispatch_ms` includes the
per-call input write and blocking queue completion, but excludes static weight packing
and upload (reported separately as `weight_upload_ms`) and any Vulkan-to-PyTorch output
handoff.

Run this odd-dimension validation-layer smoke:

```bash
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  bash experiments/vulkan/run_probe.sh best 5 255 513 4
```

Serving integration remains gated on architecture review; it still needs
reusable pipelines, long-lived model-weight ownership, PyTorch/XPU interoperability,
batching, and measurements that include all transfer and synchronization costs.

## Bonsai PQ2_0 compatibility

The local `Ternary-Bonsai-27B-PQ2_0.gguf` and
`Ternary-Bonsai-2-27B-PQ2_0.gguf` files both declare `qwen35`, and both use private
type 142 rather than upstream Q2_0. PQ2_0 uses group size 128 (34 bytes per block),
while upstream Q2_0 is group size 64. The
[Prism block definition](https://github.com/PrismML-Eng/llama.cpp/blob/prism/ggml/src/ggml-common.h#L2460-L2475)
and [upstream tracking issue](https://github.com/ggml-org/llama.cpp/issues/29058)
describe the distinction.

The classic `Ternary-Bonsai-27B-PQ2_0.gguf` has no `prism.hadamard.*` metadata. Its
private type 142 is handled by a per-reader GGUF adapter, a 128-value/34-byte block
dequantizer, and a native XPU SYCL matvec; the upstream `gguf-py` enum is not modified.
GGUF `USER_DEFINED` tokenizer entries are registered as additional special tokens. This
keeps Qwen's `<think>` at its dedicated GGUF ID (`248068`) instead of splitting it into
three ordinary tokens. FreeToken and Prism `/tokenize` now produce identical IDs for the
classic prompt (17 tokens) and Bonsai 2 prompt (59 tokens). On the host Arc B580 (device
ID `0xE20B`, driver `1.6.33578+15`, Level Zero V2), all four direct PQ2_0 dtype/batch
cases passed, along with the 59-test SYCL extension file and the targeted GGUF
reader/dequant/model tests (21 passed). A corrected short classic-checkpoint run on
`xpu:0` generated token IDs `8160, 579, 264` (`Here's a`); a Prism Vulkan run with the
same 17-token prompt produced the same text. This is a functional smoke, not a quality
or throughput claim.

The native extension was rebuilt with oneAPI DPC++ 2025.3.3; its `libsycl.so.8` matches
the `libsycl.so.8` dependency of this PyTorch XPU build. Reproduce the model smoke test
with:

```bash
FREETOKEN_ACCELERATOR=xpu PYTHONPATH=python \
  .venv/bin/python benchmarks/bench_xpu_gemma4.py \
  /home/oddsoul/models/Ternary-Bonsai-27B-PQ2_0.gguf \
  --max-tokens 4 --context 128 --kv-tokens 256
```

The older 38.59 s / 21.43 s measurement predates the tokenizer fix and used the split
`<think>` prompt. It included prompt prefill and is retained only as historical timing;
the updated benchmark reports TTFT and inter-token decode rate separately. For a longer
measurement of the corrected prompt, use `--max-tokens 64 --force-decode-length`; the
first request is cold and should not be treated as a steady-state result.

The earlier 16-token forced-length benchmark used 19 prompt tokens because `<think>` was
split into ordinary sub-tokens. Its 39.54 s load / 2.43 tokens/s sample is retained as
historical data only; it is not a performance baseline for the corrected 17-token prompt.

Bonsai 2 declares a block-1024 normalized Walsh-Hadamard transform, explicit signs,
inverse token-embedding handling, and grouped GDN values. The Qwen3.5 GGUF loader now
validates that metadata, verifies all 401 declared projection tensors are present and
attached to supported paths, applies signed transforms before selected projections,
restores embedding rows after lookup, and converts grouped GDN values before the
`ssm_out` transform. Unsupported versions, axes, sign modes, tensor names, and tied
inverse embeddings still fail closed.

The native SYCL operator has FP32/BF16 parity tests for forward and inverse transforms at
all three checkpoint widths (5120, 6144, and 17408). Model tests cover projection
selection, inverse embedding ordering, and the GDN permutation. The full
`Ternary-Bonsai-2-27B-PQ2_0.gguf` loaded on the host Arc B580 (`xpu:0`, PCI `0xE20B`,
driver `1.6.33578+15`, Level Zero V2) with the corrected 59-token prompt and generated
token IDs `1596, 1144, 310` (`We need to`). Load took 40.82 s, TTFT was 30.33 s, and two
decode intervals took 0.96 s (2.08 tokens/s). Prism Vulkan on the same B580 and Prism CPU
both produced the same short text for the identical prompt; the classic checkpoint also
matches Prism Vulkan text on its identical 17-token prompt. These checks establish prompt
token-ID alignment and short continuation agreement, not full logit parity, model quality,
or steady-state performance. Earlier outputs from prompts with the split `<think>` token
are superseded. Reproduce the transformed-checkpoint smoke with:

```bash
FREETOKEN_ACCELERATOR=xpu PYTHONPATH=python \
  .venv/bin/python benchmarks/bench_xpu_gemma4.py \
  /home/oddsoul/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --max-tokens 4 --context 128 --kv-tokens 256
```
