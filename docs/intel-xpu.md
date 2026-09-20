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

Install the XPU wheels before installing FreeToken. Do not install the NVIDIA `triton`
wheel into the same environment as `triton-xpu`; both own the `triton` Python package.
`flashlib` is presently installed without dependencies to avoid replacing Intel Triton.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/xpu \
  torch==2.12.1+xpu triton-xpu==3.7.1
uv pip install --python .venv/bin/python --no-deps flashlib==0.3.0
sudo apt-get install libze-dev
FREETOKEN_ACCELERATOR=xpu uv pip install --python .venv/bin/python -e . --no-deps
```

Install the remaining accelerator-neutral dependencies from `pyproject.toml`, excluding
the CUDA `torch`, `torchvision`, and `triton` entries. A future packaging milestone will
publish separate resolved CUDA and XPU dependency sets; until then, `--no-deps` is
required for the editable XPU install.

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
same 61-token public `LLM` benchmark improved from 0.55 prompt tokens/s on the untiled
path to 1.81 tokens/s with all active quant formats tiled (3.3x). Both runs returned
token `The` (ID 760). The latest tiled run took 39.16 s to load and 33.62 s for
generation; generation timing includes prefill. This remains experimental, and
identical-token parity against a separate reference implementation has not yet been
established.

The same public XPU path also completed a forced-length smoke for the local
`Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` checkpoint on this B580. With the standard
61-token prompt, context 128, and a 256-token KV cache, it loaded in 48.85 s, prefilled
at 1.53 tokens/s, and emitted the token IDs `760, 1156, 6587` (`The user wants`). The
two inter-token intervals measured 0.794 decode tokens/s. This is one constrained
functional sample, not a Q2_K comparison or a steady-state performance claim. During
the live run, `nvtop` identified Battlemage G21 / Arc B580, showed 99% memory use and a
1.63 GHz GPU clock; its utilization field was unavailable.

For a non-system Level Zero SDK, expose its header and loader paths through `CPATH` and
`LIBRARY_PATH` before running Intel Triton for the first time. The runtime reports the
selected device, PCI ID, driver, and Level Zero platform in the benchmark JSON.

Current constraints are explicit: one XPU only, eager execution only, and the portable
`torch` attention backend for FULL/SWA models. CUDA graphs, CUDA device identifiers, and
tensor parallel XPU launches are rejected rather than silently falling back.

## Device capability report

Run `ft devices` to list CUDA and XPU devices, memory, driver/platform identifiers, and
runtime features such as streams, events, and graph capture. Use `ft devices --json` for
machine-readable output. The JSON object contains `devices` and `backends`; each backend
reports whether its runtime is available, unavailable, failed to probe, or only partially
enumerated. This makes missing drivers and device-property errors visible even when one
backend still works. If no accelerator is found, the command reports that directly; it does
not select a CPU inference fallback.

## Native SYCL kernels

The XPU build includes a native SYCL extension for the causal depthwise convolution
used by gated-delta-network decode. It submits to PyTorch's current XPU queue, so tensor
ownership and stream ordering remain inside the existing engine. FP32 and BF16 output
and in-place state updates are validated on the Arc B580.

The same extension includes direct packed GGUF matvec kernels for Q4_0, Q8_0, Q2_K,
Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_S, IQ2_XXS, IQ2_XS, IQ3_S, IQ3_XXS, and IQ4_XS.
Synthetic FP32/BF16 device tests cover each format, real slices from the Qwen3.8
checkpoint match the FP32 dequantized reference, and the complete packed Qwen serving
model has run through the public `LLM` API on the B580. Q4_0 currently uses the native
SYCL kernel for single-token decode; larger XPU batches retain the existing chunked
dequantize-plus-matmul path. This avoids CPU fallback while keeping the slower measured
prompt path off the direct kernel. Other active-serving packed formats retain their
small-batch and four-token tiled paths. In the recorded Qwen3.8 checkpoint, Q6_K is
present only in the MTP layer that serving currently omits; synthetic tests cover the
new Q6_K path, but no real checkpoint inference has yet exercised it.

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
On the host Arc B580 (device ID `0xE20B`, driver `1.6.33578+15`, Level Zero V2), all four
direct PQ2_0 dtype/batch cases passed, along with the 59-test SYCL extension file and
the targeted GGUF reader/dequant/model tests (21 passed). A short end-to-end run of the
classic checkpoint also loaded and generated `The user wants` using the XPU runtime.

The native extension was rebuilt with oneAPI DPC++ 2025.3.3; its `libsycl.so.8` matches
the `libsycl.so.8` dependency of this PyTorch XPU build. Reproduce the model smoke test
with:

```bash
FREETOKEN_ACCELERATOR=xpu PYTHONPATH=python \
  .venv/bin/python benchmarks/bench_xpu_gemma4.py \
  /home/oddsoul/models/Ternary-Bonsai-27B-PQ2_0.gguf \
  --max-tokens 4 --context 128 --kv-tokens 256
```

The historical smoke run loaded the model in 38.59 s and generated 3 tokens in 21.43 s.
It used the earlier whole-generation timer, which included prompt prefill; the updated
benchmark reports TTFT and inter-token decode rate separately. This short run remains
functional evidence, not a throughput baseline. For a longer measurement, use
`--max-tokens 64 --force-decode-length`; the first request is cold and should not be
treated as a steady-state result.

The revised benchmark also completed a 16-token forced-length run of this checkpoint on
the host B580 (PCI `0xE20B`, driver `1.6.33578+15`, Level-Zero V2). With 19 prompt tokens,
load took 39.54 s, TTFT was 21.57 s, and 14 decode tokens arrived over 5.77 s (2.43
tokens/s); end-to-end output rate was 0.55 tokens/s over 15 returned tokens. The terminal
EOS is omitted from the returned list. This single cold sample confirms longer XPU
execution and separate timing, but it is not a steady-state benchmark.

Bonsai 2 additionally declares a block-1024 normalized Walsh-Hadamard transform,
explicit signs, and grouped GDN values. Registering type 142 alone does not establish
Bonsai 2 correctness; it remains unsupported until its model-transform handling has
reference parity. A B580 smoke of `Ternary-Bonsai-2-27B-PQ2_0.gguf` on 2026-09-20
reached this rejection before model allocation or GPU work: `Prism Hadamard-transformed
GGUF weights are not supported`. The loader rejects this transform metadata rather than
silently interpreting it as the classic Bonsai format.
