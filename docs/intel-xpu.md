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

The same benchmark also covers Llama 3 GGUF checkpoints. The validated
`Llama-3.2-1B-Instruct-Q4_K_M.gguf` path reconstructs the checkpoint's Llama 3 RoPE
scaling, reverses the GGUF q/k RoPE permutation, and dequantizes mixed Q4_K/Q6_K
weights to BF16 before fusing projections. On the Arc B580, the public `LLM` API
generated `Hello!` for the default chat prompt. This is a correctness baseline, not
the final packed-quantized execution path; the dense BF16 expansion increases device
memory use and load time.

Dense Qwen3.8 GGUF checkpoints use a memory-feasible native packed path rather than
expanding the 27B model to BF16. The adapter preserves each source tensor's GGML type,
supports the checkpoint's mixed K/IQ/Q8 formats, reverses llama.cpp's tiled GDN value-head
storage at the FreeToken recurrence boundary, and omits the stored MTP layer. The
`Qwen3.8-27B-UD-Q2_K_XL.gguf` checkpoint loads 9.47 GB of model state on the B580.
The active serving layers dispatch every quant format used by those layers to a direct
packed SYCL kernel; the Q6_K tensors are confined to its omitted MTP layer. IQ1_S,
IQ2_S, IQ2_XXS, IQ2_XS, IQ3_S, and IQ3_XXS use a four-token tiled workgroup for batches
of four or more, reusing each decoded weight across neighboring prompt tokens without
a dense-weight temporary. On the Arc B580, the same 61-token public `LLM` benchmark
improved from 0.55 prompt tokens/s on the untiled path to 1.83 tokens/s with these six
formats tiled (3.3x). Both runs returned token `The` (ID 760). The latest tiled run
took 39.73 s to load and 33.37 s for generation; generation timing includes prefill.
This remains experimental: the other formats still use direct matvec kernels, and
identical-token parity against a separate reference implementation has not yet been
established.

For a non-system Level Zero SDK, expose its header and loader paths through `CPATH` and
`LIBRARY_PATH` before running Intel Triton for the first time. The runtime reports the
selected device, PCI ID, driver, and Level Zero platform in the benchmark JSON.

Current constraints are explicit: one XPU only, eager execution only, and the portable
`torch` attention backend for FULL/SWA models. CUDA graphs, CUDA device identifiers, and
tensor parallel XPU launches are rejected rather than silently falling back.

## Native SYCL kernels

The XPU build includes a native SYCL extension for the causal depthwise convolution
used by gated-delta-network decode. It submits to PyTorch's current XPU queue, so tensor
ownership and stream ordering remain inside the existing engine. FP32 and BF16 output
and in-place state updates are validated on the Arc B580.

The same extension includes direct packed GGUF matvec kernels for Q8_0, Q2_K, Q3_K,
Q4_K, Q5_K, IQ1_S, IQ1_M, IQ2_S, IQ2_XXS, IQ2_XS, IQ3_S, IQ3_XXS, and IQ4_XS. XPU
`GGUFLinear` dispatches these formats without materializing a full dense weight or
falling back to CPU. Synthetic FP32/BF16 device tests cover each format, real slices
from the Qwen3.8 checkpoint match the FP32 dequantized reference, and the complete
packed Qwen serving model has run through the public `LLM` API on the B580. These are
direct packed kernels, with IQ1_S, IQ2_S, IQ2_XXS, IQ2_XS, IQ3_S, and IQ3_XXS also
reusing decoded weights across four tokens for batches of four or more. The remaining
formats still repeat weight reads for each token, so larger prefill batches need
equivalent tiled kernels for those formats.

Build the extension with an `icpx` compiler from the same oneAPI release as the SYCL
runtime bundled by PyTorch. For the validated `torch==2.12.1+xpu` wheel, that is oneAPI
2025.3. Mixing a 2026 compiler with the wheel's `libsycl.so.8` produces an incompatible
extension even if compilation succeeds.

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

Run its B580 benchmark with:

```bash
PYTHONPATH=python .venv/bin/python benchmarks/bench_openvino_island.py \
  --source xpu --tokens 32 --hidden-size 1024 --output-size 4096
```

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

## Vulkan prototype

`experiments/vulkan` contains a deliberately isolated Vulkan compute prototype. It
compiles a GLSL dense-matrix shader to SPIR-V, selects only an Intel discrete GPU
(`vendorID=0x8086`), dispatches an 8x256 by 512 FP32 matrix multiplication, and checks
the result against a CPU reference. It will fail instead of selecting llvmpipe or
another software device.

```bash
bash experiments/vulkan/run_probe.sh
```

This proves native Vulkan compute on the B580 while keeping the experiment outside the
serving hot path. Host-visible Vulkan buffers still imply an explicit interoperability
boundary with PyTorch XPU, and the naive shader is a correctness probe rather than a
production GEMM. Promotion into a `KernelProvider` requires device-local tiled kernels,
reusable pipelines/descriptors, and measured transfer amortization over a substantially
larger compute island.
