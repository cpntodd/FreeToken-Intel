# Intel backend architecture proposal

Status: design for review. No serving path is changed by this proposal.

## Recommendation

Keep PyTorch XPU with the existing native SYCL kernels as FreeToken's native Intel
execution path. Treat OpenVINO as a future, explicit whole-model engine option, not as
a per-layer substitute inside the current XPU engine. Keep Vulkan experiments isolated
until they demonstrate model-level value and a safe memory/synchronization boundary.

| Technology | Proposed role | User-visible selection | Current readiness |
| --- | --- | --- | --- |
| PyTorch XPU + SYCL | Primary in-process Intel engine; PyTorch owns tensors, model, scheduler and caches, with SYCL kernels for selected operations | `--accelerator xpu` | Serving and B580 model paths already have evidence; expand kernel and hardware coverage |
| OpenVINO | Optional whole-model inference engine with its own model representation and generation runtime | A distinct engine selection, only after API/behavior parity review | Dense island is experimental; current B580 zero-copy boundary is not available in the installed runtime |
| Vulkan | Isolated shader/backend research | No serving option yet | Synthetic B580 matrix probe only; no model execution or PyTorch memory sharing |

## Why the boundary belongs above individual layers

FreeToken's `Engine` owns model construction and weight loading, accelerator selection,
KV-cache sizing, model-specific state, and runtime cache pools. The scheduler asks that
engine to execute batches and consumes its device-side token results. Replacing a single
linear operation with another runtime would therefore introduce a second device/runtime
boundary inside every decode step while leaving scheduling, cache ownership, and most
model work in PyTorch.

That trade has poor evidence so far. `OpenVINODenseIsland` compiles a dense projection
for the GPU but copies FP16 activations from XPU to host and copies results back. Its
recorded 32-token B580 projection was 2.21 ms versus 0.33 ms for eager XPU (6.6x slower),
with maximum absolute error 0.0078125. It is a measurement/prototyping seam, not a
serving optimization. Vulkan's cooperative-matrix results are also isolated kernel
measurements, and the current Vulkan buffers are not shared with PyTorch XPU.

The repository's existing execution shape is consequently the preferred native Arc
path: retain the current engine and scheduler, keep XPU as the tensor/runtime owner, and
dispatch only validated kernels to the SYCL extension. Do not silently switch a failed
XPU operation to CPU, OpenVINO, or Vulkan.

## Proposed OpenVINO direction

OpenVINO has become more relevant as a separate engine than as a dense layer provider.
Its 2026.4 release notes list GPU support for Gemma 4 12B and Qwen3.8 27B and describe
MTP support for several GPU LLM families. OpenVINO Model Server's current LLM quickstart
shows an OpenVINO IR Qwen model served on an Intel iGPU or dGPU. These are promising
capabilities, but they do not establish that FreeToken's local GGUF checkpoints,
custom formats, tool parsers, or B580 execution behavior are compatible.

If approved for implementation, add an engine-level boundary that returns a common
stream of token IDs, finish reason, and usage data to FreeToken's API layer:

```text
FreeToken API and request semantics
                 |
         generation engine
          /             \
 native FreeToken       OpenVINO GenAI engine
 Scheduler + Engine     its model, tokenizer, cache,
 Torch CUDA/XPU         and generation runtime
          |
   XPU uses SYCL for selected kernels
```

The OpenVINO branch must be explicit and must own a complete request from prompt
preparation through decoding. It must not enter the native `Engine.forward_batch`
path, share that path's KV-cache objects, or be called once per model layer. Initially,
limit it to model artifacts and architectures demonstrated to work with the pinned
OpenVINO GenAI version. Keep `--accelerator xpu` independent and unchanged. A remote
OpenVINO Model Server adapter is a possible alternative, but it adds service lifecycle,
network, cancellation, and streaming/error-mapping concerns; compare it with an
in-process GenAI engine before choosing.

### OpenVINO memory interoperability gate

The current experiment exercised OpenVINO's default GPU context, which reports OCL,
against a PyTorch XPU allocation produced by the Level Zero runtime. Wrapping that
pointer as `USM_USER_BUFFER` failed before inference with an allocation-size-zero
diagnostic. A new read-only host probe against the installed OpenVINO 2026.4 plugin
also tried `CONTEXT_TYPE=ZE`; `Core.create_context("GPU", ...)` rejected it with
`Level Zero interoperability is not supported`. Thus the public headers' `ZE` enum and
USM tensor wrappers do not prove that this installed GPU plugin can share a PyTorch XPU
context or allocation. Pointer identity, lifetime, queue synchronization, and parity
remain unproven. Keep host staging in the dense-island prototype; do not pass raw device
addresses through Python or rely on zero-copy for an engine design.

This runtime result is specific to the tested B580 host, Torch 2.12.1+xpu, OpenVINO
2026.4.0, and its current driver/plugin combination. It is not a claim that every
OpenVINO GPU build or supported external-memory path is incompatible.

## SYCL and Vulkan responsibilities

SYCL is not a second FreeToken model engine. The oneAPI DPC++ Level Zero backend is the
native device-kernel layer for Intel GPUs; in this repository, PyTorch XPU keeps device
and tensor ownership while the compiled SYCL extension implements selected operations.
Kernel dispatch remains guarded by supported dtype/shape conditions, with the established
XPU path retained for unsupported shapes such as multi-token cases where direct kernels
have not won. Continue requiring reference parity, real-device execution, and separate
decode/prefill measurements for each added kernel.

Vulkan should remain a research backend, not a third path in the production scheduler.
`VK_KHR_cooperative_matrix` requires runtime feature/property discovery, and the Vulkan
specification explicitly makes supported matrix sizes/types implementation-dependent.
The B580's reported 8x16x16 tile is device evidence, not a portable Arc guarantee. Before
serving integration, Vulkan needs persistent weight/pipeline ownership, representative
model operations (not only dense GEMM), batching and cancellation semantics, explicit
cross-API memory lifetime/synchronization, and end-to-end measurements including packing,
copies, and handoff back to the engine.

## Acceptance sequence before implementation

1. Freeze the OpenVINO GenAI version and choose a candidate model artifact already
   supported by that version; verify it loads and executes on this B580 with GPU-only
   execution evidence.
2. Prototype the engine-level adapter without changing the default FreeToken engine.
   Check tokenizer/template parity, token IDs, streaming, stop/EOS handling, cancellation,
   sampling controls, usage accounting, and errors. Explicitly list unsupported features.
3. Compare the same model, prompt, context, and output length against the native XPU path.
   Report load time, TTFT, inter-token latency, throughput, peak device memory, and any
   host/device transfers over repeated warmed and cold runs. Do not claim a speedup from
   a kernel-only result.
4. Exercise concurrency and shutdown behavior through FreeToken's actual API and
   establish how model/device selection is reported. Only then decide whether to retain
   an in-process GenAI adapter or an OVMS client backend.
5. For Vulkan, first prove persistent packed-weight operation and a representative
   decode/prefill workload with full transfer/synchronization accounting. Keep it
   experimental unless it demonstrates a clear supported use case.
6. Keep a compatibility matrix by Arc generation/device ID and driver/runtime. The
   B580 is the required development reference, but it alone cannot substantiate support
   for every Arc GPU.

## Review decision requested

Approve or reject the proposed boundary before serving implementation: should OpenVINO
be pursued as an explicit whole-model engine alongside the native engine, or should this
port focus exclusively on XPU/SYCL while OpenVINO and Vulkan remain non-serving
experiments? If OpenVINO is approved, the first engineering milestone should be a
single-model engine/API parity prototype, not per-layer integration.

## Current online references

- [OpenVINO 2026.4 release notes](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html)
- [OpenVINO Model Server LLM quickstart for Intel GPUs](https://docs.openvino.ai/2026/model-server/ovms_docs_llm_quickstart.html)
- [OpenVINO GPU Remote Tensor and interoperability API](https://docs.openvino.ai/2026/api/c_cpp_api/group__ov__runtime__ocl__gpu__cpp__api.html)
- [Intel oneAPI Level Zero backend interoperability](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2023-2/intel-oneapi-level-zero-backend-specification.html)
- [oneAPI SYCL specification](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/v1.3-rev-1/elements/sycl/source/)
- [Khronos Vulkan `VK_KHR_cooperative_matrix`](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_cooperative_matrix.html)
- [Khronos Vulkan specification](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html)
