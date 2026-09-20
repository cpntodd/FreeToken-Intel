# Intel backend architecture and experiment status

Status: updated to match the active user-approved scope. The default XPU/SYCL path is
unchanged. An experimental, opt-in OpenVINO island now covers Llama layer-0 QKV; it is
not a default backend or a performance optimization.

## Recommendation

Keep PyTorch XPU with the existing native SYCL kernels as FreeToken's native Intel
execution path. Add OpenVINO only as an explicit, bounded operation-level compute island
with opt-in XPU fallback; do not create a whole-model GenAI engine or replace the default
XPU path. Keep Vulkan experiments isolated until they demonstrate model-level value and
a safe memory/synchronization boundary.

| Technology | Proposed role | User-visible selection | Current readiness |
| --- | --- | --- | --- |
| PyTorch XPU + SYCL | Primary in-process Intel engine; PyTorch owns tensors, model, scheduler and caches, with SYCL kernels for selected operations | `--accelerator xpu` | Serving and B580 model paths already have evidence; expand kernel and hardware coverage |
| OpenVINO | Opt-in GPU compute island for a selected bounded dense operation | `--openvino-island llama.layer0.qkv`; fallback is `none` or explicitly `xpu` | Llama layer-0 QKV serving call site is integrated and ran on B580; host staging benchmark is slower than eager XPU |
| Vulkan | Isolated shader/backend research | No serving option yet | Synthetic B580 matrix probe only; no model execution or PyTorch memory sharing |

## Why the OpenVINO island remains bounded and opt-in

FreeToken's `Engine` owns model construction and weight loading, accelerator selection,
KV-cache sizing, model-specific state, and runtime cache pools. The scheduler asks that
engine to execute batches and consumes its device-side token results. Calling another
runtime for a single linear operation introduces a second device/runtime boundary inside
the decode step while scheduling, cache ownership, and the rest of the model stay in
PyTorch. The active objective specifically selects this bounded-island experiment, so it
must remain explicit and measured rather than becoming an implicit backend substitution.

That trade has poor performance evidence so far. `OpenVINODenseIsland` compiles a dense
projection for the GPU but copies FP16 activations from XPU to host and copies results
back. A repeated 32-token B580 run at 5120 -> 10240 measured 1.98 ms for the complete
island versus 0.331 ms for eager XPU (5.99x slower), with maximum absolute error
0.0078125. It is a measurement/prototyping seam, not a serving optimization. Vulkan's
cooperative-matrix results are also isolated kernel measurements, and the current Vulkan
buffers are not shared with PyTorch XPU.

The repository's existing execution shape is consequently the preferred native Arc
path: retain the current engine and scheduler, keep XPU as the tensor/runtime owner, and
dispatch only validated kernels to the SYCL extension. Do not silently switch a failed
XPU operation to CPU, OpenVINO, or Vulkan.

## Active OpenVINO scope: bounded compute islands

The active objective selects bounded OpenVINO-compiled compute islands, not a separate
whole-model GenAI engine. The native XPU/SYCL engine, model representation, scheduler,
and KV-cache remain the owners of inference. An island is an explicit opt-in for a
selected dense operation with a strict token bound; it must not silently replace all
linears in a layer or model.

The current `OpenVINODenseIsland` implements GPU-only compilation, host FP16 staging,
copy/inference timing, a bounded token count, and default-fail behavior. Its single model
call site is `model.layers.0.self_attn.qkv_proj`, selected with
`--openvino-island llama.layer0.qkv`. The Llama GGUF loader provides this projection as
a dense BF16 matrix; Engine validates the model, XPU residency, and single-GPU setup,
then compiles the island after loading weights and before its memory snapshot. OpenVINO
is not imported on the default path. The default limit is 64 tokens per invocation; an
oversized prefill raises unless the user explicitly selects `--openvino-island-fallback
xpu`. XPU execution is otherwise available only with that explicit fallback. Per-call
backend, fallback reason, and transfer/inference timings are available in debug logs;
fallback use is warned once.

The opt-in serving path ran the local Llama-3.2-1B-Instruct-Q4_K_M GGUF on the host B580
(`GPU.0`, reported as Intel Arc B580), compiled the QKV island, and returned `Hello!`
for a 35-token prompt. This is a short functional smoke, not a quality or speed claim.
The 32-token synthetic projection benchmark remains 5.99x slower than eager XPU due to
host staging. Comparing the loaded QKV projection for the exact same 35-token activation
on B580 measured max absolute difference 0.0625 and relative L2 difference 0.0015125;
the API output matched the native XPU smoke. This is one prompt/checkpoint observation,
not a universal error bound or an end-to-end quality/throughput comparison. The Engine's
fixed-memory snapshot occurs after island compilation and conservatively groups its
persistent device allocations with model weights before sizing runtime caches.

Do not use the measured staging path as a performance optimization: the B580 run was
5.99x slower than eager XPU for the tested projection. Keep it as an explicit
experimental capability unless a future supported memory-interoperability path or a
larger fused island changes the end-to-end result. A whole-model OpenVINO GenAI engine
and an OVMS client are outside the current objective and require a separate decision.

### Acceptance for the selected operation

1. The selected operation is `model.layers.0.self_attn.qkv_proj`, using the dense BF16
   matrix materialized by the Llama GGUF loader and a default 64-token invocation bound;
   do not auto-replace modules across the model.
2. Keep the native XPU/SYCL implementation as the default. Require explicit island
   selection and explicit XPU fallback, and report the backend actually used.
3. The B580 same-activation loaded-QKV comparison and serving smoke pass; focused tests
   cover synthetic OpenVINO GPU parity, explicit XPU fallback, and default error
   propagation without CPU execution.
4. Compare identical inputs and report host staging, OpenVINO inference, output handoff,
   and total latency. Keep the island opt-in if end-to-end evidence does not justify it.
5. Record tested GPU IDs and runtime/driver versions. B580 results do not substantiate
   support claims for every Intel Arc GPU.

## Historical whole-model OpenVINO alternative (superseded)

The following adapter proposal and its acceptance sequence are retained as research
context only. The active objective above supersedes this whole-model direction.

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

## OpenVINO memory interoperability evidence

The current experiment exercised OpenVINO's default GPU context, which reports OCL,
against a PyTorch XPU allocation produced by the Level Zero runtime. Wrapping that
pointer as `USM_USER_BUFFER` failed before inference with an allocation-size-zero
diagnostic. A new read-only host probe against the installed OpenVINO 2026.4 plugin
also tried `CONTEXT_TYPE=ZE`; `Core.create_context("GPU", ...)` rejected it with
`Level Zero interoperability is not supported`. Thus the public headers' `ZE` enum and
USM tensor wrappers do not prove that this installed GPU plugin can share a PyTorch XPU
context or allocation. Pointer identity, lifetime, queue synchronization, and parity
remain unproven. Keep host staging in the dense-island prototype; do not pass raw device
addresses through Python or assume zero-copy in the current island integration.

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

## Historical acceptance sequence for the whole-model alternative

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

## Resolved scope decision

The active user objective resolves the earlier choice in favor of bounded OpenVINO
operation-level islands with explicit XPU fallback. It does not approve a whole-model
GenAI/OVMS engine or default per-layer dispatch. The selected Llama layer-0 QKV call
site is now integrated, and its loaded-weight parity check plus one API smoke pass on
B580. Broader serving validation and coverage beyond B580 remain unfinished.

## Current online references

- [OpenVINO 2026.4 release notes](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html)
- [OpenVINO Model Server LLM quickstart for Intel GPUs](https://docs.openvino.ai/2026/model-server/ovms_docs_llm_quickstart.html)
- [OpenVINO GPU Remote Tensor and interoperability API](https://docs.openvino.ai/2026/api/c_cpp_api/group__ov__runtime__ocl__gpu__cpp__api.html)
- [Intel oneAPI Level Zero backend interoperability](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2023-2/intel-oneapi-level-zero-backend-specification.html)
- [oneAPI SYCL specification](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/v1.3-rev-1/elements/sycl/source/)
- [Khronos Vulkan `VK_KHR_cooperative_matrix`](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_cooperative_matrix.html)
- [Khronos Vulkan specification](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html)
