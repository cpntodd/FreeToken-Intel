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

## Hardware validation scope

FreeToken has been validated on an Arc B580 only. Other Arc discrete GPUs and Intel
integrated Arc GPUs are targets for broader support, but are not yet verified by this
project. Upstream PyTorch's [XPU hardware and OS requirements](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/get_start_xpu.md)
list additional Arc and Core Ultra GPU configurations; Intel's
[compute-runtime device list](https://github.com/intel/compute-runtime) and
[DPC++ GPU target list](https://intel.github.io/llvm/design/OffloadDesign.html) describe
lower-level driver and compiler coverage. Those upstream lists identify potential
platforms, not FreeToken compatibility. Each additional GPU family still needs native
kernel checks and an end-to-end model run before it can be called validated here.

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

## Verify the installed XPU wheel

To check packaging independently of editable/source-tree imports, build against the
active XPU environment and install the wheel into a temporary target without resolving
or replacing its Torch stack:

```bash
FT_XPU_WHEEL_TMP=$(mktemp -d /tmp/freetoken-xpu-wheel.XXXXXX)
FT_REPO_ROOT=$PWD
FREETOKEN_ACCELERATOR=xpu \
ONEAPI_ROOT=/opt/intel/oneapi \
FREETOKEN_SYCL_COMPILER_VERSION=2025.3 \
CXX=/opt/intel/oneapi/compiler/2025.3/bin/icpx \
  "$FT_REPO_ROOT/.venv/bin/python" setup.py \
  build --build-base "$FT_XPU_WHEEL_TMP/build" \
  bdist_wheel \
  --dist-dir "$FT_XPU_WHEEL_TMP/wheel" \
  --bdist-dir "$FT_XPU_WHEEL_TMP/bdist"
uv pip install --target "$FT_XPU_WHEEL_TMP/install" --no-deps \
  "$FT_XPU_WHEEL_TMP"/wheel/freetoken-*.whl
```

Run the smoke from outside the checkout. It checks that both Python and the compiled
extension came from the installed wheel, reports the selected device, and compares a
native batched Q4_0 SYCL result with the dequantized reference:

```bash
(
  cd /tmp
  FT_XPU_WHEEL_ROOT="$FT_XPU_WHEEL_TMP/install" \
  PYTHONPATH="$FT_XPU_WHEEL_TMP/install" FREETOKEN_ACCELERATOR=xpu \
    "$FT_REPO_ROOT/.venv/bin/python" - <<'PY'
import os
import torch
import freetoken
from freetoken.accelerator.runtime import resolve_runtime
from freetoken.kernel import _sycl_kernels
from freetoken.layers.gguf import fused_mul_mat_gguf
from freetoken.models.gguf.dequant import GGML_Q4_0, dequantize

wheel = os.environ["FT_XPU_WHEEL_ROOT"] + "/"
assert freetoken.__file__.startswith(wheel), freetoken.__file__
assert _sycl_kernels.__file__.startswith(wheel), _sycl_kernels.__file__
runtime = resolve_runtime("xpu")
caps = runtime.capabilities(0)
generator = torch.Generator().manual_seed(89)
qweight = torch.randint(0, 256, (11, 54), dtype=torch.uint8, generator=generator)
qweight.view(11, 3, 18)[:, :, :2] = torch.tensor([0.5], dtype=torch.float16).view(torch.uint8)
x = torch.randn(4, 96, generator=generator)
weight = dequantize(qweight, GGML_Q4_0, torch.float32).reshape(11, 96)
expected = x.float() @ weight.T
actual = fused_mul_mat_gguf(x.to("xpu"), qweight.to("xpu"), GGML_Q4_0)
runtime.synchronize()
torch.testing.assert_close(actual.cpu().float(), expected, rtol=5e-5, atol=5e-5)
print(caps)
print("Q4_0 output:", tuple(actual.shape), "max abs error:",
      (actual.cpu().float() - expected).abs().max().item())
PY
  PYTHONPATH="$FT_XPU_WHEEL_TMP/install" FREETOKEN_ACCELERATOR=xpu \
    "$FT_XPU_WHEEL_TMP/install/bin/ft" devices --json
)
```

On 2026-09-21, this check passed on the host Arc B580 (`0xE20B`, driver
`1.6.33578+15`) with `torch==2.12.1+xpu` and oneAPI DPC++ 2025.3.3. Both package
imports resolved from the temporary installation, and the `[4, 96]` by Q4_0 operation
returned shape `[4, 11]` with maximum absolute error `7.63e-6`. `nvtop -s` identified
Battlemage G21 / Arc B580 and reported 4% memory use after the operation; its utilization
field was unavailable in that idle snapshot. The installed `ft devices --json` command
also reported XPU `available`, CUDA `unavailable`, device `xpu:0`, and the eager/single-GPU
engine constraints from the wheel.

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

The HTTP serving path was also exercised on 2026-09-21 with the same Llama checkpoint.
Start the B580 XPU server:

```bash
PYTHONPATH=python .venv/bin/python -m freetoken.cli serve \
  --model-path /home/oddsoul/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --accelerator xpu --text-model-only --host 127.0.0.1 --port 8787 \
  --max-output-tokens 8 --max-seq-len-override 128 --num-tokens 256 \
  --attention-backend torch
```

Then send a local OpenAI-compatible request:

```bash
curl --fail-with-body -sS http://127.0.0.1:8787/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Llama-3.2-1B-Instruct-Q4_K_M.gguf","messages":[{"role":"user","content":"Say hello in one short sentence."}],"max_tokens":4,"temperature":0}'
```

The server started with the explicit XPU runtime, allocated the 256-token KV cache,
and returned HTTP 200 with `Hello!` and three completion tokens. `nvtop` identified the
Arc B580 and observed 28% memory use during loading. This validates one API/scheduler
inference path; it is not a model-quality or throughput result.

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

As a smaller-checkpoint control, the same probe compared
`Llama-3.2-1B-Instruct-Q4_K_M.gguf` on a 35-token prompt. FreeToken XPU and Prism
Vulkan0 on the B580 both selected token ID 9906, with identical top-20 IDs; the mean
and maximum absolute log-probability differences among those tokens were 0.049 and
0.105. Prism's loaded model used B580 memory according to `nvtop`. This is closer
agreement than the Qwen Q2_K run, but the model architecture and quantization both
differ, so it only narrows the discrepancy to the Qwen/checkpoint path; it does not
identify Q2_K or prove full-logit parity.

A same-family quantized-checkpoint control used
`Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` with the same 59 prompt token IDs. FreeToken
XPU and Prism Vulkan0 on the B580 both selected token ID 1596 (`We`); 19/20 top-token
IDs overlapped, with mean and maximum shared-token absolute log-probability
differences of 0.124 and 0.299. `nvtop` showed 99% B580 memory use for FreeToken and
83% while Prism was loaded. Prism logged a memory-fit warning with the explicit
`-ngl 99` setting and ignored the checkpoint's extra MTP block, so this is exploratory
first-token evidence, not proof of full Vulkan offload or MTP parity. The closer
scores than the Qwen Q2_K run are consistent with quantization or checkpoint
differences contributing, but do not isolate either cause.

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
IQ3_S, IQ3_XXS, IQ4_NL, and IQ4_XS. Synthetic FP32/BF16 device tests cover each format.
Real slices of formats present in the Qwen3.8 checkpoint match the FP32 dequantized
reference, and the complete packed Qwen serving model has run through the public
`LLM` API on the B580. Q4_0, Q4_1, Q5_0, Q5_1, and IQ4_NL use native SYCL for
single-token decode. Q4_0 also uses direct SYCL through 16-token batches; larger batches
retain the existing chunked dequantize-plus-matmul path. The supplied model set includes
Q4_0 in Gemma 4 but has no Q4_1, IQ4_NL, Q5_0, or Q5_1 tensors, so those four have
synthetic rather than full-model coverage. Warmed 15-iteration B580 microbenchmarks at
`[1, 4096] x [4096, 4096]` (BF16)
measured median direct-SYCL vs XPU dequantize-plus-matmul times of 0.677 vs 2.841 ms for
Q4_1, 0.428 vs 4.128 ms for Q5_0, 0.520 vs 4.584 ms for Q5_1, and 0.341 vs 3.361 ms
for IQ4_NL; output parity passed for all four. These are synthetic samples, not
end-to-end model performance. The focused GGUF dequant and SYCL accelerator suites
passed 157 tests on the B580. The dispatch avoids CPU fallback; larger Q4_0 batches and
all prompt batches for the other four formats remain on XPU matmul.

A separate batch sweep compared the five kernels above with the exact XPU
dequantize-plus-matmul fallback at `[tokens, 5120] x [4096, 5120]` in BF16. On the
Arc B580, with three warmups and 15 timed iterations, all 25 cases passed output parity.
Across formats, direct-SYCL speedups were 6.0-11.8x at one token, 7.1-10.8x at four,
3.9-5.7x at eight, and 2.0-2.9x at sixteen. At 32 tokens the range narrowed to
0.99-1.46x; Q4_1 was marginally slower than the fallback. These synthetic timings do
not establish end-to-end model gains. Reproduce with
`benchmarks/bench_sycl_gguf_batches.py` (documented in `benchmarks/README.md`).

A same-model B580 comparison then tested the 16-token Q4_0 cutoff on the supplied
`gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` (329 Q4_0 tensors). Each path used this same command
with prompt `Hi.`, a 128-token context, 256 KV tokens, and EOS ignored until the 16-token
limit:

```bash
FREETOKEN_ACCELERATOR=xpu PYTHONPATH=python .venv/bin/python \
  benchmarks/bench_xpu_gemma4.py \
  /home/oddsoul/models/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
  --prompt 'Hi.' --max-tokens 16 --force-decode-length \
  --context 128 --kv-tokens 256
```

Three fresh-process runs per path returned identical token IDs. Against the existing
decode-only dispatch, the experimental cutoff changed median TTFT from 19.46 s to 18.92 s,
decode from 3.291 to 3.308 tokens/s, and end-to-end output rate from 0.633 to 0.646
tokens/s. This is a modest observed gain for this B580 checkpoint and prompt; the repeated
TTFT range shows run-to-run variance, so it is not a general performance claim. The cutoff
applies only to Q4_0; the other four formats remain decode-only until a supplied model can
validate them.
Other active-serving packed formats retain their small-batch and four-token tiled paths.
In the recorded Qwen3.8 checkpoint, Q6_K is
present only in the MTP layer that serving currently omits. A bounded probe now reads
16 packed rows from the real `blk.64.attn_output.weight` tensor (GGML shape
`[6144, 5120]`) and checks three input tokens against the canonical dequantized CPU
reference on B580 `xpu:0`. FP32 max absolute/relative errors were `2.50e-6` / `1.48e-5`;
BF16 outputs matched exactly. A subsequent FP32 run covered all 5,120 rows of that real
tensor for the same three inputs; it passed with max absolute error `3.43e-5` and max
relative error `2.18e-2`. The full-tensor run validates the kernel against every packed
row in this checkpoint tensor, but still does not execute the omitted MTP layer or a
full Q6_K model. The probe copies only the requested rows:

```bash
FREETOKEN_ACCELERATOR=xpu PYTHONPATH=python .venv/bin/python \
  experiments/probe_q6_k_real_tensor.py \
  /home/oddsoul/models/Qwen3.8-27B-UD-Q2_K_XL.gguf \
  --dtype fp32 --rows 5120 --tokens 3
```

The local `Llama-3.2-1B-Instruct-Q4_K_M.gguf` also contains Q6_K weights in active
decoder layers. A direct SYCL probe of its `blk.0.ffn_down.weight` tensor (shape
`[8192, 2048]`) covered all 2,048 rows for three FP32 inputs on B580 `xpu:0`; it passed
with max absolute/relative errors of `6.44e-6` / `7.95e-3`. This tests the native kernel
against a real active-layer tensor, but not the Llama serving dispatch: the current
Llama loader dequantizes Q6_K to BF16 before its fused projection.

```bash
FREETOKEN_ACCELERATOR=xpu PYTHONPATH=python .venv/bin/python \
  experiments/probe_q6_k_real_tensor.py \
  /home/oddsoul/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --tensor blk.0.ffn_down.weight --dtype fp32 --rows 2048 --tokens 3
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
  --source xpu --tokens 32 --hidden-size 5120 --output-size 10240
```

To measure the opt-in fallback path after an OpenVINO failure, add `--fallback xpu`.
The benchmark requires `--source xpu` for that policy and reports the actual backend,
fallback reason, and backend counts rather than labeling XPU fallback timings as
OpenVINO timings.

The benchmark compares the complete OpenVINO path, including host staging, with
source-device eager execution and checks output parity. Repeating the 32-token
`5120 -> 10240` projection on the host B580 with three warmups and 20 iterations
measured a median
1.981 ms through the OpenVINO island versus 0.331 ms for eager XPU (5.99x slower);
maximum absolute output error was 0.0078125. Median input staging, inference, and output
staging were 0.493 ms, 0.683 ms, and 0.706 ms, respectively. This is evidence for the
current staging cost, not a general OpenVINO-versus-XPU performance claim. A zero-copy
or larger fused island requires a separate benchmark.

The serving integration is experimental and limited to the dense
`model.layers.0.self_attn.qkv_proj` operation in Llama. The GGUF loader materializes
that weight as dense BF16; the island compiles after model loading, only when explicitly
selected, and defaults to a 64-token invocation limit with no fallback. A longer prefill
raises unless XPU fallback is explicitly enabled.

The island accepts only unquantized FP16/BF16/FP32 matrices; checkpoint quantization
metadata (including FP8 side scales) is rejected, even with `fallback=xpu`, because the
island's direct fallback cannot apply those scales. Leave the island disabled for those
checkpoints.

To try it on the tested B580:

```bash
FREETOKEN_ACCELERATOR=xpu PYTHONPATH=python \
  .venv/bin/python -m freetoken.cli serve \
    --model-path /home/oddsoul/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --accelerator xpu --openvino-island llama.layer0.qkv \
    --host 127.0.0.1 --port 8787 --text-model-only \
    --max-output-tokens 8 --max-seq-len-override 128 --num-tokens 256 \
    --attention-backend torch
```

Use `--openvino-island-fallback xpu` only when an explicit native-XPU fallback is
desired. Without it, import, compilation, execution, and over-limit errors propagate.
The Engine rejects other model families, non-XPU execution, and tensor parallelism.
During the 2026-09-21 host smoke, startup reported OpenVINO `GPU.0` as Intel Arc B580
with `max_tokens=64` and `fallback=none`; the 35-token prompt returned `Hello!` with
three completion tokens. The same request had already returned `Hello!` on the native
XPU path. This verifies one short functional serving path, not general model quality or
throughput. Only the B580 has been device-validated; this does not establish coverage
for every Arc GPU.

An instrumented comparison in the worker used the loaded checkpoint's actual layer-0
QKV weight and the same 35-token activation for native XPU and OpenVINO. It reported
`backend=openvino`, no fallback, max absolute output difference `0.0625`, and relative
L2 difference `0.0015125`. This is a single observed prefill activation, not a general
numerical-error guarantee. Engine cache sizing snapshots memory after island compilation
and reserves persistent island allocations together with fixed model memory.

OpenVINO's C++ GPU Remote Tensor API supports USM pointers, but direct ownership-safe
PyTorch XPU interoperability has not yet been proven in this Python integration. Host
staging remains intentional until that proof exists.

The OpenVINO 2026.4 Python `RemoteContext.create_tensor` binding was tested with a
live PyTorch XPU `data_ptr()` and `SHARED_MEM_TYPE=USM_USER_BUFFER`. Python integers,
`ctypes.c_void_p`, and NumPy pointer scalars were all rejected before tensor creation;
the binding does not expose a safe raw-pointer conversion. In addition, OpenVINO's
default GPU context reports `CONTEXT_TYPE=OCL`, while PyTorch XPU normally submits via
oneAPI/Level Zero. Current OpenVINO documentation describes the GPU plugin as
OpenCL-based and exposes its GPU remote context as `ClContext`. The current upstream
GPU-plugin source rejects shared `ContextType::ZE` contexts and marks shared remote
contexts unsupported with the SYCL runtime. A C++ extension does not by itself make a
direct PyTorch Level-Zero allocation importable by OpenVINO's GPU plugin; passing an
integer device address through Python is not a supported path. OpenVINO documents
external shared-memory handles, including DMA-BUF on Linux. The Level Zero
specification requires an allocation to request `ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF`
when it is created before that allocation can be exported as DMA-BUF; this is not a
retroactive export for an arbitrary pointer. This probe has not established a supported
way to create Torch XPU allocations with that export property or to validate handle
lifetime and synchronization. Keep the host-staged path until an interop route is
demonstrated end to end.

References: [OpenVINO GPU device documentation](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html),
[GPU Remote Tensor API](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device/remote-tensor-api-gpu-plugin.html),
[upstream GPU remote-context implementation](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/src/plugin/remote_context.cpp),
and [Level Zero external-memory programming guide](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html#external-memory-import-and-export).

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

A fresh direct `best` run on the `8x256` by `256x512` shape again selected the B580
cooperative FP16/FP32 path with an `8x16x16` tile. Across 20 iterations, median dispatch
was 0.255 ms and median upload-plus-dispatch was 0.255 ms; weight upload was 0.277 ms
and max absolute error was `8.34e-7`. The host smoke test also passed its explicit B580
device-ID (`0xE20B`) and parity assertions. This is a single synthetic-operation check,
not persistent model-weight or serving evidence.

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
