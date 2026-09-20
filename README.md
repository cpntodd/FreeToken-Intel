# FreeToken-Intel

Edge-native Mixture-of-Experts (MoE) inference on Intel Arc GPUs.

> Adding wider support for OpenVINO, SYCL, and Vulkan backends for Intel Arc
> GPUs. Hardware used in testing: Intel Arc B580 with 12 GB VRAM.

This fork is based on [FreeToken](https://github.com/FlashML-org/FreeToken),
an engine designed to serve open-weight MoE models across GPU, CPU, system
memory, and PCIe. This fork preserves that architecture and is extending its
acceleration paths to Intel Arc. The CUDA path remains available.

## Intel Arc status

Intel Arc support is experimental and under active development.

- **PyTorch XPU / Level Zero:** eager inference runs on Arc B580. XPU
  selection is explicit and does not fall back to CPU or CUDA.
- **Native SYCL:** available for causal-convolution decode and direct packed
  GGUF matvecs. Hardware validation includes Llama 3.2, Qwen3.8, Gemma 4, and
  classic Bonsai PQ2_0 checkpoints.
- **OpenVINO:** a bounded dense GPU compute-island prototype is available. It
  currently stages tensors through host memory and is not a zero-copy
  production backend.
- **Vulkan:** an isolated Intel GPU compute prototype is available for
  correctness experiments. It is not integrated into the serving path.

Production OpenVINO and Vulkan integration, broader model coverage, and further
SYCL kernel work remain in progress. See [Intel XPU development](docs/intel-xpu.md)
for the tested software stack, setup steps, benchmark commands, and known
limitations.

## Why the Arc B580

| | Intel Arc B580 |
| --- | --- |
| VRAM | 12 GB |
| PCI device ID | `0xE20B` |
| Validated runtime | PyTorch XPU, Level Zero V2 |
| Validated driver | `1.6.33578+15` |

The B580 is the current development and hardware-validation target. Results in
the Intel guide are functional or benchmark evidence for the listed workload
and configuration; they should not be treated as general performance claims.

## Current B580 benchmark snapshot

The following single-run measurements were collected on the Arc B580 on
2026-09-20 with the same prompt (`Say hello in one short sentence.`), 16 requested
tokens, forced decode length, context 128, and KV cache 256. Model-specific chat
templates produce different prompt-token counts. Decode rate excludes the first
token; TTFT includes prompt processing. Forced-length output can repeat chat/control
tokens, so these runs measure execution timing, not answer quality. Treat each row as
a development baseline for that model and path; token rates are not directly
comparable across models. Full benchmark details and backend limitations are in
[docs/intel-xpu.md](docs/intel-xpu.md).

| Model | Backend used | Decode speed | Model load time | Comments |
| --- | --- | ---: | ---: | --- |
| Llama 3.2 1B Instruct Q4_K_M | PyTorch XPU; mixed Q4_K/Q6_K weights expanded to BF16 | 18.29 tok/s | 23.45 s | TTFT 3.76 s; 35 prompt tokens and 13 measured decode tokens. Forced output repeated chat/control tokens. |
| Qwen3.8 27B UD Q2_K_XL | PyTorch XPU + native SYCL packed matvecs | 0.65 tok/s | 47.39 s | TTFT 37.58 s; 61 prompt tokens and 14 measured decode tokens. 9.47 GB model state; experimental packed path, with reference parity still pending. |
| Gemma 4 12B Q4_K_XL | PyTorch XPU prefill + native SYCL Q4_0 decode | 2.85 tok/s | 49.05 s | TTFT 18.99 s; 20 prompt tokens and 14 measured decode tokens. Forced output repeated control tokens; timing varies between runs. |
| Ternary-Bonsai 27B PQ2_0 | PyTorch XPU + native SYCL PQ2_0 matvec | 2.10 tok/s | 39.74 s | TTFT 23.20 s; 19 prompt tokens and 14 measured decode tokens. Classic PQ2_0 checkpoint; Bonsai 2 remains unsupported. |

Reproduce a row by substituting its local GGUF path in this command:

```bash
FREETOKEN_ACCELERATOR=xpu PYTHONPATH=python \
  .venv/bin/python benchmarks/bench_xpu_gemma4.py /path/to/model.gguf \
  --prompt "Say hello in one short sentence." \
  --max-tokens 16 --force-decode-length --context 128 --kv-tokens 256
```

OpenVINO and Vulkan have prototype measurements, but no model-level throughput
results: OpenVINO currently measures a host-staged dense projection, while Vulkan
is limited to an isolated correctness probe. See the Intel guide for those results.

## Repository layout

- `python/freetoken/` - inference engine, model support, scheduler, caches,
  server, and CLI
- `python/freetoken/kernel/csrc/sycl/` - native SYCL kernels
- `experiments/vulkan/` - standalone Vulkan compute prototype
- `docs/` - installation, model, CLI, and Intel development guides
- `tests/` - unit and accelerator tests

## Install

Clone this fork:

```bash
git clone https://github.com/cpntodd/FreeToken-Intel.git
cd FreeToken-Intel
```

Intel Arc development currently needs a dedicated XPU environment. Follow
[docs/intel-xpu.md](docs/intel-xpu.md) for the validated PyTorch XPU and
Triton-XPU versions and editable-install commands. The generic
`freetoken[accel]` install is CUDA-oriented and does not install the Intel
dependency set.

## CLI

Check detected accelerator devices and backend status:

```bash
ft devices
ft devices --json
```

For XPU setup, inference examples, and backend-specific experiments, see
[docs/intel-xpu.md](docs/intel-xpu.md).

## Citation

If you use FreeToken for research, cite the [FreeToken paper](https://arxiv.org/abs/2608.16157).

## Acknowledgment

FreeToken builds on ideas and code from
[mini-sglang](https://github.com/sgl-project/mini-sglang),
[SGLang](https://github.com/sgl-project/sglang),
[vLLM](https://github.com/vllm-project/vllm),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
[LightLLM](https://github.com/ModelTC/lightllm), and
[llama.cpp](https://github.com/ggml-org/llama.cpp).

## License

[Apache License 2.0](https://github.com/FlashML-org/FreeToken/blob/main/LICENSE).
