# benchmarks

Run from the repo root with `PYTHONPATH=python:.`, pinned to one GPU
(`CUDA_VISIBLE_DEVICES=0`). Each script's `--help` / docstring has the details.

**`bench_decode_moe.py`** — bs=1 decode tok/s of a served MoE model. Spawns `ft serve`
per backend and times token arrivals over streamed `/v1/chat/completions`, so numbers
include the full serving path. AIME-25 prompt, checkpoint-recommended sampling.

```bash
python benchmarks/bench_decode_moe.py --model /path/to/model --backend offload,cpu,hybrid
```

**`bench_load_weight_generic.py`** — expert-bank load time: serial vs parallel O_DIRECT
vs pre-repacked FTW, each mode in its own subprocess. Linux-only; stages the FTW under
`/var/tmp` (`--ftw-dir` overrides; roughly checkpoint-sized).

```bash
python benchmarks/bench_load_weight_generic.py --model /path/to/model
```

**`bench_offload_cache_copy.py`** — synthetic (no checkpoint): per-layer decode expert
copy cost (`ensure_experts` + `copy_missing`), swept over bank layout x cache slots x
batch size x miss rate.

```bash
python benchmarks/bench_offload_cache_copy.py
```

**`bench_sycl_gguf_batches.py`** — B580 microbenchmark comparing direct packed SYCL
matvecs for Q4_0, Q4_1, IQ4_NL, Q5_0, and Q5_1 against the XPU dequantize-plus-matmul
path. Uses synthetic packed weights and checks output parity before timing; it is not an
end-to-end model benchmark.

```bash
FREETOKEN_ACCELERATOR=xpu PYTHONPATH=python .venv/bin/python \
  benchmarks/bench_sycl_gguf_batches.py \
  --tokens 1,4,8,16,32 --rows 4096 --in-features 5120 \
  --dtype bf16 --warmup 3 --iterations 15
```

For host RAM vs PCIe bandwidth and the offload/hybrid backend pick, use `ft bench bw`
instead — it writes the JSON profile the engine reads.
