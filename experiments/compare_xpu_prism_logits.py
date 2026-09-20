from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import torch
from freetoken.core import SamplingParams
from freetoken.llm import LLM
from freetoken.mm.config import ENCODER_KINDS, MultimodalConfig


def _write_json(path: str, value: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2) + "\n")


def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _request_json(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def capture_xpu(args: argparse.Namespace) -> dict:
    if args.top_k < 1:
        raise ValueError("--top-k must be at least one")
    llm = LLM(
        args.model,
        dtype=torch.bfloat16,
        accelerator="xpu",
        attention_backend="auto",
        cuda_graph_max_bs=0,
        max_running_req=1,
        max_seq_len_override=args.context,
        num_token_override=args.kv_tokens,
        mm=MultimodalConfig(disabled_encoders=frozenset(ENCODER_KINDS)),
    )
    captured: dict = {}
    try:
        engine = llm.engine
        if engine.runtime.kind != "xpu" or engine.device.type != "xpu":
            raise RuntimeError("FreeToken did not bind an XPU device")

        encoded = llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        input_ids = encoded["input_ids"]
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        prompt_tokens = [int(token) for token in input_ids]

        original_sample = engine.sampler.sample

        def capture_sample(batch_logits, sample_args):
            if not captured:
                log_probs = torch.log_softmax(
                    batch_logits[0].detach().float().cpu(), dim=-1
                )
                values, indices = torch.topk(
                    log_probs, min(args.top_k, log_probs.numel())
                )
                captured["top_logprobs"] = [
                    {"id": int(token), "logprob": float(value)}
                    for token, value in zip(indices, values, strict=True)
                ]
                captured["logits_shape"] = list(batch_logits.shape)
            return original_sample(batch_logits, sample_args)

        engine.sampler.sample = capture_sample
        generated = llm.generate(
            [prompt_tokens],
            SamplingParams(max_tokens=1, temperature=0.0),
        )[0]
        index = 0 if engine.device.index is None else engine.device.index
        capabilities = engine.runtime.capabilities(index)
        captured.update(
            {
                "source": "FreeToken",
                "model": str(Path(args.model).resolve()),
                "prompt": args.prompt,
                "prompt_tokens": prompt_tokens,
                "prompt_token_count": len(prompt_tokens),
                "context": args.context,
                "kv_tokens": args.kv_tokens,
                "top_k": len(captured["top_logprobs"]),
                "device": capabilities.name,
                "device_id": capabilities.device_id,
                "generated_token_ids": generated["token_ids"],
                "generated_text": generated["text"],
            }
        )
    finally:
        llm.shutdown()

    if "top_logprobs" not in captured:
        raise RuntimeError("FreeToken did not expose a pre-sampler logits row")
    _write_json(args.output, captured)
    return captured


def capture_prism(args: argparse.Namespace) -> dict:
    xpu = _read_json(args.xpu_json)
    base_url = args.server_url.rstrip("/")
    properties = _request_json(f"{base_url}/props")
    prism_model = Path(properties["model_path"]).resolve()
    if prism_model != Path(xpu["model"]).resolve():
        raise RuntimeError(
            f"Prism model {prism_model} does not match FreeToken model {xpu['model']}"
        )

    count = len(xpu["top_logprobs"])
    response = _request_json(
        f"{base_url}/completion",
        {
            "prompt": xpu["prompt_tokens"],
            "n_predict": 1,
            "n_probs": count,
            "temperature": -1.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "repeat_last_n": 0,
            "samplers": ["temperature"],
            "post_sampling_probs": False,
            "return_tokens": True,
            "cache_prompt": False,
        },
    )
    probabilities = response["completion_probabilities"][0]
    result = {
        "source": "Prism llama-server",
        "model": str(prism_model),
        "server_build": properties.get("build_info"),
        "requested_backend": args.reference_backend,
        "prompt_tokens": xpu["prompt_tokens"],
        "prompt_token_count": len(xpu["prompt_tokens"]),
        "generated_token_ids": response.get("tokens", []),
        "top_logprobs": [
            {"id": int(item["id"]), "logprob": float(item["logprob"])}
            for item in probabilities["top_logprobs"]
        ],
    }
    _write_json(args.output, result)
    return result


def compare(args: argparse.Namespace) -> dict:
    xpu = _read_json(args.xpu_json)
    prism = _read_json(args.prism_json)
    if xpu["prompt_tokens"] != prism["prompt_tokens"]:
        raise RuntimeError("FreeToken and Prism prompt token IDs differ")
    if Path(xpu["model"]).resolve() != Path(prism["model"]).resolve():
        raise RuntimeError("FreeToken and Prism checkpoint paths differ")

    xpu_scores = {item["id"]: item["logprob"] for item in xpu["top_logprobs"]}
    prism_scores = {item["id"]: item["logprob"] for item in prism["top_logprobs"]}
    common = sorted(xpu_scores.keys() & prism_scores.keys())
    deltas = [
        {
            "id": token,
            "freetoken_logprob": xpu_scores[token],
            "prism_logprob": prism_scores[token],
            "delta": xpu_scores[token] - prism_scores[token],
        }
        for token in common
    ]
    absolute_deltas = [abs(item["delta"]) for item in deltas]
    xpu_top = xpu["top_logprobs"][0]["id"]
    prism_top = prism["top_logprobs"][0]["id"]
    result = {
        "model": xpu["model"],
        "prompt_token_count": xpu["prompt_token_count"],
        "same_prompt_ids": True,
        "freetoken_device": xpu["device"],
        "freetoken_device_id": xpu["device_id"],
        "freetoken_generated_token_ids": xpu["generated_token_ids"],
        "prism_generated_token_ids": prism["generated_token_ids"],
        "top1_ids_match": xpu_top == prism_top,
        "top1_id": xpu_top,
        "top_k": min(len(xpu_scores), len(prism_scores)),
        "top_k_overlap": len(common),
        "top_k_overlap_ratio": len(common) / min(len(xpu_scores), len(prism_scores)),
        "max_common_abs_logprob_delta": max(absolute_deltas),
        "mean_common_abs_logprob_delta": sum(absolute_deltas) / len(absolute_deltas),
        "common_token_deltas": deltas,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare next-token top-K scores from FreeToken XPU and Prism llama-server.",
        epilog=(
            "Run capture-xpu while Prism is stopped, then start Prism llama-server on "
            "the same checkpoint with --device Vulkan0, run capture-prism, and compare."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    xpu_parser = subparsers.add_parser("capture-xpu")
    xpu_parser.add_argument("model")
    xpu_parser.add_argument("--prompt", default="Say hello in one short sentence.")
    xpu_parser.add_argument("--top-k", type=int, default=20)
    xpu_parser.add_argument("--context", type=int, default=128)
    xpu_parser.add_argument("--kv-tokens", type=int, default=256)
    xpu_parser.add_argument("--output", required=True)

    prism_parser = subparsers.add_parser("capture-prism")
    prism_parser.add_argument("--xpu-json", required=True)
    prism_parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    prism_parser.add_argument("--reference-backend", default="unknown")
    prism_parser.add_argument("--output", required=True)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--xpu-json", required=True)
    compare_parser.add_argument("--prism-json", required=True)

    args = parser.parse_args()
    if args.command == "capture-xpu":
        result = capture_xpu(args)
        print(json.dumps(result, indent=2))
    elif args.command == "capture-prism":
        result = capture_prism(args)
        print(json.dumps(result, indent=2))
    else:
        compare(args)


if __name__ == "__main__":
    main()
