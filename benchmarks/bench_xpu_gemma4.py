from __future__ import annotations

import argparse
import json
import time

import torch
from freetoken.accelerator import resolve_runtime
from freetoken.core import SamplingParams
from freetoken.llm import LLM
from freetoken.mm.config import ENCODER_KINDS, MultimodalConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="FreeToken GGUF XPU smoke benchmark")
    parser.add_argument("model")
    parser.add_argument("--prompt", default="Say hello in one short sentence.")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--kv-tokens", type=int, default=512)
    args = parser.parse_args()

    started = time.perf_counter()
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
    load_seconds = time.perf_counter() - started
    capabilities = resolve_runtime("xpu").capabilities(0)
    encoded = llm.tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    input_ids = encoded["input_ids"]
    started = time.perf_counter()
    result = llm.generate(
        [input_ids],
        SamplingParams(max_tokens=args.max_tokens, temperature=0.0),
    )[0]
    generation_seconds = time.perf_counter() - started
    llm.shutdown()

    print(
        json.dumps(
            {
                "accelerator": capabilities.kind,
                "device": capabilities.name,
                "device_id": capabilities.device_id,
                "driver": capabilities.driver_version,
                "platform": capabilities.platform_name,
                "model": args.model,
                "prompt_tokens": len(input_ids),
                "output_tokens": len(result["token_ids"]),
                "token_ids": result["token_ids"],
                "text": result["text"],
                "load_seconds": load_seconds,
                "generation_seconds": generation_seconds,
                "output_tokens_per_second": (
                    len(result["token_ids"]) / generation_seconds
                    if generation_seconds
                    else None
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
