from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import torch
from freetoken.accelerator import resolve_runtime
from freetoken.accelerator.timing import summarize_generation_timing
from freetoken.core import SamplingParams
from freetoken.llm import LLM
from freetoken.message import DetokenizeMsg
from freetoken.mm.config import ENCODER_KINDS, MultimodalConfig


class _TimedLLM(LLM):
    def __init__(self, *args, **kwargs):
        self.token_timestamps: list[float] = []
        super().__init__(*args, **kwargs)

    def offline_send_result(self, reply: list) -> None:
        for msg in reply:
            if isinstance(msg, DetokenizeMsg) and not (
                msg.finished and msg.next_token in self.eos_token_ids
            ):
                self.token_timestamps.append(time.perf_counter())
        super().offline_send_result(reply)


def main() -> None:
    parser = argparse.ArgumentParser(description="FreeToken GGUF XPU timing benchmark")
    parser.add_argument("model")
    parser.add_argument("--prompt", default="Say hello in one short sentence.")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument(
        "--force-decode-length",
        action="store_true",
        help="Ignore EOS until max_tokens to stabilize decode-rate measurements.",
    )
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--kv-tokens", type=int, default=512)
    args = parser.parse_args()

    started = time.perf_counter()
    llm = _TimedLLM(
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
    generation_started = time.perf_counter()
    result = llm.generate(
        [input_ids],
        SamplingParams(
            max_tokens=args.max_tokens,
            temperature=0.0,
            ignore_eos=args.force_decode_length,
        ),
    )[0]
    generation_finished = time.perf_counter()
    timing = summarize_generation_timing(
        generation_started, llm.token_timestamps, generation_finished
    )
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
                "max_tokens_requested": args.max_tokens,
                "force_decode_length": args.force_decode_length,
                "token_ids": result["token_ids"],
                "text": result["text"],
                "load_seconds": load_seconds,
                **asdict(timing),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
