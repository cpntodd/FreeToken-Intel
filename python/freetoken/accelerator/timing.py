from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationTiming:
    output_tokens: int
    generation_seconds: float
    time_to_first_token_seconds: float | None
    decode_tokens: int
    decode_seconds: float | None
    decode_tokens_per_second: float | None
    end_to_end_output_tokens_per_second: float | None


def summarize_generation_timing(
    started_at: float,
    token_timestamps: Sequence[float],
    finished_at: float,
) -> GenerationTiming:
    """Summarize externally observed token arrival times for one generation."""
    timestamps = tuple(token_timestamps)
    if not math.isfinite(started_at) or not math.isfinite(finished_at):
        raise ValueError("generation timestamps must be finite")
    if finished_at < started_at:
        raise ValueError("generation finish precedes its start")

    previous = started_at
    for timestamp in timestamps:
        if (
            not math.isfinite(timestamp)
            or timestamp < previous
            or timestamp > finished_at
        ):
            raise ValueError(
                "token timestamps must be finite and ordered within generation"
            )
        previous = timestamp

    generation_seconds = finished_at - started_at
    output_tokens = len(timestamps)
    decode_tokens = max(output_tokens - 1, 0)
    decode_seconds = timestamps[-1] - timestamps[0] if decode_tokens else None
    return GenerationTiming(
        output_tokens=output_tokens,
        generation_seconds=generation_seconds,
        time_to_first_token_seconds=(
            timestamps[0] - started_at if output_tokens else None
        ),
        decode_tokens=decode_tokens,
        decode_seconds=decode_seconds,
        decode_tokens_per_second=(
            decode_tokens / decode_seconds
            if decode_seconds is not None and decode_seconds > 0
            else None
        ),
        end_to_end_output_tokens_per_second=(
            output_tokens / generation_seconds
            if output_tokens and generation_seconds > 0
            else None
        ),
    )
