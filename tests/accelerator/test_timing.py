import pytest
from freetoken.accelerator.timing import summarize_generation_timing


def test_summarize_generation_timing_separates_ttft_and_decode() -> None:
    timing = summarize_generation_timing(1.0, [1.5, 2.0, 2.5], 3.0)

    assert timing.output_tokens == 3
    assert timing.time_to_first_token_seconds == pytest.approx(0.5)
    assert timing.decode_tokens == 2
    assert timing.decode_seconds == pytest.approx(1.0)
    assert timing.decode_tokens_per_second == pytest.approx(2.0)
    assert timing.end_to_end_output_tokens_per_second == pytest.approx(1.5)


def test_decode_rate_is_unavailable_for_zero_or_one_output_token() -> None:
    no_output = summarize_generation_timing(1.0, [], 2.0)
    one_output = summarize_generation_timing(1.0, [1.25], 2.0)

    assert no_output.time_to_first_token_seconds is None
    assert no_output.end_to_end_output_tokens_per_second is None
    assert one_output.time_to_first_token_seconds == pytest.approx(0.25)
    assert one_output.decode_tokens == 0
    assert one_output.decode_seconds is None
    assert one_output.decode_tokens_per_second is None


def test_invalid_token_timestamps_are_rejected() -> None:
    with pytest.raises(ValueError, match="ordered"):
        summarize_generation_timing(1.0, [1.5, 1.25], 2.0)

    with pytest.raises(ValueError, match="ordered"):
        summarize_generation_timing(1.0, [2.1], 2.0)
