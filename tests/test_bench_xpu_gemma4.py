from types import SimpleNamespace

import pytest

from benchmarks.bench_xpu_gemma4 import _xpu_execution_capabilities


def test_xpu_benchmark_reports_capabilities_from_its_engine_runtime():
    capabilities = SimpleNamespace(device="xpu:0")
    runtime = SimpleNamespace(kind="xpu", capabilities=lambda index: capabilities)
    llm = SimpleNamespace(
        engine=SimpleNamespace(
            runtime=runtime,
            device=SimpleNamespace(type="xpu", index=0),
        )
    )

    assert _xpu_execution_capabilities(llm) is capabilities


@pytest.mark.parametrize(
    ("runtime_kind", "device_type"),
    [("cuda", "cuda"), ("xpu", "cpu")],
)
def test_xpu_benchmark_refuses_non_xpu_engine(runtime_kind, device_type):
    runtime = SimpleNamespace(kind=runtime_kind, capabilities=lambda index: None)
    llm = SimpleNamespace(
        engine=SimpleNamespace(
            runtime=runtime,
            device=SimpleNamespace(type=device_type, index=0),
        )
    )

    with pytest.raises(RuntimeError, match="did not bind an XPU device"):
        _xpu_execution_capabilities(llm)
