from types import SimpleNamespace

from freetoken.accelerator.runtime import supports_pinned_host_memory


def _backend(available: bool) -> SimpleNamespace:
    return SimpleNamespace(is_available=lambda: available)


def test_pinned_memory_is_enabled_for_xpu_without_cuda() -> None:
    torch_module = SimpleNamespace(cuda=_backend(False), xpu=_backend(True))

    assert supports_pinned_host_memory(torch_module)


def test_pinned_memory_is_enabled_for_cuda() -> None:
    torch_module = SimpleNamespace(cuda=_backend(True), xpu=_backend(False))

    assert supports_pinned_host_memory(torch_module)


def test_pinned_memory_is_disabled_without_an_accelerator() -> None:
    torch_module = SimpleNamespace(cuda=_backend(False), xpu=_backend(False))

    assert not supports_pinned_host_memory(torch_module)


def test_pinned_memory_is_disabled_when_backend_apis_are_missing() -> None:
    assert not supports_pinned_host_memory(SimpleNamespace())
