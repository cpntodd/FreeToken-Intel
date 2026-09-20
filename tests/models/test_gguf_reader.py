from __future__ import annotations

import struct

import gguf
import pytest
import torch
from freetoken.models.gguf.reader import (
    gguf_tensor_names,
    iter_gguf_tensors,
    load_gguf_metadata,
    write_metadata_gguf,
)


def _write_test_gguf(path):
    tensors = [
        (
            "output.weight",
            (128, 2),
            142,
            0,
            bytes([0, 56]) + bytes([0b11_10_01_00] * 64),
        ),
        ("test.bias", (2,), 0, 128, struct.pack("<ff", 1.25, -2.5)),
    ]
    header = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), 0))
    for name, dims, ggml_type, offset, _payload in tensors:
        encoded_name = name.encode()
        header.extend(struct.pack("<Q", len(encoded_name)))
        header.extend(encoded_name)
        header.extend(struct.pack("<I", len(dims)))
        for dim in dims:
            header.extend(struct.pack("<Q", dim))
        header.extend(struct.pack("<IQ", ggml_type, offset))
    header.extend(bytes((-len(header)) % 32))

    data = bytearray()
    for _name, _dims, _ggml_type, offset, payload in tensors:
        data.extend(bytes(offset - len(data)))
        data.extend(payload)
    path.write_bytes(header + data)


def test_reader_handles_prism_pq2_0_without_changing_gguf_enum(tmp_path):
    source = tmp_path / "pq2.gguf"
    metadata_only = tmp_path / "metadata.gguf"
    _write_test_gguf(source)
    enum_before = {int(value) for value in gguf.GGMLQuantizationType}

    tensors = list(iter_gguf_tensors(str(source)))

    assert [tensor.name for tensor in tensors] == ["output.weight", "test.bias"]
    assert tensors[0].ggml_type == 142
    assert tensors[0].shape == (2, 128)
    assert (tensors[0].rows, tensors[0].row_bytes) == (2, 34)
    with pytest.warns(UserWarning, match="NumPy array is not writable"):
        packed = tensors[0].packed()
    assert packed.shape == (2, 34)
    torch.testing.assert_close(
        tensors[1].packed().view(torch.float32),
        torch.tensor([[1.25, -2.5]], dtype=torch.float32),
    )
    assert {int(value) for value in gguf.GGMLQuantizationType} == enum_before

    assert gguf_tensor_names(str(source)) == {"output.weight", "test.bias"}
    write_metadata_gguf(str(source), str(metadata_only))
    assert gguf_tensor_names(str(metadata_only)) == set()
    assert (
        load_gguf_metadata(str(metadata_only))["freetoken.output_weight_present"]
        is True
    )
