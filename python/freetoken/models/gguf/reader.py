"""Shared GGUF access helpers: detection, metadata, and tensor enumeration.

Thin layer over ``gguf.GGUFReader`` (gguf-py). Metadata is read into a plain dict
keyed by the GGUF field name (``general.architecture``, ``gemma4.block_count`` ...);
tensors are exposed as ``GgufTensor`` records carrying the *torch* shape (ggml dims
reversed), the ggml quant type, and a zero-copy ``uint8`` view of the packed block
bytes laid out as ``[rows, row_bytes]`` (rows = product of all but the fastest ggml
dim; row_bytes spans whole quant blocks of the fastest dim).
"""

from __future__ import annotations

import functools
import os
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def is_gguf_path(model_path: str) -> bool:
    """A single ``.gguf`` file (the only GGUF layout FreeToken loads directly)."""
    return (
        isinstance(model_path, str)
        and os.path.isfile(model_path)
        and model_path.endswith(".gguf")
    )


# Canonical name of the metadata-only GGUF that ``convert_checkpoint`` drops into an FTW
# dir built from a bare ``.gguf`` source. A GGUF carries its config AND tokenizer in the
# file's KV section, not sibling files, so a converted checkpoint has nowhere else to read
# them from -- this file is the header + KV bytes verbatim (tensor_count patched to 0, no
# tensor infos, no weight data), letting the FTW dir resolve config/tokenizer the exact
# same way the original ``.gguf`` file does.
FTW_METADATA_GGUF = "source_metadata.gguf"
# Records whether the source carried an untied "output.weight" head (the tensor table
# is stripped from metadata-only gguf files, so the fact travels as a KV).
OUTPUT_WEIGHT_PRESENT_KV = "freetoken.output_weight_present"


def gguf_config_source(model_path: str) -> str | None:
    """The ``.gguf`` file to source config/tokenizer/metadata from, or ``None``.

    A bare ``.gguf`` file resolves to itself; an FTW dir carrying a
    :data:`FTW_METADATA_GGUF` resolves to that embedded metadata file. This is the single
    seam config/tokenizer dispatch uses to decide "this checkpoint is GGUF-config-sourced"
    -- a real file and a converted-FTW dir both land on a genuine ``.gguf`` path the reader
    can parse, so no downstream code learns about the FTW wrapper.
    """
    if is_gguf_path(model_path):
        return model_path
    if isinstance(model_path, str) and os.path.isdir(model_path):
        cand = os.path.join(model_path, FTW_METADATA_GGUF)
        if os.path.isfile(cand):
            return cand
    return None


def write_metadata_gguf(source_gguf: str, dest_path: str) -> None:
    """Write a metadata-only GGUF: the source's header + KV section byte-for-byte, with
    ``tensor_count`` patched to 0 (no tensor infos, no weight data). Reading only the
    header+KV is cheap; the multi-GB tensor data is never touched.

    Validates by re-parsing: the copy must list zero tensors and expose the identical KV
    key set (the KV *bytes* are copied verbatim, so identical keys imply identical values).
    """
    import gguf

    reader = _new_reader(source_gguf)
    assert reader.tensors, f"{source_gguf}: no tensors to bound the KV section"
    # The first tensor-info record starts exactly where the KV section ends (GGUF places no
    # padding between KV and tensor infos; padding is only before the tensor *data*).
    kv_end = int(reader.tensors[0].field.offset)
    buf = bytearray(reader.data[:kv_end].tobytes())  # header + all KV, verbatim
    buf[8:16] = b"\x00" * 8  # tensor_count is a u64 at byte 8; 0 is byte-order agnostic
    # The tensor table is dropped, but config derivation needs one fact from it (an
    # untied output head shows up only as an "output.weight" tensor). Append it as an
    # extra KV and bump kv_count (u64 at byte 16). Little-endian only -- the re-parse
    # below fails loudly on a big-endian source.
    key = OUTPUT_WEIGHT_PRESENT_KV.encode()
    present = any(t.name == "output.weight" for t in reader.tensors)
    buf += struct.pack("<Q", len(key)) + key
    buf += struct.pack("<I", int(gguf.GGUFValueType.BOOL)) + bytes(
        [1 if present else 0]
    )
    struct.pack_into("<Q", buf, 16, struct.unpack_from("<Q", buf, 16)[0] + 1)
    tmp = dest_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(buf)
    os.replace(tmp, dest_path)

    check = _new_reader(dest_path)
    assert not check.tensors, "metadata gguf still lists tensors after patch"
    src_keys = {k for k in reader.fields if not k.startswith("GGUF.")}
    dst_keys = {k for k in check.fields if not k.startswith("GGUF.")}
    assert dst_keys == src_keys | {OUTPUT_WEIGHT_PRESENT_KV}, (
        f"metadata gguf KV keys differ from source: "
        f"missing {sorted(src_keys - dst_keys)}, extra {sorted(dst_keys - src_keys - {OUTPUT_WEIGHT_PRESENT_KV})}"
    )


@dataclass(frozen=True)
class GgufTensor:
    name: str
    shape: tuple[int, ...]  # torch order (ggml dims reversed)
    ggml_type: int
    rows: int  # product of shape[:-1] over the *ggml* layout = blocks-major rows
    row_bytes: int  # packed bytes per row (whole quant blocks of the fastest dim)
    _raw: np.ndarray  # uint8 view, shape [rows, row_bytes]

    def packed(self) -> torch.Tensor:
        """Zero-copy ``[rows, row_bytes]`` uint8 tensor of the native block bytes."""
        return torch.from_numpy(self._raw)


def _field_value(reader, name: str) -> Any:
    field = reader.fields.get(name)
    if field is None:
        return None
    return field.contents()


@functools.cache
def _reader(model_path: str):
    return _new_reader(model_path)


def _new_reader(model_path: str):
    import gguf
    from gguf.gguf_reader import ReaderTensor

    from .dequant import GGML_PQ2_0

    class FreeTokenGGUFReader(gguf.GGUFReader):
        def _build_tensors(self, start_offs, fields):
            # Keep gguf-py's normal enum path for standard types; intercept only PQ2_0.
            pq2_fields = [
                field for field in fields if int(field.parts[4][0]) == GGML_PQ2_0
            ]
            if not pq2_fields:
                return super()._build_tensors(start_offs, fields)

            pq2_field_ids = {id(field) for field in pq2_fields}
            standard_fields = [
                field for field in fields if id(field) not in pq2_field_ids
            ]
            super()._build_tensors(start_offs, standard_fields)
            tensors = {id(tensor.field): tensor for tensor in self.tensors}
            for field in pq2_fields:
                _name_len, name_data, _n_dims, dims, _raw_type, tensor_offset = (
                    field.parts
                )
                ggml_shape = tuple(int(dim) for dim in dims)
                numpy_shape = tuple(reversed(ggml_shape))
                n_elements = int(np.prod(dims))
                if not numpy_shape or numpy_shape[-1] % 128:
                    raise ValueError(
                        f"{bytes(name_data).decode()}: PQ2_0 row width must be a "
                        "multiple of 128"
                    )
                n_bytes = n_elements // 128 * 34
                data_offset = int(start_offs + tensor_offset[0])
                byte_shape = (
                    *numpy_shape[:-1],
                    numpy_shape[-1] // 128 * 34,
                )
                tensor = ReaderTensor(
                    name=bytes(name_data).decode(),
                    tensor_type=GGML_PQ2_0,
                    shape=dims,
                    n_elements=n_elements,
                    n_bytes=n_bytes,
                    data_offset=data_offset,
                    data=self._get(data_offset, np.uint8, n_bytes).reshape(byte_shape),
                    field=field,
                )
                tensors[id(field)] = tensor
            self.tensors = [tensors[id(field)] for field in fields]

    return FreeTokenGGUFReader(model_path)


@functools.cache
def load_gguf_metadata(model_path: str) -> dict[str, Any]:
    """All GGUF KV metadata as ``{field_name: python_value}`` (arrays -> lists)."""
    reader = _reader(model_path)
    return {name: field.contents() for name, field in reader.fields.items()}


def gguf_architecture(model_path: str) -> str:
    arch = _field_value(_reader(model_path), "general.architecture")
    if arch is None:
        raise ValueError(f"GGUF file {model_path} has no general.architecture")
    return str(arch)


def iter_gguf_tensors(model_path: str) -> Iterator[GgufTensor]:
    """Yield every tensor with its torch shape, ggml type, and packed block bytes."""
    import gguf

    from .dequant import BLOCK_SHAPE, GGML_NAME, GGML_PQ2_0

    reader = _reader(model_path)
    for t in reader.tensors:
        ne = [int(s) for s in t.shape]  # ggml order, fastest dim first
        torch_shape = tuple(reversed(ne))
        if int(t.tensor_type) == GGML_PQ2_0:
            block, type_size = BLOCK_SHAPE[GGML_PQ2_0]
        else:
            block, type_size = gguf.GGML_QUANT_SIZES[t.tensor_type]
        n_fast = ne[0]
        if n_fast % block != 0:
            raise ValueError(
                f"{t.name}: fastest dim {n_fast} not a multiple of block {block} "
                f"for {getattr(t.tensor_type, 'name', GGML_NAME.get(int(t.tensor_type), t.tensor_type))}"
            )
        row_bytes = n_fast // block * type_size
        rows = int(np.prod(ne[1:])) if len(ne) > 1 else 1
        # gguf-py returns quantized tensors as raw uint8 but F32/F16 as typed arrays;
        # normalize everything to a flat byte view before shaping into [rows, row_bytes].
        flat = np.ascontiguousarray(t.data).reshape(-1).view(np.uint8)
        raw = flat.reshape(rows, row_bytes)
        yield GgufTensor(
            name=t.name,
            shape=torch_shape,
            ggml_type=int(t.tensor_type),
            rows=rows,
            row_bytes=row_bytes,
            _raw=raw,
        )


def gguf_tensor_names(model_path: str) -> set[str]:
    return {t.name for t in _reader(model_path).tensors}


__all__ = [
    "FTW_METADATA_GGUF",
    "OUTPUT_WEIGHT_PRESENT_KV",
    "GgufTensor",
    "gguf_architecture",
    "gguf_config_source",
    "gguf_tensor_names",
    "is_gguf_path",
    "iter_gguf_tensors",
    "load_gguf_metadata",
    "write_metadata_gguf",
]
