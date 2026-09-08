"""Cheap fingerprints for model artifacts loaded by inference services."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import httpx


@dataclass(frozen=True)
class GgufInfo:
    size: int
    tensors: int
    elements: int
    tensor_types: tuple[tuple[int, int], ...]
    arch: str
    name: str
    file_type: int | None
    digest: str


class _HeaderReader:
    def __init__(self, handle: BinaryIO, size: int) -> None:
        self.handle = handle
        self.size = size
        self.offset = 0
        self.header = bytearray()

    def read(self, size: int) -> bytes:
        if size < 0 or size > self.size - self.offset:
            raise ValueError("truncated GGUF header")
        chunk = self.handle.read(size)
        if len(chunk) != size:
            raise ValueError("truncated GGUF header")
        self.offset += size
        self.header.extend(chunk)
        return chunk

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        length = self.u64()
        return self.read(length).decode("utf-8", errors="replace")

    def value(self, kind: int) -> object:
        formats = {
            0: "<B",
            1: "<b",
            2: "<H",
            3: "<h",
            4: "<I",
            5: "<i",
            6: "<f",
            7: "<?",
            10: "<Q",
            11: "<q",
            12: "<d",
        }
        if kind == 8:
            return self.string()
        if kind == 9:
            item_kind = self.u32()
            count = self.u64()
            return [self.value(item_kind) for _ in range(count)]
        if kind not in formats:
            raise ValueError("unexpected GGUF type tag")
        format_string = formats[kind]
        return struct.unpack(format_string, self.read(struct.calcsize(format_string)))[0]

    def skip_value(self, kind: int) -> None:
        sizes = {
            0: 1,
            1: 1,
            2: 2,
            3: 2,
            4: 4,
            5: 4,
            6: 4,
            7: 1,
            10: 8,
            11: 8,
            12: 8,
        }
        if kind == 8:
            self.read(self.u64())
        elif kind == 9:
            item_kind = self.u32()
            count = self.u64()
            for _ in range(count):
                self.skip_value(item_kind)
        elif kind in sizes:
            self.read(sizes[kind])
        else:
            raise ValueError("unexpected GGUF type tag")


def gguf_info(path: Path) -> GgufInfo | None:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            reader = _HeaderReader(handle, size)
            if reader.read(4) != b"GGUF":
                return None
            reader.u32()
            n_tensors = reader.u64()
            n_kv = reader.u64()
            arch = ""
            name = ""
            file_type: int | None = None
            for _ in range(n_kv):
                key = reader.string()
                kind = reader.u32()
                if key in {"general.architecture", "general.name", "general.file_type"}:
                    value = reader.value(kind)
                    if key == "general.architecture" and isinstance(value, str):
                        arch = value
                    elif key == "general.name" and isinstance(value, str):
                        name = value
                    elif key == "general.file_type" and isinstance(value, int):
                        file_type = value
                else:
                    reader.skip_value(kind)
            tensor_types: dict[int, int] = {}
            elements = 0
            for _ in range(n_tensors):
                reader.string()
                dimensions = reader.u32()
                tensor_elements = 1
                for _ in range(dimensions):
                    tensor_elements *= reader.u64()
                elements += tensor_elements
                tensor_type = reader.u32()
                tensor_types[tensor_type] = tensor_types.get(tensor_type, 0) + 1
                reader.u64()
            digest = hashlib.sha256(reader.header).hexdigest()[:16]
            return GgufInfo(
                size=size,
                tensors=n_tensors,
                elements=elements,
                tensor_types=tuple(sorted(tensor_types.items())),
                arch=arch,
                name=name,
                file_type=file_type,
                digest=digest,
            )
    except (OSError, MemoryError, OverflowError, struct.error, UnicodeError, ValueError):
        return None


def gguf_fingerprint(path: Path) -> str | None:
    info = gguf_info(path)
    if info is None:
        return None
    return f"gguf:{info.tensors}:{info.size}:{info.digest}"


def ollama_fingerprint(
    model_ref: str, base_url: str = "http://127.0.0.1:11434",
) -> str | None:
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=5.0)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            return None
        names = {model_ref, f"{model_ref}:latest"}
        for entry in models:
            if not isinstance(entry, dict) or entry.get("name") not in names:
                continue
            digest = entry.get("digest")
            size = entry.get("size")
            if (
                not isinstance(digest, str)
                or not digest
                or isinstance(size, bool)
                or not isinstance(size, int)
            ):
                return None
            return f"ollama:{digest[:19]}:{size}"
        return None
    except (httpx.HTTPError, AttributeError, ValueError, TypeError, KeyError):
        return None


def service_fingerprint(backend: str, model_ref: str) -> str | None:
    if backend == "llamacpp":
        return gguf_fingerprint(Path(model_ref))
    if backend == "ollama":
        return ollama_fingerprint(model_ref)
    return None
