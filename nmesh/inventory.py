"""Inventory local GGUF artifacts across common model stores."""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from nmesh.artifact import GgufInfo, gguf_info
from nmesh.paths import nmesh_home
from nmesh.runtime.acquisition import parse_label

FILE_TYPE_QUANT = {
    0: "f32",
    1: "f16",
    2: "q4_0",
    3: "q4_1",
    7: "q8_0",
    8: "q5_0",
    9: "q5_1",
    10: "q2_k",
    11: "q3_k_s",
    12: "q3_k_m",
    13: "q3_k_l",
    14: "q4_k_s",
    15: "q4_k_m",
    16: "q5_k_s",
    17: "q5_k_m",
    18: "q6_k",
    19: "iq2_xxs",
    20: "iq2_xs",
    21: "q2_k_s",
    22: "iq3_xs",
    23: "iq3_xxs",
    24: "iq1_s",
    25: "iq4_nl",
    26: "iq3_s",
    27: "iq3_m",
    28: "iq2_s",
    29: "iq2_m",
    30: "iq4_xs",
    31: "iq1_m",
    32: "bf16",
    36: "tq1_0",
    37: "tq2_0",
}


def label_mismatch(quant: str | None, label: str | None) -> bool:
    return (
        quant is not None
        and label is not None
        and quant not in label.split("+")
    )


@dataclass(frozen=True)
class Artifact:
    store: str
    path: Path
    bytes: int
    arch: str
    name: str
    tensors: int
    elements: int
    file_type: int | None
    quant: str | None
    label: str | None
    tags: tuple[str, ...]
    identity: str

    @property
    def label_mismatch(self) -> bool:
        return label_mismatch(self.quant, self.label)


@dataclass(frozen=True)
class DuplicateGroup:
    identity: str
    artifacts: tuple[Artifact, ...]
    reclaimable_bytes: int

    @property
    def members(self) -> tuple[Artifact, ...]:
        return self.artifacts


@dataclass(frozen=True)
class VariantGroup:
    arch: str
    quant: str | None
    name: str
    artifacts: tuple[Artifact, ...]

    @property
    def members(self) -> tuple[Artifact, ...]:
        return self.artifacts


def default_stores() -> dict[str, Path]:
    stores: dict[str, Path] = {}

    nmesh_models = nmesh_home() / "models"
    if nmesh_models.is_dir():
        stores["nmesh"] = nmesh_models

    ollama_root = Path(
        os.environ.get("OLLAMA_MODELS", str(Path.home() / ".ollama" / "models"))
    ).expanduser()
    ollama_blobs = ollama_root / "blobs"
    if ollama_blobs.is_dir():
        stores["ollama"] = ollama_blobs

    for candidate in (
        Path.home() / ".lmstudio" / "models",
        Path.home() / ".cache" / "lm-studio" / "models",
    ):
        if candidate.is_dir():
            stores["lmstudio"] = candidate
            break

    hf_candidates = []
    if (value := os.environ.get("HUGGINGFACE_HUB_CACHE")):
        hf_candidates.append(Path(value).expanduser())
    if (value := os.environ.get("HF_HOME")):
        hf_candidates.append(Path(value).expanduser() / "hub")
    hf_candidates.append(Path.home() / ".cache" / "huggingface" / "hub")
    for candidate in hf_candidates:
        if candidate.is_dir():
            stores["hf"] = candidate
            break

    roots = os.environ.get("NMESH_MODEL_ROOTS", "")
    for index, raw_root in enumerate(roots.split(os.pathsep)):
        if not raw_root:
            continue
        root = Path(raw_root).expanduser()
        if root.is_dir():
            stores[f"extra:{index}"] = root
    return stores


def _manifest_layers(manifest: Path) -> tuple[Path, list[object]] | None:
    try:
        if not manifest.is_file():
            return None
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    layers = payload.get("layers") if isinstance(payload, dict) else None
    if not isinstance(layers, list):
        return None
    return manifest, layers


def ollama_tags(models_root: Path) -> dict[str, tuple[str, ...]]:
    result: dict[str, set[str]] = {}
    manifests = models_root / "manifests"
    if not manifests.is_dir():
        return {}
    # One read+parse per manifest — parallelize; blob stores can hold many.
    with ThreadPoolExecutor(max_workers=8) as pool:
        parsed = pool.map(_manifest_layers, manifests.rglob("*"))
    for entry in parsed:
        if entry is None:
            continue
        manifest, layers = entry
        relative = manifest.relative_to(manifests).parts
        if relative and relative[0] == "registry.ollama.ai":
            relative = relative[1:]
        if relative and relative[0] == "library":
            relative = relative[1:]
        if len(relative) < 2:
            continue
        ref = "/".join(relative[:-1]) + ":" + relative[-1]
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            if layer.get("mediaType") != "application/vnd.ollama.image.model":
                continue
            digest = layer.get("digest")
            if not isinstance(digest, str) or not digest.startswith("sha256:"):
                continue
            blob = "sha256-" + digest[7:]
            result.setdefault(blob, set()).add(ref)
    return {blob: tuple(sorted(refs)) for blob, refs in result.items()}


def _identity(info: GgufInfo) -> str:
    histogram = ",".join(f"{kind}:{count}" for kind, count in info.tensor_types)
    return f"{info.arch}|{info.elements}|{info.tensors}|{info.file_type}|{histogram}"


def _safe_gguf_info(path: Path) -> GgufInfo | None:
    try:
        return gguf_info(path)
    except (OSError, RuntimeError):
        return None


def scan(stores: dict[str, Path]) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for store, root in stores.items():
        if not root.is_dir():
            continue
        tags = ollama_tags(root.parent) if store == "ollama" else {}
        visited: set[str] = set()
        candidates: list[Path] = []
        for current, _directories, filenames in os.walk(root, followlinks=False):
            for filename in filenames:
                path = Path(current) / filename
                try:
                    if not path.is_file():
                        continue
                    realpath = str(path.resolve())
                    if realpath in visited:
                        continue
                    visited.add(realpath)
                    candidates.append(path)
                except OSError:
                    continue
        # Every candidate costs a GGUF header read+parse — fan out; pool.map
        # keeps candidate order so results stay deterministic.
        with ThreadPoolExecutor(max_workers=8) as pool:
            infos = pool.map(_safe_gguf_info, candidates)
        for path, info in zip(candidates, infos, strict=True):
            if info is None:
                continue
            label = parse_label(path.name)
            artifacts.append(Artifact(
                store=store,
                path=path,
                bytes=info.size,
                arch=info.arch,
                name=info.name,
                tensors=info.tensors,
                elements=info.elements,
                file_type=info.file_type,
                quant=(
                    None if info.file_type is None
                    else FILE_TYPE_QUANT.get(info.file_type)
                ),
                label=label,
                tags=tags.get(path.name, ()),
                identity=_identity(info),
            ))
    return sorted(artifacts, key=lambda artifact: (artifact.store, str(artifact.path).casefold()))


def duplicates(artifacts: list[Artifact] | tuple[Artifact, ...]) -> list[DuplicateGroup]:
    grouped: dict[str, list[Artifact]] = {}
    for artifact in artifacts:
        grouped.setdefault(artifact.identity, []).append(artifact)
    return [
        DuplicateGroup(
            identity=identity,
            artifacts=tuple(group),
            reclaimable_bytes=sum(item.bytes for item in group) - max(
                item.bytes for item in group
            ),
        )
        for identity, group in sorted(grouped.items())
        if len(group) >= 2
    ]


def _normalized_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def variants(artifacts: list[Artifact] | tuple[Artifact, ...]) -> list[VariantGroup]:
    """Group distinct artifacts with the same nominal model and quant.

    These are not duplicates: measured 1.5B Q4_K_M files differed at 338 /
    1,543,714,304 versus 339 / 1,777,088,000 tensors / elements, and PR #29/#32
    measured such tied-output differences moving pass rates.
    """
    grouped: dict[tuple[str, str | None, str], list[Artifact]] = {}
    for artifact in artifacts:
        key = (artifact.arch, artifact.quant, _normalized_name(artifact.name))
        grouped.setdefault(key, []).append(artifact)
    return [
        VariantGroup(
            arch=arch,
            quant=quant,
            name=name,
            artifacts=tuple(group),
        )
        for (arch, quant, name), group in sorted(grouped.items())
        if len(group) >= 2 and len({item.identity for item in group}) >= 2
    ]
