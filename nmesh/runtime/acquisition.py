from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from nmesh.planner import BPW, PlannedService

QUANT_ALIASES = {
    "f16": ("f16", "fp16"),
    "q8_0": ("q8_0", "q8"),
    "q6_k": ("q6_k",),
    "q5_k_m": ("q5_k_m",),
    "q4_k_m": ("q4_k_m",),
    "q4_0": ("q4_0",),
    "q3_k_m": ("q3_k_m",),
    "q2_k": ("q2_k",),
}
_SPLIT_RE = re.compile(
    r"^(?P<prefix>.+?)[-_.](?P<part>\d{5})-of-(?P<total>\d{5})\.gguf$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Acquired:
    path: Path | None
    quant: str | None
    substituted: bool


def _gguf_files(repo_id: str) -> list[str]:
    from huggingface_hub import HfApi

    return [
        name for name in HfApi().list_repo_files(repo_id)
        if name.lower().endswith(".gguf")
    ]


def _matches_quant(filename: str, quant: str) -> bool:
    stem = Path(filename).name
    return any(
        re.search(
            rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
            stem,
            re.IGNORECASE,
        )
        for alias in QUANT_ALIASES[quant]
    )


def _split_files(files: list[str], selected: str) -> list[str] | None:
    match = _SPLIT_RE.match(Path(selected).name)
    if match is None:
        return [selected]
    prefix = match.group("prefix").lower()
    total = int(match.group("total"))
    siblings: dict[int, str] = {}
    for filename in files:
        item = _SPLIT_RE.match(Path(filename).name)
        if (
            item is not None
            and item.group("prefix").lower() == prefix
            and int(item.group("total")) == total
        ):
            siblings[int(item.group("part"))] = filename
    if set(siblings) != set(range(1, total + 1)):
        return None
    return [siblings[index] for index in range(1, total + 1)]


def _files_for_quant(files: list[str], quant: str) -> list[str] | None:
    matches = sorted(filename for filename in files if _matches_quant(filename, quant))
    if not matches:
        return None
    for filename in matches:
        selected = _split_files(files, filename)
        if selected is not None:
            return selected
    return None


def _resolve_gguf(repo_id: str, quant: str) -> tuple[str, list[str]]:
    if quant not in BPW:
        raise RuntimeError(f"Unsupported planned quantization: {quant}")
    files = _gguf_files(repo_id)
    available = [
        candidate for candidate in BPW
        if _files_for_quant(files, candidate) is not None
    ]
    planned_bpw = BPW[quant]
    eligible = [candidate for candidate in available if BPW[candidate] <= planned_bpw]
    if not eligible:
        published = ", ".join(available) or "none"
        raise RuntimeError(
            f"No GGUF at or below planned quant {quant} in {repo_id}; "
            f"published quants: {published}"
        )
    chosen = max(eligible, key=BPW.__getitem__)
    resolved = _files_for_quant(files, chosen)
    if resolved is None:
        raise RuntimeError(f"Unable to resolve GGUF files for {chosen} in {repo_id}")
    return chosen, resolved


def acquire(service: PlannedService) -> Acquired:
    if service.backend == "ollama":
        subprocess.run(["ollama", "pull", service.model_ref], check=True)
        return Acquired(None, None, False)
    if service.backend in {"vllm", "mlx"}:
        from huggingface_hub import snapshot_download

        return Acquired(
            Path(snapshot_download(repo_id=service.download_repo or service.model_ref)),
            None,
            False,
        )
    if service.backend == "llamacpp":
        target = Path(service.model_ref)
        if target.exists():
            return Acquired(target, service.quant, False)
        repo_id = service.download_repo
        if repo_id is None:
            raise RuntimeError("No Hugging Face GGUF repository configured")
        chosen, files = _resolve_gguf(repo_id, service.quant)
        from huggingface_hub import hf_hub_download

        paths = [
            Path(hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(target.parent),
            ))
            for filename in files
        ]
        return Acquired(paths[0], chosen, chosen != service.quant)
    return Acquired(Path(service.model_ref), None, False)
