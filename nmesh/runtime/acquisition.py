from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from nmesh import i18n
from nmesh.artifacts import artifact_key, load_cache, record
from nmesh.paths import nmesh_home
from nmesh.planner import PlannedService

# These are llama.cpp's nominal figures for ordering candidates, not nmesh
# measurements. Artifact bytes can deviate substantially: qwen2.5-0.5b
# q4_k_m measured at 1.64x the nominal estimate.
NOMINAL_GGUF_BPW = {
    "f32": 32.0,
    "bf16": 16.0,
    "f16": 16.0,
    "q8_0": 8.5,
    "q6_k": 6.6,
    "q5_k_m": 5.7,
    "q5_k_s": 5.5,
    "q5_1": 6.0,
    "q5_0": 5.5,
    "q4_k_m": 4.85,
    "q4_k_s": 4.6,
    "q4_1": 5.0,
    "q4_0": 4.55,
    "q3_k_l": 4.3,
    "q3_k_m": 3.9,
    "q3_k_s": 3.5,
    "q2_k": 3.35,
    "iq4_nl": 4.5,
    "iq4_xs": 4.25,
    "iq3_m": 3.66,
    "iq3_s": 3.44,
    "iq3_xs": 3.3,
    "iq3_xxs": 3.06,
    "iq2_m": 2.7,
    "iq2_s": 2.5,
    "iq2_xs": 2.31,
    "iq2_xxs": 2.06,
    "iq1_m": 1.75,
    "iq1_s": 1.56,
    "tq1_0": 1.69,
    "tq2_0": 2.06,
    "q2_k_l": 3.6,
    "q3_k_xl": 4.5,
    "q4_k_l": 5.1,
    "q4_k_xl": 5.2,
    "q5_k_l": 5.9,
    "q6_k_l": 6.8,
    "q8_0_l": 8.7,
    "q4_0_4_4": 4.55,
    "q4_0_4_8": 4.55,
    "q4_0_8_8": 4.55,
    "mxfp4": 4.25,
}
_BARE_QUANT_TOKENS = ("q8", "q4", "q5", "q6", "q3", "q2")
_UNRANKED_QUANT_TOKENS = ("q3_k", "q4_k", "q5_k")
_REPACK_LABELS = frozenset({"q4_0_4_4", "q4_0_4_8", "q4_0_8_8"})
_LABEL_ALIASES = {"fp16": "f16", "fp32": "f32"}
_LABEL_TOKENS = tuple(
    sorted(
        (
            *NOMINAL_GGUF_BPW,
            *_BARE_QUANT_TOKENS,
            *_UNRANKED_QUANT_TOKENS,
            "fp16",
            "fp32",
        ),
        key=len,
        reverse=True,
    )
)
_LABEL_RE = re.compile(
    rf"(?<![a-z0-9])(?:{'|'.join(map(re.escape, _LABEL_TOKENS))})(?![a-z0-9])",
    re.IGNORECASE,
)
_SPLIT_RE = re.compile(
    r"^(?P<prefix>.+?)[-_.](?P<part>\d{5})-of-(?P<total>\d{5})\.gguf$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Acquired:
    path: Path | None
    quant: str | None
    substituted: bool
    model_ref: str | None = None
    warning: str | None = None
    artifact_bytes: int | None = None


def parse_label(filename: str) -> str | None:
    stem = Path(filename).name
    stem = re.sub(r"\.gguf$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"[-_.]\d{5}-of-\d{5}$", "", stem, flags=re.IGNORECASE)
    labels: list[str] = []
    for match in _LABEL_RE.finditer(stem):
        token = match.group(0).lower()
        label = _LABEL_ALIASES.get(token, token)
        if label not in labels:
            labels.append(label)
    return "+".join(labels) if labels else None


def _gguf_files(repo_id: str) -> dict[str, int | None]:
    from huggingface_hub import HfApi

    result: dict[str, int | None] = {}
    info = HfApi().model_info(repo_id, files_metadata=True)
    for sibling in info.siblings or ():
        name = getattr(sibling, "rfilename", None)
        if not isinstance(name, str) or not name.lower().endswith(".gguf"):
            continue
        size = getattr(sibling, "size", None)
        if size is None:
            lfs = getattr(sibling, "lfs", None)
            size = getattr(lfs, "size", None) if lfs is not None else None
        result[name] = int(size) if size is not None else None
    return result


def _split_gguf_parts(target: Path) -> list[Path]:
    """Every numbered part of a split GGUF — or just *target* for a
    single-file model. An interrupted download leaves only the first parts
    on disk; treating part 1 alone as complete would skip the resume and
    leave llama-server unable to open the model."""
    match = _SPLIT_RE.match(target.name)
    if match is None:
        return [target]
    total = int(match.group("total"))
    head = target.name[: match.start("part")]
    return [
        target.parent / f"{head}{part:05d}-of-{total:05d}.gguf"
        for part in range(1, total + 1)
    ]


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


def _resolve_gguf(repo_id: str, quant: str) -> tuple[str, list[str], int]:
    first_label = quant.split("+", 1)[0]
    planned_bpw = NOMINAL_GGUF_BPW.get(first_label)
    if planned_bpw is None:
        raise RuntimeError(f"Unsupported planned quantization: {quant}")
    metadata = _gguf_files(repo_id)
    sizes = metadata
    files = list(metadata)
    found_labels = sorted({
        label for filename in files
        if (label := parse_label(filename)) is not None
    })
    candidates: list[tuple[str, list[str], int]] = []
    seen: set[tuple[str, ...]] = set()
    for filename in files:
        label = parse_label(filename)
        if label is None or label in _REPACK_LABELS:
            continue
        selected = _split_files(files, filename)
        if selected is None or tuple(selected) in seen:
            continue
        seen.add(tuple(selected))
        total = sum(sizes.get(part) or 0 for part in selected)
        candidates.append((label, selected, total))

    excluded_repack_labels = sorted(
        label for label in found_labels if label in _REPACK_LABELS
    )
    exact = [candidate for candidate in candidates if candidate[0] == quant]
    if exact:
        return min(exact, key=lambda candidate: candidate[2])

    eligible = [
        candidate for candidate in candidates
        if (
            NOMINAL_GGUF_BPW.get(candidate[0].split("+", 1)[0]) is not None
            and NOMINAL_GGUF_BPW[candidate[0].split("+", 1)[0]] <= planned_bpw
        )
    ]
    if not eligible:
        published = ", ".join(found_labels) or "none"
        repacks = ", ".join(excluded_repack_labels) or "none"
        raise RuntimeError(
            f"No GGUF at or below planned quant {quant} in {repo_id}; "
            f"published labels: {published}; excluded repacks: {repacks}"
        )
    eligible.sort(key=lambda candidate: (
        -NOMINAL_GGUF_BPW[candidate[0].split("+", 1)[0]],
        "+" in candidate[0],
        candidate[2],
        candidate[0],
    ))
    return eligible[0]


def _recorded_artifact_bytes(
    service: PlannedService, actual_label: str | None
) -> int | None:
    """Bytes recorded by a prior successful download for this artifact.

    Returns None when nothing was recorded (e.g. a hand-placed file) — in
    that case there is no reference to validate the cache against.
    """
    repo_id = service.download_repo
    if repo_id is None:
        return None
    return load_cache().get(artifact_key(repo_id, actual_label or service.quant))


def _artifact_warning(
    service: PlannedService,
    label: str | None,
    filename: str,
    actual_bytes: int | None,
) -> str | None:
    warnings: list[str] = []
    if label is not None and "+" in label:
        warnings.append(
            i18n.t(
                "warn.gguf_mixed_precision",
                i18n.lang(),
                service=service.name,
                planned=service.quant,
                label=label,
                filename=filename,
            )
        )
    estimate = service.memory.weight_bytes
    if actual_bytes is not None and estimate > 0 and (
        abs(actual_bytes - estimate) > estimate * 0.10
    ):
        warnings.append(
            i18n.t(
                "warn.gguf_size_mismatch",
                i18n.lang(),
                service=service.name,
                actual=actual_bytes,
                estimated=round(estimate),
            )
        )
    return " ".join(warnings) or None


def acquire(service: PlannedService) -> Acquired:
    if service.backend == "ollama":
        subprocess.run(["ollama", "pull", service.model_ref], check=True)
        name = f"nmesh-{service.model_id}-c{service.context}"
        modelfile = nmesh_home() / "ollama" / f"{name}.Modelfile"
        modelfile.parent.mkdir(parents=True, exist_ok=True)
        modelfile.write_text(
            f"FROM {service.model_ref}\n"
            f"PARAMETER num_ctx {service.context}\n",
            encoding="utf-8",
        )
        try:
            subprocess.run(
                ["ollama", "create", name, "-f", str(modelfile)],
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return Acquired(
                None,
                None,
                False,
                model_ref=service.model_ref,
                warning=i18n.t(
                    "warn.ollama_context_default",
                    i18n.lang(),
                    service=service.name,
                    context=service.context,
                ),
            )
        return Acquired(None, None, False, model_ref=name)
    if service.backend in {"vllm", "mlx"}:
        from huggingface_hub import snapshot_download

        return Acquired(
            Path(snapshot_download(repo_id=service.download_repo or service.model_ref)),
            None,
            False,
        )
    if service.backend == "llamacpp":
        target = Path(service.model_ref)
        parts = _split_gguf_parts(target)
        corrupt_note: str | None = None
        if all(part.exists() for part in parts):
            artifact_bytes = sum(part.stat().st_size for part in parts)
            actual_label = parse_label(target.name)
            expected = _recorded_artifact_bytes(service, actual_label)
            if expected is not None and artifact_bytes != expected:
                # Bytes differ from what a successful acquisition recorded —
                # the cached artifact is corrupt/truncated; re-acquire it
                # instead of handing llama-server a file it will crash on.
                corrupt_note = i18n.t(
                    "warn.gguf_corrupt",
                    i18n.lang(),
                    service=service.name,
                    filename=target.name,
                    actual=artifact_bytes,
                    expected=expected,
                )
                for part in parts:
                    part.unlink(missing_ok=True)
            else:
                quant = actual_label or service.quant
                warning = _artifact_warning(
                    service, actual_label, target.name, artifact_bytes
                )
                return Acquired(
                    target, quant, quant != service.quant, warning=warning,
                    artifact_bytes=artifact_bytes,
                )
        repo_id = service.download_repo
        if repo_id is None:
            raise RuntimeError("No Hugging Face GGUF repository configured")
        chosen, files, total_bytes = _resolve_gguf(repo_id, service.quant)
        from huggingface_hub import hf_hub_download

        paths = [
            Path(hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(target.parent),
            ))
            for filename in files
        ]
        warning = _artifact_warning(service, chosen, files[0], total_bytes)
        if corrupt_note is not None:
            warning = f"{corrupt_note} {warning}" if warning else corrupt_note
        try:
            record(repo_id, chosen, total_bytes)
        except OSError:
            pass
        return Acquired(
            paths[0], chosen, chosen != service.quant, warning=warning,
            artifact_bytes=total_bytes,
        )
    return Acquired(Path(service.model_ref), None, False)
