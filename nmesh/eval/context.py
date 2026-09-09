from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from nmesh.paths import nmesh_home


@dataclass(frozen=True)
class FamilyResult:
    """Per-family task-paired depth evidence.

    ``of`` counts deep tasks whose own shallow control passed in every
    control run, and ``passed`` counts those tasks that also passed deeply.
    ``control_passed`` and ``control_of`` retain pooled raw control counts for
    reporting; they do not reject the family as a whole.
    """

    name: str
    passed: int
    of: int
    control_passed: int
    control_of: int

    @property
    def attributable(self) -> bool:
        return self.of > 0

    @property
    def lost(self) -> bool:
        return self.attributable and self.passed < self.of


@dataclass(frozen=True)
class ContextRecord:
    model_id: str
    quant: str
    backend: str
    seed: str
    requested_depth: int
    served_depth: int
    probe_digest: str
    families: tuple[FamilyResult, ...]
    at: float
    artifact: str = ""
    cache_prompt: bool | None = None


def context_key(
    model_id: str,
    quant: str,
    backend: str,
    seed: str,
    requested_depth: int,
    probe_digest: str,
    cache_prompt: bool | None = None,
) -> str:
    suffix = "|c1" if cache_prompt else "|c0" if cache_prompt is not None else ""
    suffix += f"|d{requested_depth}"
    return (
        f"{model_id}|{quant}|{backend}|{seed}|{probe_digest}{suffix}"
    )


def _family(data: object) -> FamilyResult | None:
    if not isinstance(data, dict):
        return None
    try:
        name = data["name"]
        passed = data["passed"]
        of = data["of"]
        control_passed = data["control_passed"]
        control_of = data["control_of"]
        if (
            not isinstance(name, str)
            or isinstance(passed, bool)
            or not isinstance(passed, int)
            or passed < 0
            or isinstance(of, bool)
            or not isinstance(of, int)
            or of < 0
            or passed > of
            or isinstance(control_passed, bool)
            or not isinstance(control_passed, int)
            or control_passed < 0
            or isinstance(control_of, bool)
            or not isinstance(control_of, int)
            or control_of < 0
            or control_passed > control_of
        ):
            return None
        return FamilyResult(name, passed, of, control_passed, control_of)
    except (KeyError, TypeError, ValueError):
        return None


def _record(data: object) -> ContextRecord | None:
    if not isinstance(data, dict):
        return None
    try:
        model_id = data["model_id"]
        quant = data["quant"]
        backend = data["backend"]
        seed = data["seed"]
        requested_depth = data["requested_depth"]
        served_depth = data["served_depth"]
        probe_digest = data["probe_digest"]
        families_data = data["families"]
        at = data["at"]
        artifact = data.get("artifact", "")
        cache_prompt = data.get("cache_prompt")
        if (
            not isinstance(model_id, str)
            or not isinstance(quant, str)
            or not isinstance(backend, str)
            or not isinstance(seed, str)
            or isinstance(requested_depth, bool)
            or not isinstance(requested_depth, int)
            or requested_depth < 0
            or isinstance(served_depth, bool)
            or not isinstance(served_depth, int)
            or served_depth < 0
            or not isinstance(probe_digest, str)
            or not isinstance(at, (int, float))
            or isinstance(at, bool)
            or not math.isfinite(at)
            or not isinstance(artifact, str)
            or (
                cache_prompt is not None
                and not isinstance(cache_prompt, bool)
            )
            or not isinstance(families_data, (list, dict))
        ):
            return None
        family_values = (
            list(families_data)
            if isinstance(families_data, list)
            else list(families_data.values())
        )
        if not family_values:
            return None
        families = tuple(_family(value) for value in family_values)
        if any(family is None for family in families):
            return None
        return ContextRecord(
            model_id,
            quant,
            backend,
            seed,
            requested_depth,
            served_depth,
            probe_digest,
            tuple(family for family in families if family is not None),
            float(at),
            artifact,
            cache_prompt,
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_context_cache(path: Path | None = None) -> dict[str, ContextRecord]:
    target = path or (nmesh_home() / "context.json")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        results = payload.get("results", {}) if isinstance(payload, dict) else {}
        if not isinstance(results, dict):
            return {}
        return {
            str(key): record
            for key, value in results.items()
            if (record := _record(value)) is not None
        }
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def save_context(record: ContextRecord, path: Path | None = None) -> Path:
    target = path or (nmesh_home() / "context.json")
    records = load_context_cache(target)
    key = context_key(
        record.model_id,
        record.quant,
        record.backend,
        record.seed,
        record.requested_depth,
        record.probe_digest,
        record.cache_prompt,
    )
    records[key] = record
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                {"results": {key: asdict(value) for key, value in records.items()}},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return target
