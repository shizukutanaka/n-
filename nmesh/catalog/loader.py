from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from nmesh.paths import nmesh_home


@dataclass(frozen=True)
class ModelSpec:
    id: str
    family: str
    params: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_size: int
    max_context: int
    roles: list[str]
    quality: float | None
    license: str
    sources: dict[str, str]
    languages: tuple[str, ...] = ("en",)
    pooling: str = ""
    vocab_size: int = 0
    head_layout: str = "separate"
    kv_layers: int = 0
    sliding_window: int = 0
    sliding_window_pattern: int = 0
    # MoE anatomy: active_params is the per-token activated parameter count
    # (shared + routed experts actually read), moe_expert_params the params
    # inside routed-expert tensors, n_moe_layers the number of MoE layers.
    # All zero on dense models.
    active_params: int = 0
    moe_expert_params: int = 0
    n_moe_layers: int = 0


def _model_from_mapping(
    item: object, problems: list[str] | None = None
) -> ModelSpec | None:
    def reject(reason: str) -> None:
        if problems is not None:
            label = item.get("id") if isinstance(item, dict) else None
            problems.append(f"{label}: {reason}" if label else reason)

    if not isinstance(item, dict):
        reject("entry is not a mapping")
        return None
    required = (
        "id",
        "family",
        "params",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "head_dim",
        "hidden_size",
        "max_context",
        "roles",
        "quality",
        "license",
        "sources",
    )
    missing = [key for key in required if key not in item]
    if missing:
        reject(f"missing required keys: {', '.join(missing)}")
        return None
    try:
        roles_value = item["roles"]
        sources_value = item["sources"]
        if not isinstance(roles_value, list) or not isinstance(sources_value, dict):
            reject("roles must be a list and sources a mapping")
            return None
        languages_value = item.get("languages", ["en"])
        if not isinstance(languages_value, list):
            languages_value = ["en"]
        languages = tuple(
            str(value).strip().lower().split("-", 1)[0].split("_", 1)[0]
            for value in languages_value
            if str(value).strip()
        ) or ("en",)
        quality_value = item["quality"]
        quality = None if quality_value is None else float(quality_value)
        return ModelSpec(
            id=str(item["id"]),
            family=str(item["family"]),
            params=int(item["params"]),
            n_layers=int(item["n_layers"]),
            n_heads=int(item["n_heads"]),
            n_kv_heads=int(item["n_kv_heads"]),
            head_dim=int(item["head_dim"]),
            hidden_size=int(item["hidden_size"]),
            max_context=int(item["max_context"]),
            roles=[str(role) for role in roles_value],
            quality=quality,
            license=str(item["license"]),
            sources={str(key): str(value) for key, value in sources_value.items()},
            languages=languages,
            pooling=str(item.get("pooling", "")),
            vocab_size=int(item.get("vocab_size", 0)),
            head_layout=str(item.get("head_layout", "separate")),
            kv_layers=int(item.get("kv_layers", 0)),
            sliding_window=int(item.get("sliding_window", 0)),
            sliding_window_pattern=int(item.get("sliding_window_pattern", 0)),
            active_params=int(item.get("active_params", 0)),
            moe_expert_params=int(item.get("moe_expert_params", 0)),
            n_moe_layers=int(item.get("n_moe_layers", 0)),
        )
    except (TypeError, ValueError) as error:
        reject(str(error))
        return None


def _read_models(path: Path, problems: list[str] | None = None) -> list[ModelSpec]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError:
        return []
    except yaml.YAMLError as error:
        if problems is not None:
            problems.append(f"{path}: cannot parse YAML: {error}")
        return []
    if not isinstance(payload, list):
        if problems is not None:
            problems.append(f"{path}: catalog top level is not a list")
        return []
    models: list[ModelSpec] = []
    for index, item in enumerate(payload):
        entry_problems: list[str] = []
        model = _model_from_mapping(item, entry_problems)
        if model is not None:
            models.append(model)
        elif problems is not None:
            for reason in entry_problems:
                problems.append(f"{path}: entry {index}: {reason}")
    return models


def load_catalog(
    bundled_path: Path | None = None,
    user_path: Path | None = None,
    problems: list[str] | None = None,
) -> list[ModelSpec]:
    bundled = bundled_path or Path(__file__).with_name("models.yaml")
    user = user_path or (nmesh_home() / "models.yaml")
    merged: dict[str, ModelSpec] = {
        model.id: model for model in _read_models(bundled, problems)
    }
    merged.update(
        {model.id: model for model in _read_models(user, problems)}
    )
    return list(merged.values())
