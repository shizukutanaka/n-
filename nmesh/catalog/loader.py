from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


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
    quality: float
    license: str
    sources: dict[str, str]
    languages: tuple[str, ...] = ("en",)


def _model_from_mapping(item: object) -> ModelSpec | None:
    if not isinstance(item, dict):
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
    if any(key not in item for key in required):
        return None
    try:
        roles_value = item["roles"]
        sources_value = item["sources"]
        if not isinstance(roles_value, list) or not isinstance(sources_value, dict):
            return None
        languages_value = item.get("languages", ["en"])
        if not isinstance(languages_value, list):
            languages_value = ["en"]
        languages = tuple(
            str(value).strip().lower().split("-", 1)[0].split("_", 1)[0]
            for value in languages_value
            if str(value).strip()
        ) or ("en",)
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
            quality=float(item["quality"]),
            license=str(item["license"]),
            sources={str(key): str(value) for key, value in sources_value.items()},
            languages=languages,
        )
    except (TypeError, ValueError):
        return None


def _read_models(path: Path) -> list[ModelSpec]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []
    if not isinstance(payload, list):
        return []
    models: list[ModelSpec] = []
    for item in payload:
        model = _model_from_mapping(item)
        if model is not None:
            models.append(model)
    return models


def load_catalog(
    bundled_path: Path | None = None, user_path: Path | None = None
) -> list[ModelSpec]:
    bundled = bundled_path or Path(__file__).with_name("models.yaml")
    user = user_path or (Path.home() / ".nmesh" / "models.yaml")
    merged: dict[str, ModelSpec] = {model.id: model for model in _read_models(bundled)}
    merged.update({model.id: model for model in _read_models(user)})
    return list(merged.values())
