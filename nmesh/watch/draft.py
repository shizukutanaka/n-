"""Write bounded, explicitly incomplete catalog candidates."""

from __future__ import annotations

import re
from pathlib import Path

from .verify import Finding

_REQUIRED = (
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


def _yaml(value: object) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, list):
        return "[" + ", ".join(_yaml(item) for item in value) + "]"
    return str(value)


def _draft_text(finding: Finding) -> str:
    repo = finding.value
    verified = finding.verified
    model_id = repo.lower().replace("/", "-")
    fields: dict[str, object] = {
        "id": model_id,
        "sources": {"hf": repo},
        "quality": None,
    }
    if verified.get("weight_sets"):
        fields["sources"]["hf_gguf"] = repo
    for key in ("params", "license"):
        if key in verified:
            fields[key] = verified[key]
    if verified.get("pipeline_tag") == "text-generation":
        fields["roles"] = ["chat"]
    config_map = {
        "architectures": "family",
        "num_hidden_layers": "n_layers",
        "num_attention_heads": "n_heads",
        "num_key_value_heads": "n_kv_heads",
        "hidden_size": "hidden_size",
        "head_dim": "head_dim",
        "max_position_embeddings": "max_context",
    }
    for source, target in config_map.items():
        if source in verified:
            fields[target] = verified[source]
    missing = [field for field in _REQUIRED if field not in fields]
    if "head_dim" not in fields:
        hidden = verified.get("hidden_size")
        heads = verified.get("num_attention_heads")
        if (
            isinstance(hidden, int)
            and isinstance(heads, int)
            and heads > 0
            and hidden % heads == 0
        ):
            fields["head_dim"] = hidden // heads
            missing = [item for item in missing if item != "head_dim"]
    lines = [
        f"# Candidate from verified Hugging Face repo: {repo}",
        f"# config_repo: {verified.get('config_repo', '') or 'not found'}",
        "# quality: null is intentional; run nmesh eval before planning can rank it.",
    ]
    for key in ("pipeline_tag", "gated", "smallest_weight_bytes"):
        if key in verified:
            lines.append(f"# {key}: {_yaml(verified[key])}")
    if "head_dim" not in verified and "head_dim" in fields:
        lines.append("# head_dim_derived: true")
    if missing:
        lines.append("# incomplete; missing fields: " + ", ".join(missing))
    lines.append("-")
    for field in _REQUIRED:
        value = fields.get(field)
        if field == "quality":
            lines.append("  quality: null")
        elif field == "sources" and field in fields:
            lines.append("  sources:")
            for source, url in fields[field].items():
                lines.append(f"    {source}: {_yaml(url)}")
        elif field in fields:
            lines.append(f"  {field}: {_yaml(value)}")
        else:
            lines.append(f"  # {field}: <missing>")
    return "\n".join(lines) + "\n"


def write_draft(finding: Finding, directory: str | Path) -> Path:
    """Write one candidate file; never merge it into the active catalog."""
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", finding.value)
    target = target_dir / f"{safe}.yaml"
    target.write_text(_draft_text(finding), encoding="utf-8")
    return target


__all__ = ["write_draft"]
