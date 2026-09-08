from __future__ import annotations

import json
import struct
from pathlib import Path

from nmesh import cli, inventory


def _string(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def _synthetic_gguf(
    *,
    arch: str = "qwen2",
    name: str = "Qwen 2.5 1.5B",
    file_type: int = 15,
    tensors: tuple[tuple[str, tuple[int, ...], int], ...] = (
        ("token_embd.weight", (2, 3), 1),
    ),
) -> bytes:
    key_values = [
        (_string("general.architecture"), struct.pack("<I", 8) + _string(arch)),
        (_string("general.name"), struct.pack("<I", 8) + _string(name)),
        (_string("general.file_type"), struct.pack("<II", 4, file_type)),
    ]
    payload = [
        b"GGUF",
        struct.pack("<IQQ", 3, len(tensors), len(key_values)),
    ]
    for key, value in key_values:
        payload.extend((key, value))
    for tensor_name, dimensions, tensor_type in tensors:
        payload.extend((
            _string(tensor_name),
            struct.pack("<I", len(dimensions)),
            b"".join(struct.pack("<Q", dimension) for dimension in dimensions),
            struct.pack("<IQ", tensor_type, 0),
        ))
    return b"".join(payload)


def _write(
    path: Path,
    *,
    file_type: int = 15,
    tensors: tuple[tuple[str, tuple[int, ...], int], ...] = (
        ("token_embd.weight", (2, 3), 1),
    ),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_synthetic_gguf(file_type=file_type, tensors=tensors))
    return path


def test_scan_detects_extensionless_gguf(tmp_path: Path) -> None:
    path = _write(tmp_path / "sha256-abc123")
    artifacts = inventory.scan({"store": tmp_path})
    assert [artifact.path for artifact in artifacts] == [path]


def test_header_quant_wins_over_lying_filename(tmp_path: Path) -> None:
    _write(tmp_path / "x-f16.gguf")
    artifact = inventory.scan({"store": tmp_path})[0]
    assert artifact.quant == "q4_k_m"
    assert artifact.label == "f16"
    assert artifact.label_mismatch


def test_label_mismatch_accepts_a_multi_token_label(tmp_path: Path) -> None:
    path = _write(tmp_path / "x-f16+q4_k_m.gguf")
    artifact = inventory.scan({"store": tmp_path})[0]
    assert artifact.path == path
    assert artifact.label == "f16+q4_k_m"
    assert not artifact.label_mismatch


def test_duplicates_group_reclaimable_bytes_across_stores(tmp_path: Path) -> None:
    first = _write(tmp_path / "one" / "model.gguf")
    second = _write(tmp_path / "two" / "model.gguf")
    second.write_bytes(second.read_bytes() + b"larger duplicate payload")
    artifacts = inventory.scan({"one": first.parent, "two": second.parent})
    groups = inventory.duplicates(artifacts)
    assert len(groups) == 1
    assert groups[0].reclaimable_bytes == first.stat().st_size

    third = _write(tmp_path / "three" / "model.gguf")
    groups = inventory.duplicates(inventory.scan({
        "one": first.parent,
        "two": second.parent,
        "three": third.parent,
    }))
    assert groups[0].reclaimable_bytes == first.stat().st_size * 2


def test_variants_are_not_duplicates(tmp_path: Path) -> None:
    _write(tmp_path / "one" / "model.gguf", tensors=(
        ("a", (2, 3), 1),
        ("b", (2,), 1),
    ))
    _write(tmp_path / "two" / "model.gguf", tensors=(
        ("a", (2, 3), 1),
        ("b", (2, 3), 1),
    ))
    artifacts = inventory.scan({"one": tmp_path / "one", "two": tmp_path / "two"})
    assert inventory.duplicates(artifacts) == []
    groups = inventory.variants(artifacts)
    assert len(groups) == 1
    assert len(groups[0].artifacts) == 2


def test_ollama_manifest_tags_model_layer_only(tmp_path: Path) -> None:
    models = tmp_path / "models"
    blob = _write(models / "blobs" / "sha256-abc123")
    manifest = models / "manifests" / "registry.ollama.ai" / "library" / (
        "qwen2.5"
    ) / "1.5b-instruct"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "digest": "sha256:abc123",
            },
            {
                "mediaType": "application/vnd.ollama.image.template",
                "digest": "sha256:template",
            },
        ],
    }), encoding="utf-8")
    (models / "manifests" / "broken").write_text("{", encoding="utf-8")
    artifacts = inventory.scan({"ollama": models / "blobs"})
    assert artifacts[0].path == blob
    assert artifacts[0].tags == ("qwen2.5:1.5b-instruct",)


def test_unknown_file_type_is_preserved(tmp_path: Path) -> None:
    _write(tmp_path / "unknown.gguf", file_type=99)
    artifact = inventory.scan({"store": tmp_path})[0]
    assert artifact.file_type == 99
    assert artifact.quant is None


def test_invalid_files_and_missing_roots_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "plain.bin").write_bytes(b"not GGUF")
    (tmp_path / "truncated.gguf").write_bytes(b"GGUF")
    artifacts = inventory.scan({
        "valid": tmp_path,
        "missing": tmp_path / "missing",
    })
    assert artifacts == []


def test_cli_scan_and_local_report_header_metadata(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    home = tmp_path / "home"
    model = _write(home / "models" / "x-f16.gguf")
    extra = _write(tmp_path / "extra" / "other.gguf")
    monkeypatch.setenv("NMESH_HOME", str(home))
    monkeypatch.setattr(cli, "default_stores", lambda: {"nmesh": home / "models"})
    assert cli.main(["models", "scan", "--json", "--root", str(extra.parent)]) == 0
    scanned = json.loads(capsys.readouterr().out)
    assert set(scanned["stores"]) == {"nmesh", "extra:0"}
    assert scanned["totals"]["files"] == 2
    assert any(item["label_mismatch"] for item in scanned["artifacts"])

    assert cli.main(["models", "local", "--json"]) == 0
    local = json.loads(capsys.readouterr().out)
    item = next(item for item in local if item["path"] == str(model))
    assert item["quant"] == "q4_k_m"
    assert item["label"] == "f16"
    assert item["label_mismatch"]
    assert item["quant_source"] == "header"

    missing = tmp_path / "missing"
    assert cli.main(["models", "scan", "--json", "--root", str(missing)]) == 1
    assert "not an existing directory" in capsys.readouterr().err
