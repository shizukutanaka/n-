from __future__ import annotations

import hashlib
import html
import json
import os
import platform
import re
import shutil
import subprocess
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from nmesh.paths import nmesh_home

ATOM_URL = "https://github.com/ggml-org/llama.cpp/releases.atom"
ASSETS_URL = "https://github.com/ggml-org/llama.cpp/releases/expanded_assets/{tag}"
DOWNLOAD_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{asset}"
_BUILD_TAG = re.compile(r"^b\d+$")
_ASSET_HREF = re.compile(r"releases/download/([^/]+)/([^\"'<>?#]+)")


@dataclass(frozen=True)
class EngineAsset:
    tag: str
    asset: str
    url: str
    variant: str
    extra_assets: tuple[str, ...] = ()


@dataclass(frozen=True)
class InstalledEngine:
    backend: str
    tag: str
    variant: str
    exe: Path
    version_line: str | None
    sha256: str
    asset: str
    installed_at: float
    flags: tuple[str, ...]


def engines_dir() -> Path:
    return nmesh_home() / "engines" / "llamacpp"


def _require_https(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"engine URL must use https: {url}")
    return url


def _fetch_url(url: str) -> bytes:
    with urllib.request.urlopen(_require_https(url), timeout=30) as response:
        return response.read()


def _text(value: bytes | str) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def build_tags(
    limit: int = 10,
    *,
    fetch: Callable[[str], bytes | str] = _fetch_url,
) -> list[str]:
    if limit <= 0:
        return []
    body = _text(fetch(_require_https(ATOM_URL)))
    tags: list[str] = []
    for match in re.finditer(r"<id>\s*[^<]*?/(b\d+|v[^<\s]+)\s*</id>", body):
        tag = html.unescape(match.group(1).strip())
        if _BUILD_TAG.fullmatch(tag) and tag not in tags:
            tags.append(tag)
            if len(tags) >= limit:
                break
    return tags


def published_assets(
    tag: str,
    *,
    fetch: Callable[[str], bytes | str] = _fetch_url,
) -> list[str]:
    if not _BUILD_TAG.fullmatch(tag):
        raise ValueError(f"invalid llama.cpp build tag: {tag}")
    url = _require_https(ASSETS_URL.format(tag=tag))
    body = _text(fetch(url))
    assets = {
        urllib.parse.unquote(html.unescape(match.group(2)))
        for match in _ASSET_HREF.finditer(body)
        if match.group(1) == tag
    }
    return sorted(assets)


def _asset_variant(asset: str) -> str:
    name = asset.lower()
    if "-cpu-" in name:
        return "cpu"
    if "-cuda-" in name:
        match = re.search(r"-cuda-([0-9.]+)-", name)
        return f"cuda-{match.group(1)}" if match else "cuda"
    if "-rocm-" in name:
        match = re.search(r"-rocm-([0-9.]+)-", name)
        return f"rocm-{match.group(1)}" if match else "rocm"
    if "-sycl-" in name:
        return "sycl"
    if "-vulkan-" in name:
        return "vulkan"
    if "-opencl-" in name:
        return "opencl"
    if "-openvino-" in name:
        return "openvino"
    return "cpu"


def _asset_for(assets: list[str], pattern: str) -> str | None:
    matches = [
        asset for asset in assets if re.fullmatch(pattern, asset, re.IGNORECASE)
    ]
    matches.sort(
        key=lambda asset: (
            tuple(int(value) for value in re.findall(r"\d+", asset)),
            asset,
        )
    )
    return matches[-1] if matches else None


def _normalize_system(system: str) -> str:
    value = system.lower()
    if value in {"darwin", "macos", "osx"}:
        return "macos"
    if value in {"windows", "win32", "nt"}:
        return "windows"
    return "linux"


def _platform_cpu(system: str, machine: str) -> str:
    system = _normalize_system(system)
    machine = machine.lower()
    if system == "windows":
        return "win-cpu-arm64.zip" if "arm" in machine or "aarch" in machine else "win-cpu-x64.zip"
    if system == "macos":
        return "macos-arm64.tar.gz" if "arm" in machine or "aarch" in machine else "macos-x64.tar.gz"
    return "ubuntu-arm64.tar.gz" if "arm" in machine or "aarch" in machine else "ubuntu-x64.tar.gz"


def _asset_name(tag: str, suffix: str) -> str:
    return f"llama-{tag}-bin-{suffix}"


def select_asset(
    tag: str,
    assets: list[str],
    *,
    system: str,
    machine: str,
    accelerator: str | None,
    variant: str = "auto",
) -> tuple[EngineAsset, str | None]:
    system = _normalize_system(system)
    machine = machine.lower()
    accelerator = (accelerator or "").lower()
    published = set(assets)
    cpu_name = _asset_name(tag, _platform_cpu(system, machine))
    warning: str | None = None
    chosen: str | None = None
    requested_variant = variant.lower()
    extra: tuple[str, ...] = ()

    if system == "windows" and not ("arm" in machine or "aarch" in machine):
        if requested_variant == "auto":
            if accelerator == "nvidia":
                chosen = _asset_for(
                    assets,
                    rf"{re.escape(_asset_name(tag, 'win-cuda'))}-\d+\.\d+-x64\.zip",
                )
                if chosen is not None:
                    version = re.search(r"-cuda-(\d+\.\d+)-", chosen, re.IGNORECASE)
                    if version:
                        cudart = (
                            f"cudart-llama-bin-win-cuda-{version.group(1)}-x64.zip"
                        )
                        if cudart in published:
                            extra = (cudart,)
                else:
                    warning = "no published CUDA asset; falling back to the CPU asset"
            elif accelerator == "amd":
                chosen = _asset_for(
                    assets,
                    rf"{re.escape(_asset_name(tag, 'win-rocm'))}-\d+\.\d+-x64\.zip",
                )
            elif accelerator == "intel":
                chosen = _asset_name(tag, "win-sycl-x64.zip")
            if chosen is None:
                chosen = cpu_name
        elif requested_variant == "cpu":
            chosen = cpu_name
        elif requested_variant in {"cuda", "rocm"}:
            family = requested_variant
            chosen = _asset_for(
                assets,
                rf"{re.escape(_asset_name(tag, f'win-{family}'))}-\d+\.\d+-x64\.zip",
            )
            if chosen is not None and family == "cuda":
                version = re.search(r"-cuda-(\d+\.\d+)-", chosen, re.IGNORECASE)
                if version:
                    cudart = (
                        f"cudart-llama-bin-win-cuda-{version.group(1)}-x64.zip"
                    )
                    if cudart in published:
                        extra = (cudart,)
        elif requested_variant.startswith("cuda"):
            suffix = requested_variant.removeprefix("cuda-")
            chosen = _asset_name(tag, f"win-cuda-{suffix}-x64.zip")
            extra_name = f"cudart-llama-bin-win-cuda-{suffix}-x64.zip"
            if extra_name in published:
                extra = (extra_name,)
        elif requested_variant == "vulkan":
            chosen = _asset_name(tag, "win-vulkan-x64.zip")
        elif requested_variant.startswith("rocm"):
            suffix = requested_variant.removeprefix("rocm-")
            chosen = _asset_name(tag, f"win-rocm-{suffix}-x64.zip")
        elif requested_variant == "sycl":
            chosen = _asset_name(tag, "win-sycl-x64.zip")
        else:
            chosen = _asset_name(tag, f"win-{requested_variant}-x64.zip")
    elif system == "windows":
        chosen = cpu_name
        if requested_variant not in {"auto", "cpu"}:
            chosen = _asset_name(tag, f"{requested_variant}-arm64.zip")
    elif system == "macos":
        chosen = cpu_name
        if requested_variant not in {"auto", "cpu"}:
            chosen = _asset_name(tag, f"{requested_variant}-{machine}.tar.gz")
    else:
        arm = "arm" in machine or "aarch" in machine
        if arm:
            chosen = cpu_name if requested_variant in {"auto", "cpu"} else (
                _asset_name(tag, f"ubuntu-{requested_variant}-arm64.tar.gz")
            )
        elif requested_variant == "auto":
            if accelerator == "nvidia":
                chosen = _asset_name(tag, "ubuntu-vulkan-x64.tar.gz")
                warning = (
                    "no ubuntu-cuda asset is published; using the published "
                    "ubuntu-vulkan-x64 asset for NVIDIA"
                )
            elif accelerator == "amd":
                chosen = _asset_for(
                    assets,
                    rf"{re.escape(_asset_name(tag, 'ubuntu-rocm'))}-\d+\.\d+-x64\.tar\.gz",
                )
            elif accelerator == "intel":
                chosen = _asset_name(tag, "ubuntu-sycl-fp16-x64.tar.gz")
            else:
                chosen = cpu_name
        elif requested_variant == "cpu":
            chosen = cpu_name
        elif requested_variant in {"rocm"}:
            chosen = _asset_for(
                assets,
                rf"{re.escape(_asset_name(tag, 'ubuntu-rocm'))}-\d+\.\d+-x64\.tar\.gz",
            )
        elif requested_variant == "vulkan":
            chosen = _asset_name(tag, "ubuntu-vulkan-x64.tar.gz")
        elif requested_variant.startswith("rocm"):
            suffix = requested_variant.removeprefix("rocm-")
            chosen = _asset_name(tag, f"ubuntu-rocm-{suffix}-x64.tar.gz")
        elif requested_variant == "sycl":
            chosen = _asset_name(tag, "ubuntu-sycl-fp16-x64.tar.gz")
        else:
            chosen = _asset_name(tag, f"ubuntu-{requested_variant}-x64.tar.gz")

    if chosen not in published:
        available = ", ".join(sorted(assets)) or "none"
        if requested_variant != "auto":
            raise ValueError(
                f"variant {variant!r} is not published for {tag}; published assets: {available}"
            )
        actual = ", ".join(sorted({_asset_variant(asset) for asset in assets})) or "none"
        warning = (
            f"requested automatic asset {chosen!r} is not published; "
            f"falling back to {cpu_name!r}; published variants: {actual}"
        )
        chosen = cpu_name
        if chosen not in published:
            raise ValueError(
                f"no CPU asset is published for {tag}; published assets: {available}"
            )

    selected_variant = _asset_variant(chosen)
    return (
        EngineAsset(
            tag,
            chosen,
            _require_https(DOWNLOAD_URL.format(tag=tag, asset=chosen)),
            selected_variant,
            extra,
        ),
        warning,
    )


def _download(url: str, path: Path) -> None:
    with (
        urllib.request.urlopen(_require_https(url), timeout=120) as response,
        path.open("wb") as output,
    ):
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def _safe_member(name: str, root: Path) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"archive member escapes installation directory: {name}")
    target = (root / path).resolve()
    if target != root.resolve() and root.resolve() not in target.parents:
        raise ValueError(f"archive member escapes installation directory: {name}")
    return target


def _extract_archive(archive: Path, destination: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                target = _safe_member(member.filename, destination)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(member) as input_file, target.open("wb") as output:
                        shutil.copyfileobj(input_file, output)
        return
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as source:
            members = source.getmembers()
            for member in members:
                target = _safe_member(member.name, destination)
                if member.islnk():
                    raise ValueError(f"archive link is not allowed: {member.name}")
                if member.issym():
                    link = Path(member.linkname)
                    destination_root = destination.resolve()
                    link_target = (target.parent / link).resolve()
                    if (
                        link.is_absolute()
                        or ".." in link.parts
                        or (
                            link_target != destination_root
                            and destination_root not in link_target.parents
                        )
                    ):
                        raise ValueError(
                            f"archive link escapes installation directory: "
                            f"{member.name} -> {member.linkname}"
                        )
            for member in members:
                target = _safe_member(member.name, destination)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.issym():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists() or target.is_symlink():
                        target.unlink()
                    os.symlink(member.linkname, target)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    input_file = source.extractfile(member)
                    if input_file is None:
                        raise ValueError(f"unable to read archive member: {member.name}")
                    with input_file, target.open("wb") as output:
                        shutil.copyfileobj(input_file, output)
                    os.chmod(target, member.mode & 0o777)
        return
    raise ValueError(f"unsupported engine archive: {archive.name}")


def _manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def _from_manifest(payload: dict[str, object]) -> InstalledEngine:
    return InstalledEngine(
        str(payload.get("backend", "llamacpp")),
        str(payload["tag"]),
        str(payload["variant"]),
        Path(str(payload["exe"])),
        str(payload["version_line"]) if payload.get("version_line") else None,
        str(payload["sha256"]),
        str(payload["asset"]),
        float(payload["installed_at"]),
        tuple(str(flag) for flag in payload.get("flags", [])),
    )


def _read_manifest(path: Path) -> InstalledEngine:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"invalid engine manifest: {path}")
    return _from_manifest(payload)


def install(
    tag: str | None = None,
    variant: str = "auto",
    *,
    dest: Path | None = None,
    fetch: Callable[[str], bytes | str] = _fetch_url,
    download: Callable[[str, Path], None] = _download,
    system: str | None = None,
    machine: str | None = None,
    accelerator: str | None = None,
) -> tuple[InstalledEngine, list[str]]:
    root = Path(dest) if dest is not None else engines_dir()
    tags = build_tags(fetch=fetch) if tag is None else []
    selected_tag = tag or (tags[0] if tags else None)
    if selected_tag is None:
        raise RuntimeError("no llama.cpp build tags were published")
    assets = published_assets(selected_tag, fetch=fetch)
    if system is None:
        system = platform.system()
    if machine is None:
        machine = platform.machine()
    if accelerator is None:
        try:
            from nmesh.probe import detect_hardware

            profile = detect_hardware()
            accelerator = profile.gpus[0].vendor if profile.gpus else None
        except (OSError, RuntimeError, ValueError):
            accelerator = None
    asset, selection_warning = select_asset(
        selected_tag,
        assets,
        system=system,
        machine=machine,
        accelerator=accelerator,
        variant=variant,
    )
    target = root / selected_tag
    target.mkdir(parents=True, exist_ok=True)
    archive = target / asset.asset
    warnings = [selection_warning] if selection_warning else []
    try:
        download(asset.url, archive)
        digest = hashlib.sha256()
        with archive.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        _extract_archive(archive, target)
        for extra_name in asset.extra_assets:
            extra_path = target / extra_name
            download(
                _require_https(DOWNLOAD_URL.format(tag=selected_tag, asset=extra_name)),
                extra_path,
            )
            _extract_archive(extra_path, target)
            extra_path.unlink()
        exe_name = "llama-server.exe" if system.lower() == "windows" else "llama-server"
        executables = list(target.rglob(exe_name))
        if not executables:
            raise FileNotFoundError(f"{exe_name} was not found in {asset.asset}")
        exe = min(
            executables,
            key=lambda candidate: (len(candidate.relative_to(target).parts), str(candidate)),
        ).resolve()
        result = subprocess.run(
            [str(exe), "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        version_text = (result.stdout or "") + "\n" + (result.stderr or "")
        version_lines = [line.strip() for line in version_text.splitlines() if line.strip()]
        version_line = next(
            (line for line in version_lines if "version" in line.lower()),
            version_lines[0] if version_lines else None,
        )
        from nmesh.probe.caps import llamacpp_caps

        caps = llamacpp_caps(str(exe))
        flags = tuple(sorted(caps.flags)) if caps is not None else ()
        if caps is None:
            warnings.append("llama.cpp capabilities could not be detected; installed with no flags")
        installed = InstalledEngine(
            "llamacpp",
            selected_tag,
            asset.variant,
            exe,
            version_line,
            digest.hexdigest(),
            asset.asset,
            time.time(),
            flags,
        )
        manifest = _manifest_path(target)
        temporary = manifest.with_suffix(".json.tmp")
        manifest_payload = asdict(installed)
        manifest_payload["sha256_note"] = (
            "observed download hash; not publisher provenance verification"
        )
        temporary.write_text(
            json.dumps(manifest_payload, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(manifest)
        active = root / "active.json"
        active_tmp = active.with_suffix(".json.tmp")
        active_tmp.write_text(
            json.dumps({"tag": selected_tag, "manifest": str(manifest)}, indent=2),
            encoding="utf-8",
        )
        active_tmp.replace(active)
        archive.unlink()
        return installed, warnings
    except Exception:
        if archive.exists():
            archive.unlink()
        raise


def installed() -> list[InstalledEngine]:
    root = engines_dir()
    if not root.exists():
        return []
    result: list[InstalledEngine] = []
    for manifest in sorted(root.glob("*/manifest.json")):
        try:
            result.append(_read_manifest(manifest))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
    return result


def active() -> InstalledEngine | None:
    path = engines_dir() / "active.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        manifest = payload.get("manifest")
        if isinstance(manifest, str):
            candidate = Path(manifest)
        else:
            candidate = engines_dir() / str(payload["tag"]) / "manifest.json"
        if not candidate.exists():
            return None
        return _read_manifest(candidate)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def use(tag: str) -> InstalledEngine:
    candidate = next((item for item in installed() if item.tag == tag), None)
    if candidate is None:
        raise FileNotFoundError(f"llama.cpp engine is not installed: {tag}")
    path = engines_dir() / "active.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"tag": candidate.tag, "manifest": str(engines_dir() / tag / "manifest.json")}),
        encoding="utf-8",
    )
    temporary.replace(path)
    return candidate


def remove(tag: str) -> bool:
    target = engines_dir() / tag
    if not target.exists():
        raise FileNotFoundError(f"llama.cpp engine is not installed: {tag}")
    active_engine = active()
    was_active = active_engine is not None and active_engine.tag == tag
    shutil.rmtree(target)
    if was_active:
        try:
            (engines_dir() / "active.json").unlink()
        except FileNotFoundError:
            pass
    return was_active
