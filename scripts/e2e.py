"""Run the optional real-backend nmesh end-to-end workflow."""

from __future__ import annotations

import itertools
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import NoReturn

SUCCESS = 0
FAILURE = 1
BACKEND_UNAVAILABLE = 77

_STEP_LOGS = itertools.count()


class HarnessFailure(RuntimeError):
    pass


def _fail(step: str, message: str) -> NoReturn:
    raise HarnessFailure(f"{step} failed: {message}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _free_service_base(count: int = 1) -> int:
    for _ in range(100):
        candidate = _free_port()
        sockets: list[socket.socket] = []
        try:
            for port in range(candidate, candidate + count):
                probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                probe.bind(("127.0.0.1", port))
                sockets.append(probe)
            return candidate
        except OSError:
            continue
        finally:
            for probe in sockets:
                probe.close()
    raise RuntimeError("unable to find a contiguous range of free service ports")


def _listener(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _backend() -> Path | None:
    configured = os.environ.get("NMESH_LLAMA_SERVER") or os.environ.get("NMESH_LLAMA_CPP")
    if configured:
        candidate = Path(configured)
        return candidate.resolve() if candidate.is_file() else None
    candidates: list[Path] = []
    found = shutil.which("llama-server")
    if found:
        candidates.append(Path(found))
    candidates.extend((
        Path.home() / "llamacpp" / "llama-server.exe",
        Path.home() / "llamacpp" / "llama-server",
    ))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _model_source(model_id: str | None = None, quant: str | None = None) -> Path | None:
    configured = os.environ.get("NMESH_E2E_MODEL")
    if configured and Path(configured).is_file():
        return Path(configured).resolve()
    model_dir = Path.home() / ".nmesh" / "models"
    if not model_dir.is_dir():
        return None
    models = sorted(model_dir.glob("*.gguf"))
    if model_id:
        models = [item for item in models if model_id in item.name]
    if quant:
        aliases = {"f16": ("f16", "fp16"), "q4_k_m": ("q4_k_m",)}
        matches = [item for item in models if any(alias in item.name for alias in aliases.get(quant, (quant,)))]
        if matches:
            models = matches
    preferred = [
        item for item in models
        if "qwen2.5-1.5b-instruct-q4_k_m" in item.name
    ]
    return (preferred or models)[0] if preferred or models else None


def _command(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    # Detached backend processes inherit the child's output handles, so a pipe
    # would stay open until they exit; a per-step file returns immediately.
    log = Path(env["NMESH_HOME"]) / f"cli-step-{next(_STEP_LOGS)}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8", errors="replace") as sink:
        completed = subprocess.run(
            [sys.executable, "-m", "nmesh.cli", *args],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            text=True,
            stdout=sink,
            stderr=subprocess.STDOUT,
            check=False,
        )
    output = log.read_text(encoding="utf-8", errors="replace")
    return subprocess.CompletedProcess(completed.args, completed.returncode, output, "")


def _run_step(env: dict[str, str], step: str, *args: str) -> str:
    result = _command(env, *args)
    output = result.stdout
    print(f"\n== {step} ==\n{output}", end="")
    if result.returncode != 0:
        _fail(step, f"process exit status {result.returncode}\n{output}")
    return output


def _json_step(env: dict[str, str], step: str, *args: str) -> object:
    output = _run_step(env, step, *args)
    try:
        return json.loads(output)
    except json.JSONDecodeError as error:
        _fail(step, f"invalid JSON output: {error}")


def _http(
    step: str,
    url: str,
    payload: dict[str, object] | None = None,
    stream: bool = False,
) -> bytes:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = response.read()
            print(f"\n== {step} ==\nSTATUS={response.status}\n{body.decode(errors='replace')}", end="")
            if response.status != 200:
                _fail(step, f"HTTP status {response.status}")
            return body
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        _fail(step, f"HTTP status {error.code}\n{body}")
    except (OSError, TimeoutError) as error:
        _fail(step, f"request error: {error}")


def _link_model(plan: object, source: Path) -> None:
    services = plan.get("services", []) if isinstance(plan, dict) else []
    for service in services:
        if not isinstance(service, dict) or service.get("backend") != "llamacpp":
            continue
        target = Path(str(service["model_ref"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except FileExistsError:
            pass
        except OSError:
            try:
                target.symlink_to(source)
            except (OSError, FileExistsError):
                shutil.copy2(source, target)


def _assert_no_listeners(ports: list[int]) -> None:
    remaining = [str(port) for port in ports if _listener(port)]
    if remaining:
        _fail("listener assertion", f"listeners remain on ports {', '.join(remaining)}")
    print(f"\n== listener assertion ==\nNo listeners on: {', '.join(map(str, ports))}\n")


def main() -> int:
    backend = _backend()
    if backend is None:
        print(
            "SKIP: real llama.cpp backend unavailable; "
            "set NMESH_LLAMA_SERVER or install llama-server.",
            file=sys.stderr,
        )
        return BACKEND_UNAVAILABLE
    scratch = Path(tempfile.mkdtemp(prefix="nmesh-e2e-"))
    gateway_port = _free_port()
    service_base = _free_service_base()
    ports = [gateway_port, service_base]
    env = os.environ.copy()
    env["NMESH_HOME"] = str(scratch)
    env["NMESH_SERVICE_PORT_BASE"] = str(service_base)
    env["PATH"] = str(backend.parent) + os.pathsep + env.get("PATH", "")
    started = False
    asserted = False
    try:
        _json_step(env, "doctor --json", "doctor", "--json")
        plan = _json_step(env, "plan", "plan", "--roles", "chat", "--json")
        services = plan.get("services", []) if isinstance(plan, dict) else []
        planned_model = next(
            (
                item for item in services
                if isinstance(item, dict) and item.get("backend") == "llamacpp"
            ),
            None,
        )
        source = _model_source(
            str(planned_model.get("model_id")) if isinstance(planned_model, dict) else None,
            str(planned_model.get("quant")) if isinstance(planned_model, dict) else None,
        )
        if source is None:
            _fail(
                "plan",
                "real GGUF model unavailable; set NMESH_E2E_MODEL to a local "
                "model matching the planned llama.cpp model",
            )
        _link_model(plan, source)
        ports.extend(
            int(item["port"]) for item in services
            if isinstance(item, dict) and "port" in item
        )
        ports = list(dict.fromkeys(ports))
        started = True
        _run_step(env, "up", "up", "--detach", "--no-download", "--port", str(gateway_port), "--json")
        base = f"http://127.0.0.1:{gateway_port}"
        _http("GET /v1/models", f"{base}/v1/models")
        chat = {
            "model": "nmesh-auto",
            "messages": [{"role": "user", "content": "Say hello briefly."}],
            "max_tokens": 16,
        }
        body = _http("chat completion non-streaming", f"{base}/v1/chat/completions", chat)
        if not json.loads(body).get("choices"):
            _fail("chat completion non-streaming", "response has no choices")
        streaming_chat = {**chat, "stream": True}
        body = _http("chat completion streaming", f"{base}/v1/chat/completions", streaming_chat)
        if b"data:" not in body or b"[DONE]" not in body:
            _fail("chat completion streaming", "response was not a complete SSE stream")
        for label, prompt in (("string", "Say hello briefly."), ("list", ["Say ", "hello briefly."])):
            completion = {"model": "nmesh-auto", "prompt": prompt, "max_tokens": 16}
            body = _http(f"/v1/completions {label}", f"{base}/v1/completions", completion)
            if not json.loads(body).get("choices"):
                _fail(f"/v1/completions {label}", "response has no choices")
        _http("GET /metrics/prometheus", f"{base}/metrics/prometheus")
        _run_step(env, "bench", "bench", "--service", "chat", "--tokens", "16", "--json")
        _run_step(env, "status", "status", "--json")
        _run_step(env, "down", "down", "--json")
        started = False
        _assert_no_listeners(ports)
        asserted = True
        return SUCCESS
    except HarnessFailure as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return FAILURE
    finally:
        if started:
            result = _command(env, "down", "--json")
            print(f"\n== teardown down ==\n{result.stdout}", end="")
        if not asserted:
            time.sleep(0.2)
            try:
                _assert_no_listeners(ports)
            except HarnessFailure as error:
                print(f"\nERROR: {error}", file=sys.stderr)
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
