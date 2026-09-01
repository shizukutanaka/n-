from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Protocol

from nmesh.planner import Plan, PlannedService, load_plan, save_plan

from .acquisition import acquire

STATE_PATH = Path.home() / ".nmesh" / "state.json"
HEALTH_TIMEOUT = 120.0


class ProcessLike(Protocol):
    pid: int

    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...


class Launcher(Protocol):
    def __call__(self, service: PlannedService) -> ProcessLike: ...


@dataclass
class RuntimeStatus:
    running: bool
    services: list[dict[str, object]]


class Supervisor:
    def __init__(self, launcher: Launcher | None = None, state_path: Path = STATE_PATH,
                 health_timeout: float = HEALTH_TIMEOUT):
        self.launcher = launcher or self._launch
        self.state_path = state_path
        self.health_timeout = health_timeout
        self.processes: dict[str, ProcessLike] = {}
        self._lock = RLock()
        atexit.register(self.down)

    def _launch(self, service: PlannedService) -> ProcessLike:
        kwargs: dict[str, object] = {"env": {**os.environ, **service.launch.env}}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        return subprocess.Popen(service.launch.argv, **kwargs)

    def _healthy(self, service: PlannedService) -> bool:
        if service.launch.health_url is None:
            return True
        try:
            with urllib.request.urlopen(service.launch.health_url, timeout=2) as response:
                return 200 <= response.status < 500
        except (OSError, ValueError):
            return False

    def _wait_health(self, service: PlannedService, timeout: float | None = None) -> bool:
        timeout = self.health_timeout if timeout is None else timeout
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            process = self.processes.get(service.name)
            if process is not None and process.poll() is not None:
                return False
            if self._healthy(service):
                return True
            time.sleep(0.2)
        return False

    def _persist(self, plan: Plan) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "services": [{"service": name, "pid": process.pid, "port": next(
                (item.port for item in plan.services if item.name == name), 0
            ), "started_at": time.time()} for name, process in self.processes.items()],
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _fallback(self, plan: Plan, attempt: int) -> Plan:
        from dataclasses import replace

        quant_order = ("f16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q4_0", "q3_k_m", "q2_k")
        services: list[PlannedService] = []
        for service in plan.services:
            quant = service.quant
            context = service.context
            layers = service.n_gpu_layers
            if attempt == 1 and quant in quant_order:
                quant = quant_order[min(quant_order.index(quant) + 1, len(quant_order) - 1)]
            elif attempt == 2:
                context = max(context // 2, 128)
            elif attempt == 3 and layers is not None:
                layers = max(layers - max(1, layers // 4), 0)
            argv = list(service.launch.argv)
            if "--max-model-len" in argv:
                argv[argv.index("--max-model-len") + 1] = str(context)
            if "-c" in argv:
                argv[argv.index("-c") + 1] = str(context)
            if "-ngl" in argv and layers is not None:
                argv[argv.index("-ngl") + 1] = str(layers)
            model_ref = service.model_ref
            if service.backend == "llamacpp":
                model_ref = model_ref.replace(f"-{service.quant}.gguf", f"-{quant}.gguf")
                if "-m" in argv:
                    argv[argv.index("-m") + 1] = model_ref
            launch = replace(service.launch, argv=argv)
            services.append(replace(service, quant=quant, context=context,
                                    model_ref=model_ref, n_gpu_layers=layers, launch=launch))
        return replace(plan, services=services,
                       warnings=[*plan.warnings, f"Runtime fallback attempt {attempt}"])

    def up(self, plan: Plan, no_download: bool = False, dry_run: bool = False) -> RuntimeStatus:
        if dry_run:
            return RuntimeStatus(False, [{"service": item.name, "argv": item.launch.argv}
                                         for item in plan.services])
        with self._lock:
            current = plan
            for attempt in range(1, 4):
                try:
                    for service in current.services:
                        if service.launch.shared_daemon and self.processes:
                            continue
                        if not no_download:
                            acquire(service)
                        self.processes[service.name] = self.launcher(service)
                        if not self._wait_health(service):
                            raise RuntimeError(f"Service did not become healthy: {service.name}")
                    save_plan(current)
                    self._persist(current)
                    return self.status()
                except (OSError, RuntimeError):
                    self.down()
                    if attempt == 3:
                        raise
                    current = self._fallback(plan, attempt)
            raise RuntimeError("Runtime startup failed")

    def down(self) -> RuntimeStatus:
        with self._lock:
            for process in list(self.processes.values()):
                if process.poll() is not None:
                    continue
                if os.name == "nt":
                    process.terminate()
                else:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except OSError:
                        process.terminate()
                try:
                    process.wait(timeout=10)
                except (subprocess.TimeoutExpired, TimeoutError):
                    if os.name != "nt":
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except OSError:
                            process.kill()
                    else:
                        process.kill()
            self.processes.clear()
            try:
                self.state_path.unlink()
            except FileNotFoundError:
                pass
        return RuntimeStatus(False, [])

    def status(self) -> RuntimeStatus:
        entries = [{"service": name, "pid": process.pid, "running": process.poll() is None}
                   for name, process in self.processes.items()]
        if not entries and self.state_path.exists():
            try:
                payload = json.loads(self.state_path.read_text(encoding="utf-8"))
                entries = list(payload.get("services", []))
            except (OSError, json.JSONDecodeError):
                entries = []
        return RuntimeStatus(any(bool(item.get("running", True)) for item in entries), entries)


_default = Supervisor()


def up(plan: Plan | None = None, no_download: bool = False, dry_run: bool = False) -> RuntimeStatus:
    selected = plan or load_plan()
    if selected is None:
        raise FileNotFoundError("No plan found")
    return _default.up(selected, no_download=no_download, dry_run=dry_run)


def down() -> RuntimeStatus:
    return _default.down()


def status() -> RuntimeStatus:
    return _default.status()
