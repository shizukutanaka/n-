from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Protocol

from nmesh.catalog import ModelSpec, load_catalog
from nmesh.planner import Plan, PlannedService, build_plan, free_budgets, load_plan, save_plan
from nmesh.probe import HardwareProfile, detect_hardware

from .acquisition import acquire

STATE_PATH = Path.home() / ".nmesh" / "state.json"
HEALTH_TIMEOUT = 120.0
MAX_RESTARTS = 3
RESTART_WINDOW = 300.0
GIB = 1024**3


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
                 health_timeout: float = HEALTH_TIMEOUT,
                 probe: Callable[[], HardwareProfile] | None = None,
                 catalog: Callable[[], Sequence[ModelSpec]] | None = None):
        self.launcher = launcher or self._launch
        self.state_path = state_path
        self.health_timeout = health_timeout
        self.probe = probe or detect_hardware
        self.catalog = catalog or load_catalog
        self.processes: dict[str, ProcessLike] = {}
        self.shared_services: set[str] = set()
        self.external_shared: set[str] = set()
        self.restarts: dict[str, list[float]] = {}
        self.failed: dict[str, str] = {}
        self.active_plan: Plan | None = None
        self._lock = RLock()
        atexit.register(self.down)

    def _already_up(self, service: PlannedService) -> bool:
        if service.name in self.processes:
            return self._alive(service.name)
        if service.name in self.external_shared:
            if service.launch.health_url is not None and self._healthy(service):
                return True
            self.external_shared.discard(service.name)
        return (
            service.launch.shared_daemon
            and service.launch.health_url is not None
            and self._healthy(service)
        )

    def _alive(self, name: str) -> bool:
        process = self.processes.get(name)
        return process is not None and process.poll() is None

    def _adopt(self, service: PlannedService) -> bool:
        if service.name in self.processes or service.launch.health_url is None:
            return False
        if self._healthy(service):
            self.external_shared.add(service.name)
            return True
        self.external_shared.discard(service.name)
        return False

    def _restart_budget(self, name: str) -> bool:
        cutoff = time.monotonic() - RESTART_WINDOW
        timestamps = [stamp for stamp in self.restarts.get(name, []) if stamp >= cutoff]
        self.restarts[name] = timestamps
        return len(timestamps) < MAX_RESTARTS

    def _record_restart(self, name: str) -> None:
        self.restarts.setdefault(name, []).append(time.monotonic())

    def _admit(self, plan: Plan,
               bench_cache: Mapping[object, float] | None = None) -> Plan:
        profile = self.probe()
        vram, ram = free_budgets(profile)
        pending = [service for service in plan.services if not self._already_up(service)]
        if not pending:
            return plan
        resident = [service for service in pending if service.name not in plan.swap_group]
        swapped = [service for service in pending if service.name in plan.swap_group]
        need_gpu = sum(service.memory.gpu_bytes for service in resident) + max(
            [service.memory.gpu_bytes for service in swapped], default=0.0
        )
        need_cpu = sum(service.memory.cpu_bytes for service in resident) + max(
            [service.memory.cpu_bytes for service in swapped], default=0.0
        )
        if need_gpu <= vram + 1 and need_cpu <= ram + 1:
            return plan
        replanned = build_plan(
            profile, self.catalog(), replace(plan.policy, budget_source="free"), bench_cache
        )
        warning = (
            f"空きメモリ不足: 必要 {need_gpu / GIB:.2f}GiB VRAM / "
            f"{need_cpu / GIB:.2f}GiB RAM、利用可能 {vram / GIB:.2f}GiB VRAM / "
            f"{ram / GIB:.2f}GiB RAM に合わせて再計画しました"
        )
        if replanned.services:
            return replace(replanned, warnings=[*replanned.warnings, warning])
        return replace(
            plan,
            warnings=[
                *plan.warnings,
                warning + "。再計画でサービスを選べなかったため既存のfallbackを試みます",
            ],
        )

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
        entries: list[dict[str, object]] = [
            {"service": name, "pid": process.pid, "port": next(
                (item.port for item in plan.services if item.name == name), 0
            ), "started_at": time.time(), "shared": False,
             "parallel_slots": next(
                 (item.memory.parallel_slots for item in plan.services if item.name == name), 1
             )}
            for name, process in self.processes.items()
        ]
        entries.extend(
            {"service": name, "pid": None, "port": next(
                (item.port for item in plan.services if item.name == name), 11434
            ), "started_at": time.time(), "shared": True,
             "parallel_slots": next(
                 (item.memory.parallel_slots for item in plan.services if item.name == name), 1
             )}
            for name in self.shared_services if name not in self.processes
        )
        entries.extend(
            {"service": name, "pid": None, "port": next(
                (item.port for item in plan.services if item.name == name), 0
            ), "started_at": time.time(), "shared": True, "external": True,
             "parallel_slots": next(
                 (item.memory.parallel_slots for item in plan.services if item.name == name), 1
             )}
            for name in self.external_shared
            if name not in self.processes and name not in self.shared_services
        )
        payload = {
            "services": entries,
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _fallback(self, plan: Plan, attempt: int) -> Plan:
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
                argv[argv.index("-c") + 1] = str(context * service.memory.parallel_slots)
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

    def up(self, plan: Plan, no_download: bool = False, dry_run: bool = False,
           admit: bool = True,
           bench_cache: Mapping[object, float] | None = None) -> RuntimeStatus:
        if dry_run:
            return RuntimeStatus(False, [{
                "service": item.name,
                "backend": item.backend,
                "model_ref": item.model_ref,
                "port": item.port,
                "context": item.context,
                "parallel_slots": item.memory.parallel_slots,
                "n_gpu_layers": item.n_gpu_layers,
                "argv": item.launch.argv,
            } for item in plan.services])
        with self._lock:
            current = plan
            if admit:
                try:
                    current = self._admit(plan, bench_cache)
                except Exception as error:  # noqa: BLE001
                    current = replace(
                        plan,
                        warnings=[*plan.warnings, f"Free-memory admission skipped: {error}"],
                    )
            self.active_plan = current
            for attempt in range(1, 4):
                try:
                    for service in current.services:
                        dead = service.name in self.processes and not self._alive(service.name)
                        if dead:
                            self.processes.pop(service.name, None)
                        if service.name not in self.processes and self._adopt(service):
                            continue
                        if self._already_up(service):
                            if service.launch.shared_daemon:
                                self.shared_services.add(service.name)
                            continue
                        if dead:
                            if not self._restart_budget(service.name):
                                self.failed[service.name] = "Restart budget exhausted"
                                raise RuntimeError(
                                    f"Restart budget exhausted: {service.name}"
                                )
                            self._record_restart(service.name)
                        if not no_download:
                            acquire(service)
                        self.processes[service.name] = self.launcher(service)
                        self.failed.pop(service.name, None)
                        if not self._wait_health(service):
                            raise RuntimeError(f"Service did not become healthy: {service.name}")
                    if current is plan:
                        save_plan(current)
                    self._persist(current)
                    return self.status()
                except (OSError, RuntimeError):
                    self.down()
                    if attempt == 3:
                        raise
                    current = self._fallback(current, attempt)
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
            self.shared_services.clear()
            self.external_shared.clear()
            self.restarts.clear()
            self.failed.clear()
            self.active_plan = None
            try:
                self.state_path.unlink()
            except FileNotFoundError:
                pass
        return RuntimeStatus(False, [])

    def ensure_running(self, service_name: str, plan: Plan | None = None) -> RuntimeStatus:
        with self._lock:
            selected = plan or self.active_plan
            if selected is None:
                raise FileNotFoundError("No active plan")
            self.active_plan = selected
            target = next((item for item in selected.services if item.name == service_name), None)
            if target is None:
                raise KeyError(f"Unknown service: {service_name}")
            if service_name in selected.swap_group:
                for name in list(self.processes):
                    if name != service_name and name in selected.swap_group:
                        self._stop_process(name)
            dead = service_name in self.processes and not self._alive(service_name)
            if dead:
                self.processes.pop(service_name, None)
            if service_name not in self.processes and self._adopt(target):
                pass
            elif self._already_up(target):
                if target.launch.shared_daemon:
                    self.shared_services.add(service_name)
            else:
                if dead:
                    if not self._restart_budget(service_name):
                        self.failed[service_name] = "Restart budget exhausted"
                        self._persist(selected)
                        raise RuntimeError(f"Restart budget exhausted: {service_name}")
                    self._record_restart(service_name)
                self.processes[service_name] = self.launcher(target)
                if not self._wait_health(target):
                    self._stop_process(service_name)
                    raise RuntimeError(f"Service did not become healthy: {service_name}")
                self.failed.pop(service_name, None)
            self._persist(selected)
            return self.status()

    def _stop_process(self, service_name: str) -> None:
        process = self.processes.pop(service_name, None)
        if process is None:
            return
        if process.poll() is None:
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

    def status(self) -> RuntimeStatus:
        entries = [{
            "service": name,
            "pid": process.pid,
            "running": process.poll() is None,
            "restarts": len(self.restarts.get(name, [])),
            "parallel_slots": (
                next(
                    (item.memory.parallel_slots for item in self.active_plan.services
                     if item.name == name),
                    1,
                ) if self.active_plan is not None else 1
            ),
        } for name, process in self.processes.items()]
        entries.extend({
            "service": name,
            "pid": None,
            "running": True,
            "shared": True,
            "parallel_slots": next(
                (item.memory.parallel_slots for item in self.active_plan.services
                 if item.name == name), 1
            ) if self.active_plan is not None else 1,
        } for name in self.shared_services if name not in self.processes)
        entries.extend({
            "service": name,
            "pid": None,
            "running": True,
            "shared": True,
            "external": True,
            "parallel_slots": next(
                (item.memory.parallel_slots for item in self.active_plan.services
                 if item.name == name), 1
            ) if self.active_plan is not None else 1,
        } for name in self.external_shared if name not in self.processes)
        from_state = False
        if not entries and self.state_path.exists():
            try:
                payload = json.loads(self.state_path.read_text(encoding="utf-8"))
                entries = list(payload.get("services", []))
                from_state = True
            except (OSError, json.JSONDecodeError):
                entries = []
        names = {str(item.get("service")) for item in entries}
        entries.extend({
            "service": name,
            "pid": None,
            "running": False,
            "failed": reason,
            "restarts": len(self.restarts.get(name, [])),
            "parallel_slots": next(
                (item.memory.parallel_slots for item in self.active_plan.services
                 if item.name == name), 1
            ) if self.active_plan is not None else 1,
        } for name, reason in self.failed.items() if name not in names)
        default_running = bool(from_state)
        return RuntimeStatus(
            any(bool(item.get("running", default_running)) for item in entries), entries
        )

    def heartbeat(self) -> RuntimeStatus:
        with self._lock:
            if self.active_plan is None:
                return self.status()
            changed = False
            for service in self.active_plan.services:
                if (
                    service.name in self.active_plan.swap_group
                    and service.name not in self.processes
                    and service.name not in self.external_shared
                ):
                    continue
                if self._adopt(service):
                    self.failed.pop(service.name, None)
                    changed = True
                    continue
                if self._already_up(service):
                    continue
                if service.name in self.failed:
                    continue
                if service.name in self.processes:
                    self.processes.pop(service.name, None)
                if not self._restart_budget(service.name):
                    self.failed[service.name] = "Restart budget exhausted"
                    changed = True
                    continue
                self._record_restart(service.name)
                changed = True
                try:
                    self.processes[service.name] = self.launcher(service)
                    if not self._wait_health(service, timeout=min(self.health_timeout, 30.0)):
                        self._stop_process(service.name)
                        if not self._restart_budget(service.name):
                            self.failed[service.name] = (
                                f"Service failed health check: {service.name}"
                            )
                except Exception as error:  # noqa: BLE001
                    self.processes.pop(service.name, None)
                    if not self._restart_budget(service.name):
                        self.failed[service.name] = str(error)
            if changed:
                self._persist(self.active_plan)
            return self.status()


_default = Supervisor()


def up(plan: Plan | None = None, no_download: bool = False, dry_run: bool = False,
       admit: bool = True,
       bench_cache: Mapping[object, float] | None = None) -> RuntimeStatus:
    selected = plan or load_plan()
    if selected is None:
        raise FileNotFoundError("No plan found")
    return _default.up(
        selected, no_download=no_download, dry_run=dry_run, admit=admit,
        bench_cache=bench_cache,
    )


def down() -> RuntimeStatus:
    return _default.down()


def status() -> RuntimeStatus:
    return _default.status()


def ensure_running(service_name: str, plan: Plan | None = None) -> RuntimeStatus:
    return _default.ensure_running(service_name, plan)


def heartbeat() -> RuntimeStatus:
    return _default.heartbeat()
