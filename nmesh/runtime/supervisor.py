from __future__ import annotations

import atexit
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import RLock
from typing import Protocol

import psutil

from nmesh import i18n
from nmesh.artifacts import load_cache as load_artifact_cache
from nmesh.catalog import ModelSpec, load_catalog
from nmesh.paths import is_windows, nmesh_home
from nmesh.planner import (
    BPW,
    Plan,
    PlannedService,
    build_plan,
    estimate_memory,
    free_budgets,
    load_plan,
    save_plan,
    split_memory,
)
from nmesh.probe import HardwareProfile, detect_hardware
from nmesh.runtime import engine

from .acquisition import Acquired, acquire
from .logs import log_path, open_log, tail

STATE_PATH = nmesh_home() / "state.json"
HEALTH_TIMEOUT = 120.0
MAX_RESTARTS = 3
RESTART_WINDOW = 300.0
GIB = 1024**3
STATE_VERSION = 2
PID_CREATE_TIME_TOLERANCE = 2.0


def _slots(plan: Plan | None, name: str) -> int:
    if plan is None:
        return 1
    return next(
        (item.memory.parallel_slots for item in plan.services if item.name == name),
        1,
    )


def _port(plan: Plan, name: str, default: int) -> int:
    return next((item.port for item in plan.services if item.name == name), default)


def _pid_alive(pid: int, create_time: float | None = None) -> bool:
    try:
        process = psutil.Process(pid)
        if not process.is_running():
            return False
        if create_time is not None:
            return abs(process.create_time() - create_time) <= PID_CREATE_TIME_TOLERANCE
        return True
    except (psutil.Error, OSError, ValueError):
        return False


class ProcessLike(Protocol):
    pid: int

    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...


def gateway_listener_pid(port: int) -> int | None:
    """Return the pid of the nmesh gateway listening on *port*, if any."""
    try:
        connections = psutil.net_connections(kind="tcp")
    except (psutil.Error, OSError):
        return None
    for connection in connections:
        if (
            connection.laddr
            and connection.laddr.port == port
            and connection.status == "LISTEN"
            and connection.pid is not None
        ):
            try:
                cmdline = psutil.Process(connection.pid).cmdline()
            except (psutil.Error, OSError):
                continue
            if any("nmesh.gateway" in part for part in cmdline):
                return connection.pid
    return None


def engine_listener_pid(port: int) -> int | None:
    """Return the pid of an nmesh-managed engine listening on *port*, if any.

    Ownership is proven by the executable living under NMESH_HOME — a foreign
    process that merely bound the port is never returned.
    """
    root = str(nmesh_home())
    try:
        connections = psutil.net_connections(kind="tcp")
    except (psutil.Error, OSError):
        return None
    for connection in connections:
        if (
            connection.laddr
            and connection.laddr.port == port
            and connection.status == "LISTEN"
            and connection.pid is not None
        ):
            try:
                exe = psutil.Process(connection.pid).exe()
            except (psutil.Error, OSError):
                continue
            if exe.startswith(root):
                return connection.pid
    return None


def _pid_serves_model(pid: int, model_ref: object) -> bool:
    """True when *pid*'s command line names *model_ref* — e.g. llama-server's
    ``-m`` argument. Used to distinguish a stale engine serving a different
    model from one actually running the planned artifact."""
    try:
        cmdline = psutil.Process(pid).cmdline()
    except (psutil.Error, OSError):
        return False
    needle = str(model_ref)
    return any(needle in part for part in cmdline)


def _port_in_use(port: int) -> bool:
    """True when *port* cannot be bound — i.e. a live listener owns it.

    Probing with connect() would consume the foreign listener's accept
    backlog and flip to False on retries; a bind attempt fails exactly the
    way the backend's own bind did.
    """
    try:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))
    except OSError:
        return True
    return False


def gateway_health(port: int) -> bool:
    """Return True when an nmesh gateway answers /health on *port*."""
    return Supervisor._health_url_alive(f"http://127.0.0.1:{port}/health")



class Launcher(Protocol):
    def __call__(self, service: PlannedService) -> ProcessLike: ...


@dataclass
class RuntimeStatus:
    running: bool
    services: list[dict[str, object]]
    warnings: list[str] = field(default_factory=list)


class Supervisor:
    def __init__(self, launcher: Launcher | None = None, state_path: Path = STATE_PATH,
                 health_timeout: float = HEALTH_TIMEOUT,
                 probe: Callable[[], HardwareProfile] | None = None,
                 catalog: Callable[[], Sequence[ModelSpec]] | None = None,
                 terminator: Callable[[int], None] | None = None,
                 plan_path: Path | None = None):
        self.launcher = launcher or self._launch
        self.state_path = state_path
        self.plan_path = plan_path
        self.health_timeout = health_timeout
        self.probe = probe or detect_hardware
        self.catalog = catalog or load_catalog
        self.processes: dict[str, ProcessLike] = {}
        self.shared_services: set[str] = set()
        self.external_shared: set[str] = set()
        self.adopted: dict[str, dict[str, object]] = {}
        self.idle: set[str] = set()
        self.restarts: dict[str, list[float]] = {}
        self.failed: dict[str, str] = {}
        self.active_plan: Plan | None = None
        self._boot_recovery = False
        self.notes: dict[str, str] = {}
        self._lock = RLock()
        self._atexit_armed = False
        self._terminator = terminator or self._terminate_pid

    def _arm_atexit(self) -> None:
        if not self._atexit_armed:
            atexit.register(self.down)
            self._atexit_armed = True

    def disarm_atexit(self) -> None:
        if self._atexit_armed:
            atexit.unregister(self.down)
            self._atexit_armed = False

    @staticmethod
    def _terminate_pid(pid: int) -> None:
        try:
            process = psutil.Process(pid)
            process.terminate()
            try:
                process.wait(timeout=10)
            except psutil.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        except (psutil.Error, OSError, ValueError):
            return

    @staticmethod
    def _create_time(pid: int) -> float | None:
        try:
            return psutil.Process(pid).create_time()
        except (psutil.Error, OSError, ValueError):
            return None

    @staticmethod
    def _health_url_alive(url: object) -> bool:
        if not isinstance(url, str) or not url:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except urllib.error.HTTPError as error:
            return error.code < 500
        except (OSError, ValueError):
            return False

    @classmethod
    def _entry_alive(cls, entry: Mapping[str, object]) -> bool:
        pid = entry.get("pid")
        if pid is not None:
            if not isinstance(pid, (int, float, str)):
                return False
            try:
                raw_created = entry.get("create_time")
                created = (
                    float(raw_created)
                    if isinstance(raw_created, (int, float, str))
                    else None
                )
                return _pid_alive(int(pid), created)
            except (TypeError, ValueError):
                return False
        if entry.get("shared") or entry.get("external"):
            return cls._health_url_alive(entry.get("health_url"))
        return False

    def _load_state(self) -> dict[str, object] | None:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _write_state(self, payload: Mapping[str, object]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_name(
            f".{self.state_path.name}.{os.getpid()}.tmp"
        )
        try:
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(temporary, self.state_path)
        except OSError:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise

    def record_gateway(self, pid: int, port: int) -> None:
        with self._lock:
            state = self._load_state() or {
                "version": STATE_VERSION,
                "owner_pid": os.getpid(),
                "services": [],
            }
            state["version"] = STATE_VERSION
            state["gateway"] = {
                "pid": pid,
                "create_time": self._create_time(pid),
                "port": port,
                "owner_pid": os.getpid(),
            }
            self._write_state(state)

    def clear_gateway(self, pid: int | None = None) -> None:
        with self._lock:
            state = self._load_state()
            if state is None:
                return
            gateway = state.get("gateway")
            if isinstance(gateway, dict) and (
                pid is None or gateway.get("pid") == pid
            ):
                state.pop("gateway", None)
                if state.get("services"):
                    self._write_state(state)
                else:
                    try:
                        self.state_path.unlink()
                    except FileNotFoundError:
                        pass

    def stop_gateway(self, foreign: bool = False) -> bool:
        with self._lock:
            state = self._load_state()
            if state is None:
                return False
            gateway = state.get("gateway")
            if not isinstance(gateway, dict):
                return False
            owner = gateway.get("owner_pid", state.get("owner_pid"))
            if not foreign and owner != os.getpid():
                return False
            pid = gateway.get("pid")
            live = False
            try:
                live = pid is not None and self._entry_alive(gateway)
            except (TypeError, ValueError):
                live = False
            if live and isinstance(pid, (int, float, str)):
                self._terminator(int(pid))
            elif foreign:
                port = gateway.get("port")
                orphan = (
                    gateway_listener_pid(port)
                    if isinstance(port, int) and not isinstance(port, bool)
                    else None
                )
                if orphan is not None:
                    self._terminator(orphan)
                    live = True
            state.pop("gateway", None)
            if state.get("services"):
                self._write_state(state)
            else:
                try:
                    self.state_path.unlink()
                except FileNotFoundError:
                    pass
            return bool(live)

    def _already_up(self, service: PlannedService) -> bool:
        if service.name in self.processes:
            return self._alive(service.name)
        adopted = self.adopted.get(service.name)
        if adopted is not None:
            if self._entry_alive(adopted) and self._healthy(service):
                return True
            self.adopted.pop(service.name, None)
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
        state = self._load_state()
        entries = state.get("services") if isinstance(state, dict) else None
        entry = next(
            (
                item for item in entries
                if isinstance(item, dict) and item.get("service") == service.name
            ),
            None,
        ) if isinstance(entries, list) else None
        pid = entry.get("pid") if isinstance(entry, dict) else None
        if (
            isinstance(pid, int)
            and not isinstance(pid, bool)
            and isinstance(entry, dict)
            and self._entry_alive(entry)
            and self._healthy(service)
        ):
            recorded_model = entry.get("model_ref")
            if recorded_model is not None and recorded_model != service.model_ref:
                # Stale process still bound to the port after a replan —
                # do not adopt it; reclaim only if provably nmesh-managed.
                if engine_listener_pid(service.port) == pid:
                    self._terminator(int(pid))
                return False
            create_time = entry.get("create_time")
            port = entry.get("port")
            self.adopted[service.name] = {
                "pid": pid,
                "create_time": (
                    float(create_time)
                    if isinstance(create_time, (int, float))
                    and not isinstance(create_time, bool)
                    else None
                ),
                "port": (
                    int(port)
                    if isinstance(port, int) and not isinstance(port, bool)
                    else None
                ),
            }
            self.external_shared.discard(service.name)
            return True
        self.adopted.pop(service.name, None)
        if self._healthy(service):
            orphan_pid = engine_listener_pid(service.port)
            if orphan_pid is not None and not _pid_serves_model(
                orphan_pid, service.model_ref
            ):
                # Our engine but serving a different model — reclaim the
                # port instead of silently proxying to the wrong model.
                self._terminator(orphan_pid)
                return False
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

    def _apply_acquired(
        self, plan: Plan, service: PlannedService, acquired: Acquired
    ) -> tuple[Plan, PlannedService, bool, bool]:
        if acquired.warning is not None:
            plan = replace(
                plan,
                warnings=[*plan.warnings, acquired.warning],
            )
            self.notes[service.name] = acquired.warning
        model_ref = acquired.model_ref or (
            str(acquired.path) if acquired.path is not None else None
        )
        if model_ref is None and acquired.artifact_bytes is None:
            return plan, service, False, False
        quant = acquired.quant or service.quant
        memory = service.memory
        real_bytes_warning = None
        artifact_replanned = False
        if (
            acquired.artifact_bytes is not None
            and acquired.artifact_bytes != service.memory.weight_bytes
        ):
            estimated_bytes = service.memory.weight_bytes
            model = next(
                (item for item in self.catalog() if item.id == service.model_id),
                None,
            )
            if model is not None:
                profile = self.probe()
                memory = estimate_memory(
                    model,
                    service.quant,
                    service.context,
                    parallel_slots=service.memory.parallel_slots,
                    profile=profile,
                    kv_quant=service.kv_quant,
                    budget_source=plan.policy.budget_source,
                    weight_bytes=float(acquired.artifact_bytes),
                )
                layers = service.memory.n_gpu_layers or service.n_gpu_layers or 0
                gpu_bytes, cpu_bytes = split_memory(
                    memory, model.n_layers, layers
                )
                memory = replace(
                    memory,
                    gpu_bytes=gpu_bytes,
                    cpu_bytes=cpu_bytes,
                    n_gpu_layers=layers,
                )
                if acquired.artifact_bytes > estimated_bytes * 1.10:
                    artifact_replanned = True
                    real_bytes_warning = i18n.t(
                        "warn.real_artifact_replanned",
                        i18n.lang(),
                        service=service.name,
                    )
                    plan = replace(
                        plan,
                        warnings=[*plan.warnings, real_bytes_warning],
                    )
                    self.notes[service.name] = real_bytes_warning
        if (
            model_ref is None
            or (
                model_ref == service.model_ref
                and quant == service.quant
                and memory == service.memory
            )
        ):
            updated = replace(service, memory=memory)
            updated_services = [
                updated if item.name == service.name else item
                for item in plan.services
            ]
            return (
                replace(plan, services=updated_services),
                updated,
                memory != service.memory,
                artifact_replanned,
            )
        argv = list(service.launch.argv)
        if service.backend == "llamacpp" and "-m" in argv:
            argv[argv.index("-m") + 1] = model_ref
        elif "--model" in argv:
            argv[argv.index("--model") + 1] = model_ref
        elif service.backend == "vllm" and len(argv) > 2:
            argv[2] = model_ref
        updated = replace(
            service,
            model_ref=model_ref,
            quant=quant,
            memory=memory,
            launch=replace(service.launch, argv=argv),
        )
        self.notes[service.name] = (
            f"{service.quant} -> {quant}" if quant != service.quant
            else i18n.t("info.resolved_gguf", i18n.lang(), name=Path(model_ref).name)
        )
        updated_services = [
            updated if item.name == service.name else item for item in plan.services
        ]
        return replace(plan, services=updated_services), updated, True, artifact_replanned

    def _resolve_launch_exe(
        self, service: PlannedService
    ) -> tuple[PlannedService, str | None]:
        """Re-resolve the engine binary behind a planned launch command when
        the engine it pointed at was removed or replaced after planning."""
        if service.backend != "llamacpp" or not service.launch.argv:
            return service, None
        exe = service.launch.argv[0]
        if Path(exe).exists():
            return service, None
        candidate = engine.active()
        if candidate is None or candidate.backend != service.backend:
            candidate = next(
                (
                    item
                    for item in engine.installed()
                    if item.backend == service.backend and item.exe.exists()
                ),
                None,
            )
        if candidate is None or not candidate.exe.exists():
            return service, None
        healed = replace(
            service,
            launch=replace(
                service.launch, argv=[str(candidate.exe), *service.launch.argv[1:]]
            ),
        )
        warning = i18n.t(
            "warn.engine_substituted", i18n.lang(),
            service=service.name, old=exe, tag=candidate.tag,
        )
        return healed, warning

    def _admit(
        self,
        plan: Plan,
        bench_cache: Mapping[object, float] | None = None,
        *,
        drop_unaffordable: bool = False,
    ) -> Plan:
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
            profile,
            self.catalog(),
            replace(plan.policy, budget_source="free"),
            bench_cache=bench_cache,
            artifact_cache=load_artifact_cache(),
        )
        warning = i18n.t(
            "warn.free_admission", i18n.lang(),
            need_gpu=need_gpu / GIB, need_cpu=need_cpu / GIB,
            vram=vram / GIB, ram=ram / GIB,
        )
        if replanned.services:
            return replace(
                replanned,
                warnings=[*plan.warnings, *replanned.warnings, warning],
            )
        if drop_unaffordable:
            return replace(
                plan,
                services=[],
                warnings=[*plan.warnings, warning],
            )
        return replace(
            plan,
            warnings=[
                *plan.warnings,
                warning + " " + i18n.t("warn.runtime_fallback", i18n.lang(), attempt=0),
            ],
        )

    def _launch(self, service: PlannedService) -> ProcessLike:
        env: dict[str, str] = {**os.environ, **service.launch.env}
        handle = None
        if os.environ.get("NMESH_BACKEND_LOG") != "0":
            try:
                handle = open_log(service.name)
            except OSError:
                handle = None
        stdout = handle
        stderr = subprocess.STDOUT if handle is not None else None
        try:
            if is_windows():
                return subprocess.Popen(
                    service.launch.argv,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    # only defined on Windows; is_windows() guards it
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,  # type: ignore[attr-defined]
                )
            return subprocess.Popen(
                service.launch.argv,
                env=env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
        finally:
            if handle is not None:
                handle.close()

    def _with_log_tail(self, service_name: str, message: str) -> str:
        recent = tail(service_name, 5)
        if not recent:
            return message
        detail = i18n.t(
            "err.service_log_tail",
            i18n.lang(),
            path=log_path(service_name),
            tail="\n".join(recent),
        )
        return (
            f"{message} "
            f"{detail}"
        )

    def _unhealthy_message(
        self, service_name: str, port: int | None = None
    ) -> str:
        message = i18n.t(
            "err.service_unhealthy", i18n.lang(), service=service_name
        )
        if port is not None and _port_in_use(port):
            message = f"{message} {i18n.t('err.service_port_in_use', i18n.lang(), port=port)}"
        return self._with_log_tail(service_name, message)

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
        existing = self._load_state() or {}
        previous_owner = existing.get("owner_pid")
        previous_entries = existing.get("services", [])
        if not isinstance(previous_entries, list):
            previous_entries = []
        owned_names = (
            set(self.processes)
            | self.shared_services
            | self.external_shared
            | set(self.adopted)
        )
        entries = [
            dict(entry) for entry in previous_entries
            if isinstance(entry, dict)
            and str(entry.get("service")) not in owned_names
            and (
                entry.get("owner_pid", previous_owner) != os.getpid()
                and self._entry_alive(entry)
            )
        ]
        entries.extend(
            {
                "service": name,
                "pid": process.pid,
                "port": _port(plan, name, 0),
                "started_at": time.time(),
                "create_time": self._create_time(process.pid),
                "shared": False,
                "external": False,
                "parallel_slots": _slots(plan, name),
                "health_url": next(
                    (item.launch.health_url for item in plan.services if item.name == name),
                    None,
                ),
                "model_ref": next(
                    (item.model_ref for item in plan.services if item.name == name),
                    None,
                ),
                "quant": next(
                    (item.quant for item in plan.services if item.name == name), None
                ),
                "backend": next(
                    (item.backend for item in plan.services if item.name == name), None
                ),
                "owner_pid": os.getpid(),
            }
            for name, process in self.processes.items()
        )
        entries.extend(
            {
                "service": name,
                "pid": record["pid"],
                "port": record.get("port") or _port(plan, name, 0),
                "started_at": time.time(),
                "create_time": record.get("create_time"),
                "shared": False,
                "external": True,
                "adopted": True,
                "parallel_slots": _slots(plan, name),
                "health_url": next(
                    (item.launch.health_url for item in plan.services if item.name == name),
                    None,
                ),
                "model_ref": next(
                    (item.model_ref for item in plan.services if item.name == name), None
                ),
                "quant": next(
                    (item.quant for item in plan.services if item.name == name), None
                ),
                "backend": next(
                    (item.backend for item in plan.services if item.name == name), None
                ),
                "owner_pid": os.getpid(),
            }
            for name, record in self.adopted.items()
            if name not in self.processes
        )
        entries.extend(
            {
                "service": name,
                "pid": None,
                "port": _port(plan, name, 11434),
                "started_at": time.time(),
                "shared": True,
                "external": False,
                "parallel_slots": _slots(plan, name),
                "health_url": next(
                    (item.launch.health_url for item in plan.services if item.name == name),
                    None,
                ),
                "model_ref": next(
                    (item.model_ref for item in plan.services if item.name == name),
                    None,
                ),
                "quant": next(
                    (item.quant for item in plan.services if item.name == name), None
                ),
                "backend": next(
                    (item.backend for item in plan.services if item.name == name), None
                ),
                "owner_pid": os.getpid(),
            }
            for name in self.shared_services if name not in self.processes
        )
        entries.extend(
            {
                "service": name,
                "pid": None,
                "port": _port(plan, name, 0),
                "started_at": time.time(),
                "shared": True,
                "external": True,
                "parallel_slots": _slots(plan, name),
                "health_url": next(
                    (item.launch.health_url for item in plan.services if item.name == name),
                    None,
                ),
                "model_ref": next(
                    (item.model_ref for item in plan.services if item.name == name),
                    None,
                ),
                "quant": next(
                    (item.quant for item in plan.services if item.name == name), None
                ),
                "backend": next(
                    (item.backend for item in plan.services if item.name == name), None
                ),
                "owner_pid": os.getpid(),
            }
            for name in self.external_shared
            if name not in self.processes and name not in self.shared_services
        )
        for entry in entries:
            if entry.get("create_time") is None:
                entry.pop("create_time", None)
            if entry.get("service") in self.notes:
                entry["note"] = self.notes[str(entry["service"])]
        payload: dict[str, object] = {
            "version": STATE_VERSION,
            "owner_pid": os.getpid(),
            "services": entries,
        }
        existing_gateway = existing.get("gateway")
        if isinstance(existing_gateway, dict) and self._entry_alive(existing_gateway):
            payload["gateway"] = existing_gateway
        self._write_state(payload)

    def _fallback(self, plan: Plan, attempt: int) -> Plan:
        quant_order = tuple(BPW)
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
                context_value = context
                if "--parallel" in argv:
                    context_value *= service.memory.parallel_slots
                argv[argv.index("-c") + 1] = str(context_value)
            for flag in ("-ngl", "--gpu-layers", "--n-gpu-layers"):
                if flag in argv and layers is not None:
                    argv[argv.index(flag) + 1] = str(layers)
            model_ref = service.model_ref
            if service.backend == "llamacpp":
                model_ref = model_ref.replace(f"-{service.quant}.gguf", f"-{quant}.gguf")
                if "-m" in argv:
                    argv[argv.index("-m") + 1] = model_ref
            launch = replace(service.launch, argv=argv)
            services.append(replace(service, quant=quant, context=context,
                                    model_ref=model_ref, n_gpu_layers=layers, launch=launch))
        warnings = [*plan.warnings]
        if attempt == 1:
            warnings.extend(
                i18n.t(
                    "warn.quant_fallback_skipped",
                    i18n.lang(),
                    service=service.name,
                    quant=service.quant,
                )
                for service in plan.services
                if service.quant not in quant_order
            )
        warnings.append(i18n.t("warn.runtime_fallback", i18n.lang(), attempt=attempt))
        return replace(plan, services=services, warnings=warnings)

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
            self._boot_recovery = False
            actualized = False
            replan_done = False
            if admit:
                try:
                    current = self._admit(plan, bench_cache)
                except Exception as error:  # noqa: BLE001
                    current = replace(
                        plan,
                         warnings=[*plan.warnings,
                                   i18n.t("warn.admission_skipped", i18n.lang(), error=error)],
                    )
            self.active_plan = current
            for attempt in range(1, 4):
                try:
                    for index in range(len(current.services)):
                        service = current.services[index]
                        dead = service.name in self.processes and not self._alive(service.name)
                        if dead:
                            self.processes.pop(service.name, None)
                        if service.name not in self.processes and self._adopt(service):
                            self.idle.discard(service.name)
                            continue
                        if self._already_up(service):
                            self.idle.discard(service.name)
                            if service.launch.shared_daemon:
                                self.shared_services.add(service.name)
                            continue
                        if dead:
                            if not self._restart_budget(service.name):
                                self.failed[service.name] = i18n.t(
                                    "err.restart_budget", i18n.lang(), service=service.name
                                )
                                raise RuntimeError(
                                    i18n.t("err.restart_budget", i18n.lang(),
                                           service=service.name)
                                )
                            self._record_restart(service.name)
                        if not no_download:
                            acquired = acquire(service)
                            current, service, changed, artifact_replanned = self._apply_acquired(
                                current, service, acquired
                            )
                            actualized = actualized or changed
                            if (
                                artifact_replanned
                                and admit
                                and not replan_done
                            ):
                                current = self._admit(
                                    current,
                                    bench_cache,
                                    drop_unaffordable=True,
                                )
                                replan_done = True
                                refreshed = next(
                                    (
                                        item for item in current.services
                                        if item.name == service.name
                                    ),
                                    None,
                                )
                                if refreshed is None:
                                    self.active_plan = current
                                    continue
                                service = refreshed
                            self.active_plan = current
                        elif service.backend == "ollama":
                            warning = i18n.t(
                                "warn.ollama_context_default",
                                i18n.lang(),
                                service=service.name,
                                context=service.context,
                            )
                            if warning not in current.warnings:
                                current = replace(
                                    current,
                                    warnings=[*current.warnings, warning],
                                )
                                self.active_plan = current
                            self.notes[service.name] = warning
                        service, heal_warning = self._resolve_launch_exe(service)
                        if heal_warning is not None:
                            current = replace(
                                current,
                                services=[
                                    service if item.name == service.name else item
                                    for item in current.services
                                ],
                            )
                            actualized = True
                            self.active_plan = current
                            self.notes[service.name] = heal_warning
                        self.processes[service.name] = self.launcher(service)
                        self.idle.discard(service.name)
                        self._arm_atexit()
                        self.failed.pop(service.name, None)
                        if not self._wait_health(service):
                            raise RuntimeError(
                                self._unhealthy_message(service.name, service.port)
                            )
                    if (current is plan or actualized) and {
                        item.name for item in current.services
                    } == {item.name for item in plan.services}:
                        # Only persist when the service set is unchanged: an
                        # admission drop under transient memory pressure must
                        # not silently shrink the user's saved plan.
                        save_plan(current, self.plan_path)
                    self._persist(current)
                    result = self.status()
                    # Admission may have dropped services this run; surface
                    # those warnings so the user sees the degradation.
                    result.warnings.extend(
                        warning
                        for warning in current.warnings
                        if warning not in plan.warnings
                    )
                    return result
                except (OSError, RuntimeError):
                    self.down()
                    if attempt == 3:
                        raise
                    current = self._fallback(current, attempt)
            raise RuntimeError(i18n.t("err.runtime_start", i18n.lang()))

    def down(self, foreign: bool = False, gateway_port: int | None = None) -> RuntimeStatus:
        swept: int | None = None
        stopped: list[dict[str, object]] = []
        seen_stopped: set[str] = set()

        def report(name: str, fields: dict[str, object]) -> None:
            if name in seen_stopped:
                return
            seen_stopped.add(name)
            stopped.append(
                {**fields, "service": name, "running": False}
            )

        with self._lock:
            state = self._load_state()
            if state is not None:
                gateway = state.get("gateway")
                if isinstance(gateway, dict):
                    recorded_port = gateway.get("port")
                    if (
                        gateway_port is None
                        and isinstance(recorded_port, int)
                        and not isinstance(recorded_port, bool)
                    ):
                        gateway_port = recorded_port
                    owner = gateway.get("owner_pid", state.get("owner_pid"))
                    if (
                        (owner == os.getpid() or foreign)
                        and self._entry_alive(gateway)
                        and gateway.get("pid") is not None
                    ):
                        self._terminator(int(gateway["pid"]))
                        swept = int(gateway["pid"])
                        report("gateway", {"pid": swept, "port": gateway_port})
            # A gateway owns the watchdog that respawns services — kill it
            # (recorded or orphaned) before touching service processes.
            if foreign and self._sweep_gateway(gateway_port, swept) is not None:
                report("gateway", {"port": gateway_port})
            # state.json may be lost while plan.json survives — reclaim
            # service orphans still bound to their planned ports.
            plan = self.active_plan if self.active_plan is not None else load_plan()
            if plan is not None:
                for service in plan.services:
                    if (
                        service.name in seen_stopped
                        or service.name in self.processes
                        or service.name in self.adopted
                        or service.port is None
                    ):
                        continue
                    orphan_pid = engine_listener_pid(service.port)
                    if orphan_pid is not None:
                        self._terminator(orphan_pid)
                        report(service.name, {
                            "pid": orphan_pid,
                            "port": service.port,
                            "model_ref": service.model_ref,
                            "backend": service.backend,
                        })
            adopted_names = set(self.adopted)
            for name, record in self.adopted.items():
                pid = record.get("pid")
                if isinstance(pid, int) and not isinstance(pid, bool):
                    self._terminator(pid)
                    report(name, record)
            for name, process in list(self.processes.items()):
                report_fields: dict[str, object] = {"pid": process.pid}
                planned = (
                    next(
                        (
                            item for item in self.active_plan.services
                            if item.name == name
                        ),
                        None,
                    )
                    if self.active_plan is not None
                    else None
                )
                if planned is not None:
                    report_fields.update({
                        "port": planned.port,
                        "model_ref": planned.model_ref,
                        "backend": planned.backend,
                    })
                if process.poll() is not None:
                    report(name, report_fields)
                    continue
                if is_windows():
                    process.terminate()
                else:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except OSError:
                        process.terminate()
                try:
                    process.wait(timeout=10)
                except (subprocess.TimeoutExpired, TimeoutError):
                    if not is_windows():
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except OSError:
                            process.kill()
                    else:
                        process.kill()
                report(name, report_fields)
            if state is not None:
                gateway_retained = False
                gateway = state.pop("gateway", None)
                if isinstance(gateway, dict):
                    owner = gateway.get("owner_pid", state.get("owner_pid"))
                    if not (owner == os.getpid() or foreign):
                        gateway_retained = bool(self._entry_alive(gateway))
                        if gateway_retained:
                            state["gateway"] = gateway
                retained: list[dict[str, object]] = []
                entries_state = state.get("services")
                for entry in entries_state if isinstance(entries_state, list) else ():
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("service") in adopted_names:
                        continue
                    owner = entry.get("owner_pid", state.get("owner_pid"))
                    if owner == os.getpid():
                        continue
                    if foreign and entry.get("pid") is not None and self._entry_alive(entry):
                        self._terminator(int(entry["pid"]))
                        report(str(entry.get("service")), entry)
                        continue
                    if self._entry_alive(entry):
                        retained.append(entry)
                state["services"] = retained
                if retained or gateway_retained:
                    self._write_state(state)
                else:
                    try:
                        self.state_path.unlink()
                    except FileNotFoundError:
                        pass
            self.processes.clear()
            self.shared_services.clear()
            self.external_shared.clear()
            self.adopted.clear()
            self.idle.clear()
            self.restarts.clear()
            self.failed.clear()
            self.active_plan = None
        return RuntimeStatus(False, stopped)

    def _sweep_gateway(self, port: int | None, swept: int | None) -> int | None:
        """Terminate an nmesh gateway still listening on *port* that state missed."""
        if port is None:
            return None
        orphan = gateway_listener_pid(port)
        if orphan is not None and orphan != swept:
            self._terminator(orphan)
            return orphan
        return None

    def ensure_running(self, service_name: str, plan: Plan | None = None) -> RuntimeStatus:
        with self._lock:
            selected = plan or self.active_plan
            if selected is None:
                raise FileNotFoundError(i18n.t("err.no_active_plan", i18n.lang()))
            self.active_plan = selected
            target = next((item for item in selected.services if item.name == service_name), None)
            if target is None:
                raise KeyError(i18n.t(
                    "err.unknown_service", i18n.lang(), service=service_name
                ))
            self.idle.discard(service_name)
            actualized = False
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
                        self.failed[service_name] = i18n.t(
                            "err.restart_budget", i18n.lang(), service=service_name
                        )
                        self._persist(selected)
                        raise RuntimeError(i18n.t(
                            "err.restart_budget", i18n.lang(), service=service_name
                        ))
                    self._record_restart(service_name)
                if service_name not in self.processes:
                    selected, target, actualized, _ = self._apply_acquired(
                        selected, target, acquire(target)
                    )
                    self.active_plan = selected
                self.processes[service_name] = self.launcher(target)
                self._arm_atexit()
                if not self._wait_health(target):
                    self._stop_process(service_name)
                    raise RuntimeError(
                        self._unhealthy_message(service_name, target.port)
                    )
                self.failed.pop(service_name, None)
            if actualized:
                save_plan(selected, self.plan_path)
            self._persist(selected)
            return self.status()

    def unload(self, service_name: str) -> bool:
        with self._lock:
            if service_name in self.shared_services or service_name in self.external_shared:
                return False
            adopted = self.adopted.get(service_name)
            if adopted is None and service_name not in self.processes:
                lookup_plan = self.active_plan
                if lookup_plan is None:
                    lookup_plan = load_plan()
                planned = (
                    next(
                        (
                            service for service in lookup_plan.services
                            if service.name == service_name
                        ),
                        None,
                    )
                    if lookup_plan is not None
                    else None
                )
                if planned is None:
                    # The in-memory plan can predate the saved plan when this
                    # supervisor belongs to a gateway that outlived an earlier
                    # `up`; the persisted plan is the source of truth.
                    persisted = load_plan()
                    if persisted is not None and persisted is not lookup_plan:
                        planned = next(
                            (
                                service
                                for service in persisted.services
                                if service.name == service_name
                            ),
                            None,
                        )
                        if planned is not None:
                            lookup_plan = persisted
                if planned is not None:
                    self._adopt(planned)
                    if service_name in self.external_shared:
                        return False
                    adopted = self.adopted.get(service_name)
                    if adopted is not None and (
                        self.active_plan is None
                        or lookup_plan is not self.active_plan
                    ):
                        self.active_plan = lookup_plan
            if adopted is not None:
                pid = adopted.get("pid")
                if not isinstance(pid, int) or isinstance(pid, bool):
                    return False
                self._terminator(pid)
                self.adopted.pop(service_name, None)
            elif service_name not in self.processes:
                return False
            else:
                self._stop_process(service_name)
            self.idle.add(service_name)
            self.restarts.pop(service_name, None)
            self.failed.pop(service_name, None)
            if self.active_plan is not None:
                self._persist(self.active_plan)
            return True

    def idle_services(self) -> set[str]:
        with self._lock:
            return set(self.idle)

    def _stop_process(self, service_name: str) -> None:
        process = self.processes.pop(service_name, None)
        if process is None:
            return
        if process.poll() is None:
            if is_windows():
                process.terminate()
            else:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except OSError:
                    process.terminate()
            try:
                process.wait(timeout=10)
            except (subprocess.TimeoutExpired, TimeoutError):
                if not is_windows():
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except OSError:
                        process.kill()
                else:
                    process.kill()

    def status(self) -> RuntimeStatus:
        def planned(name: str) -> PlannedService | None:
            if self.active_plan is None:
                return None
            return next(
                (item for item in self.active_plan.services if item.name == name), None
            )

        def planned_fields(name: str) -> dict[str, object]:
            service = planned(name)
            if service is None:
                return {}
            return {
                "model_ref": service.model_ref,
                "quant": service.quant,
                "backend": service.backend,
                "port": service.port,
            }

        entries = [{
            "service": name,
            "pid": process.pid,
            "running": process.poll() is None,
            "restarts": len(self.restarts.get(name, [])),
            "parallel_slots": _slots(self.active_plan, name),
            **planned_fields(name),
            **({"note": self.notes[name]} if name in self.notes else {}),
        } for name, process in self.processes.items()]
        entries.extend({
            "service": name,
            "pid": None,
            "running": True,
            "shared": True,
            "parallel_slots": _slots(self.active_plan, name),
            **planned_fields(name),
            **({"note": self.notes[name]} if name in self.notes else {}),
        } for name in self.shared_services if name not in self.processes)
        entries.extend({
            "service": name,
            "pid": record.get("pid"),
            "running": (
                self._healthy(target)
                if (target := planned(name)) is not None
                else self._entry_alive(record)
            ),
            "external": True,
            "adopted": True,
            "parallel_slots": _slots(self.active_plan, name),
            **planned_fields(name),
            **({"note": self.notes[name]} if name in self.notes else {}),
        } for name, record in self.adopted.items() if name not in self.processes)
        entries.extend({
            "service": name,
            "pid": None,
            "running": True,
            "shared": True,
            "external": True,
            "parallel_slots": _slots(self.active_plan, name),
            **planned_fields(name),
            **({"note": self.notes[name]} if name in self.notes else {}),
        } for name in self.external_shared if name not in self.processes)
        entries.extend({
            "service": name,
            "pid": None,
            "running": False,
            "idle": True,
            "restarts": len(self.restarts.get(name, [])),
            "parallel_slots": _slots(self.active_plan, name),
            **planned_fields(name),
            **({"note": self.notes[name]} if name in self.notes else {}),
        } for name in self.idle if name not in self.processes
        and name not in self.shared_services
        and name not in self.external_shared)
        payload = self._load_state() if self.state_path.exists() else None
        gateway = payload.get("gateway") if payload else None
        gateway_entry = None
        gateway_running = False
        if isinstance(gateway, dict):
            gateway_entry = dict(gateway)
            gateway_entry["service"] = "gateway"
            gateway_running = self._entry_alive(gateway)
            gateway_entry["running"] = gateway_running
        if not entries and payload is not None:
            persisted = payload.get("services")
            state_services = (
                [dict(item) for item in persisted if isinstance(item, dict)]
                if isinstance(persisted, list)
                else []
            )
            entries = state_services
            live_services = []
            for item in state_services:
                item["running"] = self._entry_alive(item)
                if item["running"]:
                    live_services.append(item)
            if gateway_entry is not None:
                entries.append(gateway_entry)
            if len(live_services) != len(state_services) or (
                gateway_entry is not None and not gateway_running
            ):
                if live_services or gateway_running:
                    payload["services"] = live_services
                    if gateway_running:
                        payload["gateway"] = gateway
                    else:
                        payload.pop("gateway", None)
                    self._write_state(payload)
                else:
                    try:
                        self.state_path.unlink()
                    except FileNotFoundError:
                        pass
        elif payload is not None:
            # state.json records everything nmesh started; this supervisor may
            # only track a subset in memory (e.g. the gateway never spawned
            # the services `nmesh up` did), so merge what it does not know.
            seen = {str(item.get("service")) for item in entries}
            persisted = payload.get("services")
            if isinstance(persisted, list):
                for item in persisted:
                    if (
                        isinstance(item, dict)
                        and str(item.get("service")) not in seen
                    ):
                        merged = dict(item)
                        merged["running"] = self._entry_alive(item)
                        entries.append(merged)
                        seen.add(str(item.get("service")))
            if gateway_entry is not None:
                entries.append(gateway_entry)
                if not gateway_running:
                    payload.pop("gateway", None)
                    if payload.get("services"):
                        self._write_state(payload)
                    else:
                        try:
                            self.state_path.unlink()
                        except FileNotFoundError:
                            pass
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
        return RuntimeStatus(
            any(bool(item.get("running", False)) for item in entries), entries
        )

    def heartbeat(self) -> RuntimeStatus:
        with self._lock:
            boot_recovery = self._boot_recovery
            if self.active_plan is None:
                loaded = load_plan()
                if loaded is None:
                    return self.status()
                self.active_plan = loaded
                self._boot_recovery = True
                boot_recovery = True
            persisted_names: set[str] = set()
            if boot_recovery:
                state = self._load_state()
                if state is not None:
                    persisted = state.get("services")
                    persisted_names = (
                        {
                            str(entry.get("service"))
                            for entry in persisted
                            if isinstance(entry, dict)
                            and entry.get("service") is not None
                        }
                        if isinstance(persisted, list)
                        else set()
                    )
            changed = False
            for service in self.active_plan.services:
                if service.name in self.idle:
                    continue
                adopted = self.adopted.get(service.name)
                adopted_dead = adopted is not None and not self._entry_alive(adopted)
                if adopted_dead:
                    self.adopted.pop(service.name, None)
                    self.external_shared.discard(service.name)
                    changed = True
                if (
                    service.name in self.active_plan.swap_group
                    and service.name not in self.processes
                    and service.name not in self.external_shared
                ):
                    continue
                if not adopted_dead and self._adopt(service):
                    self.failed.pop(service.name, None)
                    changed = True
                    continue
                if self._already_up(service):
                    continue
                if boot_recovery and (
                    not service.resident and service.name not in persisted_names
                ):
                    continue
                if service.name in self.failed:
                    continue
                if service.name in self.processes:
                    self.processes.pop(service.name, None)
                if not self._restart_budget(service.name):
                    self.failed[service.name] = i18n.t(
                        "err.restart_budget", i18n.lang(), service=service.name
                    )
                    changed = True
                    continue
                self._record_restart(service.name)
                changed = True
                try:
                    self.processes[service.name] = self.launcher(service)
                    self._arm_atexit()
                    if not self._wait_health(service, timeout=min(self.health_timeout, 30.0)):
                        self._stop_process(service.name)
                        if not self._restart_budget(service.name):
                            self.failed[service.name] = self._with_log_tail(
                                service.name,
                                i18n.t("warn.health_failed", i18n.lang(),
                                       service=service.name),
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
        raise FileNotFoundError(i18n.t("err.no_plan", i18n.lang()))
    return _default.up(
        selected, no_download=no_download, dry_run=dry_run, admit=admit,
        bench_cache=bench_cache,
    )


def down(foreign: bool = False, gateway_port: int | None = None) -> RuntimeStatus:
    return _default.down(foreign=foreign, gateway_port=gateway_port)


def unload(service_name: str) -> bool:
    return _default.unload(service_name)


def idle_services() -> set[str]:
    return _default.idle_services()


def record_gateway(pid: int, port: int) -> None:
    _default.record_gateway(pid, port)


def clear_gateway(pid: int | None = None) -> None:
    _default.clear_gateway(pid)


def disarm_atexit() -> None:
    _default.disarm_atexit()


def stop_gateway(foreign: bool = False) -> bool:
    return _default.stop_gateway(foreign=foreign)


def status() -> RuntimeStatus:
    return _default.status()


def ensure_running(service_name: str, plan: Plan | None = None) -> RuntimeStatus:
    return _default.ensure_running(service_name, plan)


def heartbeat() -> RuntimeStatus:
    return _default.heartbeat()
