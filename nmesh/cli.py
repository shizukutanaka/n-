from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

from rich.console import Console
from rich.table import Table

from nmesh.bench import benchmark_key, load_cache, measure, save_cache
from nmesh.catalog import load_catalog
from nmesh.planner import (
    PlannedService,
    Policy,
    build_plan,
    free_budgets,
    load_plan,
    save_plan,
)
from nmesh.probe import detect_hardware
from nmesh.runtime import RuntimeStatus, clear_gateway, disarm_atexit, record_gateway
from nmesh.runtime import down as runtime_down
from nmesh.runtime import status as runtime_status
from nmesh.runtime import up as runtime_up
from nmesh.runtime.service_unit import service_unit
from nmesh.telemetry import bench_overlay
from nmesh.telemetry import summary as telemetry_summary


def _bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    number = value
    for unit in units:
        if abs(number) < 1024 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024
    return f"{value:.2f} B"


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, default=str))


def _doctor(as_json: bool) -> int:
    profile = detect_hardware()
    free_vram, free_ram = free_budgets(profile)
    if as_json:
        data = asdict(profile)
        data["free_budgets"] = {"vram_bytes": free_vram, "ram_bytes": free_ram}
        _print_json(data)
        return 0
    table = Table(title="nmesh doctor")
    table.add_column("Item")
    table.add_column("Value")
    table.add_row("OS", profile.os)
    table.add_row("CPU", profile.cpu_name)
    table.add_row("RAM", f"{_bytes(profile.total_ram_bytes)} / {_bytes(profile.available_ram_bytes)} free")
    table.add_row("Tier", profile.tier.value)
    table.add_row("GPU", ", ".join(gpu.name for gpu in profile.gpus) or "none")
    for gpu in profile.gpus:
        table.add_row(
            f"GPU {gpu.index} VRAM",
            f"{_bytes(gpu.total_vram_bytes)} / {_bytes(gpu.free_vram_bytes)} free",
        )
    table.add_row("Free budget VRAM", _bytes(free_vram))
    table.add_row("Free budget RAM", _bytes(free_ram))
    Console().print(table)
    backend = Table(title="Backends")
    backend.add_column("Backend")
    backend.add_column("Binary")
    backend.add_column("Version")
    backend.add_column("Flags")
    for name, version in profile.available_backends.items():
        flags = profile.backend_flags.get(name)
        backend.add_row(
            name,
            profile.backend_paths.get(name, "not found"),
            version or "not found",
            str(len(flags)) if flags is not None else "unknown",
        )
    Console().print(backend)
    for warning in profile.warnings:
        Console().print(f"[yellow]- {warning}[/yellow]")
    return 0


def _make_plan(args: argparse.Namespace) -> object:
    profile = detect_hardware()
    roles = [role.strip() for role in args.roles.split(",") if role.strip()]
    policy = Policy(roles=roles or ["chat", "code", "embed"], prefer=args.prefer,
                    max_context=args.context, budget_source=getattr(args, "budget", "total"),
                    parallel_slots=getattr(args, "parallel_slots", None))
    live = bench_overlay()
    args._telemetry_keys = len(live)
    cache = {**load_cache(), **live}
    return build_plan(profile, load_catalog(), policy, cache)


def _plan(args: argparse.Namespace) -> int:
    result = _make_plan(args)
    path = save_plan(result)
    if args.json:
        _print_json(asdict(result))
        return 0
    table = Table(title=f"nmesh plan ({result.tier.value})")
    for column in (
        "Service", "Roles", "Model", "Backend", "Context", "Slots", "GPU layers", "tok/s",
    ):
        table.add_column(column)
    for service in result.services:
        table.add_row(service.name, ",".join(service.roles), service.model_id, service.backend,
                      str(service.context), str(service.memory.parallel_slots),
                      str(service.n_gpu_layers), f"{service.decode_tps:.1f}")
    Console().print(table)
    Console().print(f"Saved to: {path}")
    if result.policy.budget_source == "free":
        Console().print("Budgets use currently-free memory.")
    if getattr(args, "_telemetry_keys", 0):
        Console().print(
            f"Live telemetry overlay: {args._telemetry_keys} benchmark key(s)"
        )
    for hint in result.install_hints:
        Console().print(f"[yellow]Install: {hint}[/yellow]")
    for warning in result.warnings:
        Console().print(f"[yellow]Warning: {warning}[/yellow]")
    if args.explain:
        memory = Table(title="Memory")
        for column in ("Service", "Weights", "KV", "GPU / CPU"):
            memory.add_column(column)
        for service in result.services:
            item = service.memory
            memory.add_row(service.name, _bytes(item.weight_bytes), _bytes(item.kv_cache_bytes),
                           f"{_bytes(item.gpu_bytes)} / {_bytes(item.cpu_bytes)}")
        Console().print(memory)
    return 0


def _runtime(args: argparse.Namespace) -> int:
    if args.command == "serve":
        process, _ = _launch_gateway(args.port, detach=False)
        try:
            process.wait()
        except KeyboardInterrupt:
            process.terminate()
        finally:
            clear_gateway(process.pid)
        return 0
    if args.command == "up":
        gateway_log: Path | None = None
        plan = load_plan()
        if plan is None:
            _plan(argparse.Namespace(roles="chat,code,embed", prefer="balanced",
                                     context=None, budget="total", json=False, explain=False))
            plan = load_plan()
        if plan is None:
            return 1
        cache = {**load_cache(), **bench_overlay()}
        result = runtime_up(
            plan, no_download=args.no_download, dry_run=args.dry_run,
            admit=not args.ignore_free_memory, bench_cache=cache,
        )
        if not args.dry_run:
            process, log_path = _launch_gateway(args.port, detach=args.detach)
            if args.detach:
                gateway_log = log_path
                disarm_atexit()
                if not _wait_gateway(args.port, process):
                    clear_gateway(process.pid)
                    process.terminate()
                    print(
                        f"gateway did not become ready; see {log_path}",
                        file=sys.stderr,
                    )
                    runtime_down()
                    return 1
                result = runtime_status()
            else:
                try:
                    process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    runtime_down()
                finally:
                    clear_gateway(process.pid)
    elif args.command == "down":
        result = runtime_down(foreign=True)
    else:
        result = runtime_status()
        gateway = next(
            (item for item in result.services if item.get("service") == "gateway"),
            None,
        )
        try:
            port = int(gateway.get("port", 18000)) if gateway else 18000
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                if gateway is None:
                    result.services.append({"service": "gateway", "port": port, "running": True})
                elif gateway.get("pid") is not None and not gateway.get("running"):
                    gateway["note"] = (
                        "gateway health endpoint responds but its recorded PID is not alive"
                    )
                else:
                    gateway["running"] = True
        except OSError:
            if gateway is None:
                result.services.append({"service": "gateway", "port": 18000, "running": False})
            elif gateway.get("running"):
                gateway["note"] = "gateway PID is live but its health endpoint is unavailable"
            else:
                gateway["running"] = False
        result.running = any(bool(item.get("running")) for item in result.services)
    status_data = asdict(result)
    if args.command == "up" and args.detach and gateway_log is not None:
        status_data["gateway_log"] = str(gateway_log)
    if args.command == "status":
        status_data["telemetry"] = telemetry_summary()
    if args.json:
        _print_json(status_data)
    elif args.command == "up" and args.dry_run:
        for item in result.services:
            Console().print(
                f"{item['service']}: backend={item['backend']} model_ref={item['model_ref']} "
                f"port={item['port']} context={item['context']} slots={item['parallel_slots']} "
                f"n_gpu_layers={item['n_gpu_layers']} argv={' '.join(str(x) for x in item['argv'])}"
            )
    else:
        Console().print(result)
        if args.command == "up" and args.detach and gateway_log is not None:
            Console().print(f"Gateway log: {gateway_log}")
        if args.command == "status":
            for item in result.services:
                if item.get("note"):
                    Console().print(
                        f"{item.get('service')}: note: {item['note']}"
                    )
            telemetry = telemetry_summary()
            table = Table(title="Telemetry")
            table.add_column("Service")
            table.add_column("Samples")
            table.add_column("Decode median")
            table.add_column("TTFT median")
            table.add_column("TTFT p95")
            table.add_column("Total median")
            for service, metrics in telemetry.items():
                table.add_row(
                    service,
                    str(int(metrics["samples"])),
                    f"{metrics.get('decode_tps_median', 0):.2f}",
                    f"{metrics.get('ttft_s_median', 0):.3f}",
                    f"{metrics.get('ttft_s_p95', 0):.3f}",
                    f"{metrics.get('total_s_median', 0):.3f}",
                )
            Console().print(table)
    return 0


def _launch_gateway(port: int, detach: bool) -> tuple[subprocess.Popen[bytes], Path | None]:
    command = [sys.executable, "-m", "nmesh.gateway.server", "--port", str(port)]
    kwargs: dict[str, object] = {}
    log_path: Path | None = None
    log = None
    if detach:
        if os.name == "nt":
            kwargs["creationflags"] = 0x00000008 | 0x00000200
        else:
            kwargs["start_new_session"] = True
        log_path = Path.home() / ".nmesh" / "gateway.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("ab")
        kwargs["stdout"] = log
        kwargs["stderr"] = log
        kwargs["close_fds"] = True
    process = subprocess.Popen(command, **kwargs)
    if log is not None:
        log.close()
    record_gateway(process.pid, port)
    return process, log_path


def _wait_gateway(port: int, process: subprocess.Popen[bytes], timeout: float = 20.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                if response.status < 500:
                    return True
        except OSError:
            time.sleep(0.2)
    return False


def _reload(args: argparse.Namespace) -> int:
    request = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/admin/reload",
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status >= 400:
                return 1
            data = json.loads(response.read().decode())
    except (OSError, json.JSONDecodeError) as error:
        print(f"gateway reload failed: {error}", file=sys.stderr)
        return 1
    if args.json:
        _print_json(data)
    else:
        print(
            f"Reloaded: {', '.join(data.get('services', []))} "
            f"(created_at={data.get('created_at')})"
        )
    return 0


def _models(args: argparse.Namespace) -> int:
    models = load_catalog()
    if args.role:
        models = [model for model in models if args.role in model.roles]
    if args.json:
        _print_json([asdict(model) for model in models])
        return 0
    table = Table(title="Models")
    for column in ("ID", "Family", "Params", "Roles", "Context"):
        table.add_column(column)
    for model in models:
        table.add_row(model.id, model.family, str(model.params), ",".join(model.roles),
                      str(model.max_context))
    Console().print(table)
    return 0


def _service_running(service: PlannedService, runtime: RuntimeStatus) -> bool:
    if not any(item.get("service") == service.name and item.get("running", True)
               for item in runtime.services):
        return False
    health_url = service.launch.health_url
    if health_url is None:
        return True
    try:
        with urllib.request.urlopen(health_url, timeout=2):
            return True
    except OSError:
        return False


def _bench(args: argparse.Namespace) -> int:
    plan = load_plan()
    if plan is None or not plan.services:
        return 1
    service = next((item for item in plan.services if item.name == args.service), plan.services[0])
    running = runtime_status()
    if not _service_running(service, running):
        print("このサービスは起動していません。先に nmesh up を実行してください", file=sys.stderr)
        return 1
    base_url = "http://127.0.0.1:11434" if service.backend == "ollama" else (
        f"http://127.0.0.1:{service.port}"
    )
    try:
        measurement = measure(service, base_url, decode_tokens=args.tokens)
    except (OSError, RuntimeError) as error:
        print(f"ベンチマークに失敗しました: {error}", file=sys.stderr)
        return 1
    cache = load_cache()
    key = benchmark_key(service.model_id, service.quant, service.backend,
                        plan.profile.gpus[0].name if plan.profile.gpus else "cpu",
                        service.n_gpu_layers)
    cache[key] = measurement.decode_tps
    save_cache(cache)
    result = {"key": key, "prefill_tokens": 512, "decode_tokens": args.tokens,
              "median_tps": cache[key], "prefill_tps": measurement.prefill_tps,
              "ttft_s": measurement.ttft_s}
    _print_json(result) if args.json else Console().print(result)
    return 0


def _run_prompt(args: argparse.Namespace) -> int:
    payload = json.dumps({"model": f"nmesh-{args.role}",
                          "messages": [{"role": "user", "content": args.prompt}]}).encode()
    request = urllib.request.Request("http://127.0.0.1:18000/v1/chat/completions", payload,
                                     {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.loads(response.read().decode())
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                print(payload["choices"][0]["message"]["content"])
            return 0
    except (OSError, json.JSONDecodeError, KeyError, IndexError) as error:
        output = f"gateway unavailable: {error}"
    print(output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nmesh")
    parser.add_argument("--dry-run", action="store_true", dest="global_dry_run")
    parser.add_argument("--json", action="store_true", dest="global_json")
    sub = parser.add_subparsers(dest="command")
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--json", action="store_true")
    plan = sub.add_parser("plan")
    plan.add_argument("--json", action="store_true")
    plan.add_argument("--explain", action="store_true")
    plan.add_argument("--prefer", choices=("quality", "speed", "balanced"), default="balanced")
    plan.add_argument("--roles", default="chat,code,embed")
    plan.add_argument("--context", type=int)
    plan.add_argument("--budget", choices=("total", "free"), default="total")
    plan.add_argument("--parallel-slots", type=int)
    up_parser = sub.add_parser("up")
    up_parser.add_argument("--json", action="store_true")
    up_parser.add_argument("--dry-run", action="store_true")
    up_parser.add_argument("--no-download", action="store_true")
    up_parser.add_argument("--detach", action="store_true")
    up_parser.add_argument("--port", type=int, default=18000)
    up_parser.add_argument("--ignore-free-memory", action="store_true")
    serve_parser = sub.add_parser("serve")
    serve_parser.add_argument("--port", type=int, default=18000)
    reload_parser = sub.add_parser("reload")
    reload_parser.add_argument("--port", type=int, default=18000)
    reload_parser.add_argument("--json", action="store_true")
    for name in ("status", "down"):
        item = sub.add_parser(name)
        item.add_argument("--json", action="store_true")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("prompt")
    run_parser.add_argument("--role", default="chat")
    run_parser.add_argument("--json", action="store_true")
    bench_parser = sub.add_parser("bench")
    bench_parser.add_argument("--service", default="chat")
    bench_parser.add_argument("--tokens", type=int, default=128)
    bench_parser.add_argument("--json", action="store_true")
    auto = sub.add_parser("autotune")
    auto.add_argument("--json", action="store_true")
    autostart = sub.add_parser("autostart")
    autostart.add_argument("--port", type=int, default=18000)
    autostart.add_argument("--json", action="store_true")
    models = sub.add_parser("models")
    models.add_argument("--role")
    models.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.global_json and hasattr(args, "json"):
        args.json = True
    if args.command == "doctor":
        return _doctor(args.json)
    if args.command == "plan":
        return _plan(args)
    if args.command == "reload":
        return _reload(args)
    if args.command in {"up", "down", "status", "serve"}:
        if args.command == "up":
            args.dry_run = args.dry_run or args.global_dry_run
        return _runtime(args)
    if args.command == "models":
        return _models(args)
    if args.command == "bench":
        return _bench(args)
    if args.command == "autotune":
        plan = load_plan()
        if plan is None or not plan.services:
            return 1
        service = plan.services[0]
        context_values = sorted({max(service.context // 2, 128), service.context})
        layer_values = sorted({service.n_gpu_layers or 0, max((service.n_gpu_layers or 0) // 2, 0)})
        running = runtime_status()
        if not _service_running(service, running):
            print("このサービスは起動していません。先に nmesh up を実行してください", file=sys.stderr)
            return 1
        base_url = "http://127.0.0.1:11434" if service.backend == "ollama" else (
            f"http://127.0.0.1:{service.port}"
        )
        best: tuple[int, int, float] | None = None
        best_plan = None
        for context in context_values:
            for layers in layer_values:
                argv = list(service.launch.argv)
                if "--max-model-len" in argv:
                    argv[argv.index("--max-model-len") + 1] = str(context)
                if "-c" in argv:
                    argv[argv.index("-c") + 1] = str(context)
                if "-ngl" in argv:
                    argv[argv.index("-ngl") + 1] = str(layers)
                tuned = replace(
                    service, context=context, n_gpu_layers=layers,
                    launch=replace(service.launch, argv=argv),
                )
                tuned_plan = replace(plan, services=[
                    tuned if item.name == service.name else item for item in plan.services
                ])
                runtime_down()
                try:
                    runtime_up(tuned_plan, no_download=True)
                    result = measure(tuned, base_url)
                except (OSError, RuntimeError) as error:
                    print(f"オートチューンに失敗しました: {error}", file=sys.stderr)
                    runtime_down()
                    try:
                        runtime_up(plan, no_download=True)
                    except (OSError, RuntimeError) as restore_error:
                        print(
                            f"failed to restore original autotune configuration: {restore_error}",
                            file=sys.stderr,
                        )
                    return 1
                save_plan(plan)
                if best is None or result.decode_tps > best[2]:
                    best = (context, layers, result.decode_tps)
                    best_plan = tuned_plan
        if best is None:
            runtime_down()
            try:
                runtime_up(plan, no_download=True)
            except (OSError, RuntimeError):
                pass
            return 1
        runtime_down()
        assert best_plan is not None
        save_plan(best_plan)
        try:
            runtime_up(best_plan, no_download=True)
        except (OSError, RuntimeError) as error:
            save_plan(plan)
            runtime_down()
            try:
                runtime_up(plan, no_download=True)
            except (OSError, RuntimeError) as restore_error:
                print(
                    f"failed to restore original autotune configuration: {restore_error}",
                    file=sys.stderr,
                )
            print(f"failed to restore winning autotune configuration: {error}", file=sys.stderr)
            return 1
        result = {"service": service.name, "context": best[0], "n_gpu_layers": best[1],
                  "decode_tps": best[2]}
        _print_json(result) if args.json else Console().print(result)
        return 0
    if args.command == "autostart":
        filename, text, install_command = service_unit(args.port)
        data = {
            "filename": filename,
            "text": text,
            "install_command": install_command,
        }
        if args.json:
            _print_json(data)
        else:
            print(f"Filename: {filename}\n\n{text}\nInstall with:\n{install_command}")
        return 0
    if args.command == "run":
        return _run_prompt(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
