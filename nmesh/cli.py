from __future__ import annotations

import argparse
import json
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict

from rich.console import Console
from rich.table import Table

from nmesh.bench import autotune, benchmark, benchmark_key, load_cache, save_cache
from nmesh.catalog import load_catalog
from nmesh.planner import Policy, build_plan, load_plan, save_plan
from nmesh.probe import detect_hardware
from nmesh.runtime import down as runtime_down
from nmesh.runtime import status as runtime_status
from nmesh.runtime import up as runtime_up


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
    if as_json:
        _print_json(asdict(profile))
        return 0
    table = Table(title="nmesh doctor")
    table.add_column("Item")
    table.add_column("Value")
    table.add_row("OS", profile.os)
    table.add_row("CPU", profile.cpu_name)
    table.add_row("RAM", f"{_bytes(profile.total_ram_bytes)} / {_bytes(profile.available_ram_bytes)} free")
    table.add_row("Tier", profile.tier.value)
    table.add_row("GPU", ", ".join(gpu.name for gpu in profile.gpus) or "none")
    Console().print(table)
    backend = Table(title="Backends")
    backend.add_column("Backend")
    backend.add_column("Version")
    for name, version in profile.available_backends.items():
        backend.add_row(name, version or "not found")
    Console().print(backend)
    for warning in profile.warnings:
        Console().print(f"[yellow]- {warning}[/yellow]")
    return 0


def _make_plan(args: argparse.Namespace) -> object:
    profile = detect_hardware()
    roles = [role.strip() for role in args.roles.split(",") if role.strip()]
    policy = Policy(roles=roles or ["chat", "code", "embed"], prefer=args.prefer,
                    max_context=args.context)
    return build_plan(profile, load_catalog(), policy, load_cache())


def _plan(args: argparse.Namespace) -> int:
    result = _make_plan(args)
    path = save_plan(result)
    if args.json:
        _print_json(asdict(result))
        return 0
    table = Table(title=f"nmesh plan ({result.tier.value})")
    for column in ("Service", "Roles", "Model", "Backend", "Context", "GPU layers", "tok/s"):
        table.add_column(column)
    for service in result.services:
        table.add_row(service.name, ",".join(service.roles), service.model_id, service.backend,
                      str(service.context), str(service.n_gpu_layers), f"{service.decode_tps:.1f}")
    Console().print(table)
    Console().print(f"Saved to: {path}")
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
    if args.command == "up":
        plan = load_plan()
        if plan is None:
            _plan(argparse.Namespace(roles="chat,code,embed", prefer="balanced",
                                     context=None, json=False, explain=False))
            plan = load_plan()
        if plan is None:
            return 1
        result = runtime_up(plan, no_download=args.no_download, dry_run=args.dry_run)
    elif args.command == "down":
        result = runtime_down()
    else:
        result = runtime_status()
    if args.json:
        _print_json(asdict(result))
    elif args.command == "up" and args.dry_run:
        for item in result.services:
            Console().print(f"{item['service']}: {' '.join(str(x) for x in item['argv'])}")
    else:
        Console().print(result)
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


def _bench(args: argparse.Namespace) -> int:
    plan = load_plan()
    if plan is None or not plan.services:
        return 1
    service = next((item for item in plan.services if item.name == args.service), plan.services[0])
    cache = load_cache()
    key = benchmark_key(service.model_id, service.quant, service.backend,
                        plan.profile.gpus[0].name if plan.profile.gpus else "cpu",
                        service.n_gpu_layers)
    cache[key] = benchmark(lambda _prefill, _decode: service.decode_tps)
    save_cache(cache)
    result = {"key": key, "prefill_tokens": 512, "decode_tokens": args.tokens,
              "median_tps": cache[key]}
    _print_json(result) if args.json else Console().print(result)
    return 0


def _run_prompt(args: argparse.Namespace) -> int:
    payload = json.dumps({"model": f"nmesh-{args.role}",
                          "messages": [{"role": "user", "content": args.prompt}]}).encode()
    request = urllib.request.Request("http://127.0.0.1:18000/v1/chat/completions", payload,
                                     {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            output = response.read().decode()
    except OSError as error:
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
    up_parser = sub.add_parser("up")
    up_parser.add_argument("--json", action="store_true")
    up_parser.add_argument("--dry-run", action="store_true")
    up_parser.add_argument("--no-download", action="store_true")
    up_parser.add_argument("--detach", action="store_true")
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
    if args.command in {"up", "down", "status"}:
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
        best_context, best_layers, best_tps = autotune(
            lambda _context, _layers: service.decode_tps, context_values, layer_values
        )
        result = {"service": service.name, "context": best_context, "n_gpu_layers": best_layers,
                  "decode_tps": best_tps}
        _print_json(result) if args.json else Console().print(result)
        return 0
    if args.command == "run":
        return _run_prompt(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
