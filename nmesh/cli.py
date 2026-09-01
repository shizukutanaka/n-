from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict

from rich.console import Console
from rich.table import Table

from nmesh.catalog import load_catalog
from nmesh.planner import Policy, build_plan, save_plan
from nmesh.probe import detect_hardware


def _bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    number = value
    for unit in units:
        if abs(number) < 1024 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024
    return f"{value:.2f} B"


def _doctor() -> int:
    profile = detect_hardware()
    console = Console()
    table = Table(title="nmesh doctor")
    table.add_column("Item")
    table.add_column("Value")
    table.add_row("OS", profile.os)
    table.add_row("CPU", profile.cpu_name)
    table.add_row("CPU cores", f"{profile.physical_cores} physical / {profile.logical_cores} logical")
    table.add_row("RAM", f"{_bytes(profile.total_ram_bytes)} total / {_bytes(profile.available_ram_bytes)} available")
    table.add_row("Tier", profile.tier.value)
    if profile.gpus:
        for gpu in profile.gpus:
            table.add_row(
                f"GPU {gpu.index}",
                f"{gpu.name} ({gpu.vendor}, {_bytes(gpu.total_vram_bytes)})",
            )
    else:
        table.add_row("GPU", "none")
    console.print(table)
    backend_table = Table(title="Backends")
    backend_table.add_column("Backend")
    backend_table.add_column("Detected version")
    for name, version in profile.available_backends.items():
        backend_table.add_row(name, version or "not found")
    console.print(backend_table)
    if profile.warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in profile.warnings:
            console.print(f"- {warning}")
    return 0


def _plan(args: argparse.Namespace) -> int:
    profile = detect_hardware()
    roles = [role.strip() for role in args.roles.split(",") if role.strip()]
    policy = Policy(roles=roles or ["chat", "code", "embed"], prefer=args.prefer)
    result = build_plan(profile, load_catalog(), policy)
    path = save_plan(result)
    if args.json:
        print(json.dumps(asdict(result), indent=2, default=str))
        return 0
    console = Console()
    table = Table(title=f"nmesh plan ({result.tier.value})")
    table.add_column("Service")
    table.add_column("Roles")
    table.add_column("Model")
    table.add_column("Quant")
    table.add_column("Backend")
    table.add_column("Context")
    table.add_column("GPU layers")
    table.add_column("tok/s")
    table.add_column("Resident")
    for service in result.services:
        table.add_row(
            service.name,
            ",".join(service.roles),
            service.model_id,
            service.quant,
            service.backend,
            str(service.context),
            str(service.n_gpu_layers),
            f"{service.decode_tps:.1f}{'*' if service.estimated else ''}",
            "yes" if service.resident else "swap",
        )
    console.print(table)
    console.print(f"Saved to: {path}")
    if result.swap_group:
        console.print(f"Swap group: {', '.join(result.swap_group)}")
    if result.install_hints:
        console.print("[yellow]Install hints:[/yellow]")
        for hint in result.install_hints:
            console.print(f"- {hint}")
    if result.warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in result.warnings:
            console.print(f"- {warning}")
    if args.explain:
        explain = Table(title="Memory")
        explain.add_column("Service")
        explain.add_column("Weights")
        explain.add_column("KV cache")
        explain.add_column("Overhead")
        explain.add_column("GPU / CPU")
        for service in result.services:
            memory = service.memory
            explain.add_row(
                service.name,
                _bytes(memory.weight_bytes),
                _bytes(memory.kv_cache_bytes),
                _bytes(memory.compute_overhead),
                f"{_bytes(memory.gpu_bytes)} / {_bytes(memory.cpu_bytes)}",
            )
        console.print(explain)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nmesh", description="Local LLM hardware planner")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("doctor", help="detect hardware and available backends")
    plan_parser = subparsers.add_parser("plan", help="create a model placement plan")
    plan_parser.add_argument("--explain", action="store_true")
    plan_parser.add_argument("--json", action="store_true")
    plan_parser.add_argument("--prefer", choices=("quality", "speed", "balanced"), default="balanced")
    plan_parser.add_argument("--roles", default="chat,code,embed")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor()
    if args.command == "plan":
        return _plan(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
