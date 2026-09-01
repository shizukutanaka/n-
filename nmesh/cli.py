from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, replace

from rich.console import Console
from rich.table import Table

from nmesh import i18n
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
from nmesh.runtime import RuntimeStatus
from nmesh.runtime import down as runtime_down
from nmesh.runtime import status as runtime_status
from nmesh.runtime import up as runtime_up
from nmesh.telemetry import bench_overlay
from nmesh.telemetry import summary as telemetry_summary


def _configure_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _parse_languages(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    result: list[str] = []
    for item in value.split(","):
        primary = item.strip().lower().replace("_", "-").split("-", 1)[0]
        if primary and primary not in result:
            result.append(primary)
    return tuple(result)


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


def _profile_warnings(profile: object, language: str) -> list[str]:
    warnings = getattr(profile, "warnings", [])
    params = getattr(profile, "warning_params", [])
    return [
        i18n.t(
            warning,
            language,
            **(
                params[index]
                if index < len(params) and isinstance(params[index], dict)
                else {}
            ),
        )
        for index, warning in enumerate(warnings)
    ]


def _doctor(as_json: bool) -> int:
    profile = detect_hardware()
    language = i18n.lang()
    free_vram, free_ram = free_budgets(profile)
    selected = load_plan()
    selected_models = [
        {"service": service.name, "model": service.model_id,
         "languages": list(service.languages)}
        for service in selected.services
    ] if selected is not None else []
    localized_warnings = _profile_warnings(profile, language)
    if as_json:
        data = asdict(profile)
        data["warnings"] = localized_warnings
        data["free_budgets"] = {"vram_bytes": free_vram, "ram_bytes": free_ram}
        data["selected_models"] = selected_models
        _print_json(data)
        return 0
    table = Table(title="nmesh doctor")
    table.add_column(i18n.t("label.item", language))
    table.add_column(i18n.t("label.value", language))
    table.add_row("OS", profile.os)
    table.add_row("CPU", profile.cpu_name)
    table.add_row("RAM", f"{_bytes(profile.total_ram_bytes)} / {_bytes(profile.available_ram_bytes)} "
                  f"{i18n.t('label.free', language)}")
    table.add_row("Tier", profile.tier.value)
    table.add_row("GPU", ", ".join(gpu.name for gpu in profile.gpus)
                  or i18n.t("label.none", language))
    for gpu in profile.gpus:
        table.add_row(
            f"GPU {gpu.index} VRAM",
            f"{_bytes(gpu.total_vram_bytes)} / {_bytes(gpu.free_vram_bytes)} "
            f"{i18n.t('label.free', language)}",
        )
    table.add_row("Free budget VRAM", _bytes(free_vram))
    table.add_row("Free budget RAM", _bytes(free_ram))
    Console(legacy_windows=False).print(table)
    backend = Table(title=i18n.t("label.backends", language))
    backend.add_column(i18n.t("label.backend", language))
    backend.add_column(i18n.t("label.binary", language))
    backend.add_column(i18n.t("label.version", language))
    backend.add_column(i18n.t("label.flags", language))
    for name, version in profile.available_backends.items():
        flags = profile.backend_flags.get(name)
        backend.add_row(
            name,
            profile.backend_paths.get(name, i18n.t("label.not_found", language)),
            version or i18n.t("label.not_found", language),
            str(len(flags)) if flags is not None else i18n.t("label.unknown", language),
        )
    Console(legacy_windows=False).print(backend)
    for warning in localized_warnings:
        Console(legacy_windows=False).print(f"[yellow]- {warning}[/yellow]")
    if selected_models:
        models = Table(title=i18n.t("label.selected_models", language))
        models.add_column(i18n.t("label.service", language))
        models.add_column(i18n.t("label.model", language))
        models.add_column(i18n.t("label.languages", language))
        for item in selected_models:
            models.add_row(item["service"], item["model"], ",".join(item["languages"]))
        Console(legacy_windows=False).print(models)
    return 0


def _make_plan(args: argparse.Namespace) -> object:
    profile = detect_hardware()
    roles = [role.strip() for role in args.roles.split(",") if role.strip()]
    policy = Policy(roles=roles or ["chat", "code", "embed"], prefer=args.prefer,
                    max_context=args.context, budget_source=getattr(args, "budget", "total"),
                    parallel_slots=getattr(args, "parallel_slots", None),
                    lang=i18n.lang(),
                    languages=_parse_languages(getattr(args, "lang", None)))
    live = bench_overlay()
    args._telemetry_keys = len(live)
    cache = {**load_cache(), **live}
    return build_plan(profile, load_catalog(), policy, cache)


def _plan(args: argparse.Namespace) -> int:
    result = _make_plan(args)
    path = save_plan(result)
    if args.json:
        data = asdict(result)
        data["profile"]["warnings"] = _profile_warnings(result.profile, result.policy.lang)
        _print_json(data)
        return 0
    language = result.policy.lang
    table = Table(title=f"nmesh plan ({result.tier.value})")
    for column in (
        i18n.t("label.service", language), i18n.t("label.roles", language),
        i18n.t("label.model", language), i18n.t("label.backend", language),
        i18n.t("label.context", language), i18n.t("label.slots", language),
        i18n.t("label.gpu_layers", language), i18n.t("label.languages", language),
        i18n.t("label.tps", language),
    ):
        table.add_column(column)
    for service in result.services:
        table.add_row(service.name, ",".join(service.roles), service.model_id, service.backend,
                      str(service.context), str(service.memory.parallel_slots),
                      str(service.n_gpu_layers), ",".join(service.languages),
                      f"{service.decode_tps:.1f}")
    Console(legacy_windows=False).print(table)
    Console(legacy_windows=False).print(i18n.t("label.saved_to", language, path=path))
    if result.policy.budget_source == "free":
        Console(legacy_windows=False).print(i18n.t("label.free_budgets", language))
    if getattr(args, "_telemetry_keys", 0):
        Console(legacy_windows=False).print(
            i18n.t("label.telemetry_overlay", language, count=args._telemetry_keys)
        )
    for hint in result.install_hints:
        Console(legacy_windows=False).print(f"[yellow]{i18n.t('label.install', language, hint=hint)}[/yellow]")
    for warning in result.warnings:
        Console(legacy_windows=False).print(f"[yellow]{i18n.t('label.warning', language, warning=warning)}[/yellow]")
    if not getattr(args, "lang", None) and language != "en":
        Console(legacy_windows=False).print(i18n.t("hint.language", language, locale=language, language=language))
    if args.explain:
        memory = Table(title=i18n.t("label.memory", language))
        for column in (i18n.t("label.service", language),
                       i18n.t("label.weights", language),
                       i18n.t("label.kv", language),
                       i18n.t("label.gpu_cpu", language)):
            memory.add_column(column)
        for service in result.services:
            item = service.memory
            memory.add_row(service.name, _bytes(item.weight_bytes), _bytes(item.kv_cache_bytes),
                           f"{_bytes(item.gpu_bytes)} / {_bytes(item.cpu_bytes)}")
        Console(legacy_windows=False).print(memory)
    return 0


def _runtime(args: argparse.Namespace) -> int:
    if args.command == "serve":
        command = [sys.executable, "-m", "nmesh.gateway.server", "--port", str(args.port)]
        try:
            subprocess.run(command, check=False)
        except KeyboardInterrupt:
            pass
        return 0
    if args.command == "up":
        plan = load_plan()
        if plan is None:
            _plan(argparse.Namespace(roles="chat,code,embed", prefer="balanced",
                                     context=None, budget="total", json=False, explain=False,
                                     parallel_slots=None, lang=None))
            plan = load_plan()
        if plan is None:
            return 1
        if args.lang:
            plan = build_plan(
                detect_hardware(), load_catalog(),
                replace(plan.policy, lang=i18n.lang(),
                        languages=_parse_languages(args.lang)),
                {**load_cache(), **bench_overlay()},
            )
            save_plan(plan)
        cache = {**load_cache(), **bench_overlay()}
        result = runtime_up(
            plan, no_download=args.no_download, dry_run=args.dry_run,
            admit=not args.ignore_free_memory, bench_cache=cache,
        )
        if not args.dry_run:
            command = [sys.executable, "-m", "nmesh.gateway.server", "--port", str(args.port)]
            if args.detach:
                subprocess.Popen(command, start_new_session=True)
            else:
                try:
                    subprocess.run(command, check=False)
                except KeyboardInterrupt:
                    runtime_down()
    elif args.command == "down":
        result = runtime_down(foreign=True)
    else:
        result = runtime_status()
        try:
            with urllib.request.urlopen("http://127.0.0.1:18000/health", timeout=2):
                result.services.append({"service": "gateway", "port": 18000, "running": True})
                result.running = True
        except OSError:
            result.services.append({"service": "gateway", "port": 18000, "running": False})
    status_data = asdict(result)
    language = i18n.lang()
    if args.command == "status":
        status_data["telemetry"] = telemetry_summary()
    if args.json:
        _print_json(status_data)
    elif args.command == "up" and args.dry_run:
        for item in result.services:
            Console(legacy_windows=False).print(
                i18n.t("label.backend_detail", language, service=item["service"],
                       backend=item["backend"], model_ref=item["model_ref"],
                       port=item["port"], context=item["context"],
                       slots=item["parallel_slots"], layers=item["n_gpu_layers"],
                       argv=" ".join(str(x) for x in item["argv"]))
            )
    else:
        Console(legacy_windows=False).print(result)
        if args.command == "status":
            for item in result.services:
                if item.get("note"):
                    Console(legacy_windows=False).print(
                        i18n.t("label.acquisition_note", language,
                               service=item.get("service"), note=item["note"])
                    )
            telemetry = telemetry_summary()
            table = Table(title=i18n.t("label.telemetry", language))
            table.add_column(i18n.t("label.service", language))
            table.add_column(i18n.t("label.samples", language))
            table.add_column(i18n.t("label.decode_median", language))
            table.add_column(i18n.t("label.ttft_median", language))
            table.add_column(i18n.t("label.ttft_p95", language))
            table.add_column(i18n.t("label.total_median", language))
            for service, metrics in telemetry.items():
                table.add_row(
                    service,
                    str(int(metrics["samples"])),
                    f"{metrics.get('decode_tps_median', 0):.2f}",
                    f"{metrics.get('ttft_s_median', 0):.3f}",
                    f"{metrics.get('ttft_s_p95', 0):.3f}",
                    f"{metrics.get('total_s_median', 0):.3f}",
                )
            Console(legacy_windows=False).print(table)
    return 0


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
        print(i18n.t("err.gateway_reload", i18n.lang(), error=error), file=sys.stderr)
        return 1
    if args.json:
        _print_json(data)
    else:
        print(i18n.t("label.reloaded", i18n.lang(),
                     services=", ".join(data.get("services", [])),
                     created_at=data.get("created_at")))
    return 0


def _models(args: argparse.Namespace) -> int:
    models = load_catalog()
    if args.role:
        models = [model for model in models if args.role in model.roles]
    if args.json:
        _print_json([asdict(model) for model in models])
        return 0
    language = i18n.lang()
    table = Table(title=i18n.t("label.models", language))
    for column in ("ID", "Family", "Params", i18n.t("label.roles", language),
                   i18n.t("label.context", language), i18n.t("label.languages", language)):
        table.add_column(column)
    for model in models:
        table.add_row(model.id, model.family, str(model.params), ",".join(model.roles),
                      str(model.max_context), ",".join(model.languages))
    Console(legacy_windows=False).print(table)
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
        print(i18n.t("err.bench_up", i18n.lang()), file=sys.stderr)
        return 1
    base_url = "http://127.0.0.1:11434" if service.backend == "ollama" else (
        f"http://127.0.0.1:{service.port}"
    )
    try:
        measurement = measure(service, base_url, decode_tokens=args.tokens)
    except (OSError, RuntimeError) as error:
        print(i18n.t("err.bench_measure", i18n.lang(), error=error), file=sys.stderr)
        return 1
    cache = load_cache()
    key = benchmark_key(service.model_id, service.quant, service.backend,
                        plan.profile.gpus[0].name if plan.profile.gpus else "cpu",
                        service.n_gpu_layers)
    cache[key] = measurement.decode_tps
    save_cache(cache)
    result = {"key": key, "prefill_tokens": 512, "decode_tokens": args.tokens,
              "median_tps": cache[key], "prefill_tps": measurement.prefill_tps,
              "ttft_s": measurement.ttft_s, "approximate": measurement.approximate,
              "prompt_tokens": measurement.prompt_tokens}
    if args.json:
        _print_json(result)
    else:
        marker = "~" if measurement.approximate else ""
        language = i18n.lang()
        Console(legacy_windows=False).print("\n".join((
            i18n.t("label.median_decode", language, marker=marker,
                   value=measurement.decode_tps),
            i18n.t("label.prefill", language, marker=marker,
                   value=measurement.prefill_tps),
            i18n.t("label.ttft", language, marker=marker, value=measurement.ttft_s),
        )))
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
        output = i18n.t("err.gateway_unavailable", i18n.lang(), error=error)
    print(output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    _configure_output()
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
    plan.add_argument("--lang")
    up_parser = sub.add_parser("up")
    up_parser.add_argument("--json", action="store_true")
    up_parser.add_argument("--dry-run", action="store_true")
    up_parser.add_argument("--no-download", action="store_true")
    up_parser.add_argument("--detach", action="store_true")
    up_parser.add_argument("--port", type=int, default=18000)
    up_parser.add_argument("--ignore-free-memory", action="store_true")
    up_parser.add_argument("--lang")
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
            print(i18n.t("err.bench_up", i18n.lang()), file=sys.stderr)
            return 1
        base_url = "http://127.0.0.1:11434" if service.backend == "ollama" else (
            f"http://127.0.0.1:{service.port}"
        )
        best: tuple[int, int, float] | None = None
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
                    print(i18n.t("err.autotune_measure", i18n.lang(), error=error),
                          file=sys.stderr)
                    runtime_down()
                    return 1
                if best is None or result.decode_tps > best[2]:
                    best = (context, layers, result.decode_tps)
        if best is None:
            return 1
        tuned = replace(service, context=best[0], n_gpu_layers=best[1])
        save_plan(replace(plan, services=[
            tuned if item.name == service.name else item for item in plan.services
        ]))
        result = {"service": service.name, "context": best[0], "n_gpu_layers": best[1],
                  "decode_tps": best[2]}
        _print_json(result) if args.json else Console(legacy_windows=False).print(result)
        return 0
    if args.command == "run":
        return _run_prompt(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
