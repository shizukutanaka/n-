from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path

from rich.console import Console
from rich.table import Table

from nmesh import i18n
from nmesh.artifact import service_fingerprint
from nmesh.bench import benchmark_key, load_cache, measure, save_cache
from nmesh.catalog import load_catalog
from nmesh.eval import (
    CATEGORIES,
    EXTENDED_TASKS,
    SUITES,
    EvalRun,
    EvalSummary,
    load_eval_cache,
    save_eval,
)
from nmesh.eval import run as eval_run
from nmesh.eval.cache import EvalRecord
from nmesh.eval.stats import min_resolvable_difference, wilson_interval
from nmesh.paths import nmesh_home
from nmesh.planner import (
    PlannedService,
    Policy,
    build_plan,
    free_budgets,
    load_plan,
    save_plan,
)
from nmesh.probe import HardwareProfile, detect_hardware, profile_from_dict
from nmesh.runtime import RuntimeStatus, clear_gateway, disarm_atexit, record_gateway
from nmesh.runtime import down as runtime_down
from nmesh.runtime import status as runtime_status
from nmesh.runtime import up as runtime_up
from nmesh.runtime.service_unit import launcher_script, service_unit
from nmesh.telemetry import bench_overlay, overlay_report
from nmesh.telemetry import summary as telemetry_summary


def _console() -> Console:
    return Console(legacy_windows=False)


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


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer of at least 1") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, default=str))


def _profile_warnings(profile: HardwareProfile, language: str) -> list[str]:
    params = profile.warning_params
    return [
        i18n.t(warning, language, **(params[index] if index < len(params) else {}))
        for index, warning in enumerate(profile.warnings)
    ]


def _load_profile(path: str) -> HardwareProfile:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("profile must be an object")
        return profile_from_dict(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(str(error)) from error


def _doctor(as_json: bool, profile_path: str | None = None) -> int:
    try:
        profile = _load_profile(profile_path) if profile_path else detect_hardware()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        key = "err.profile_load" if profile_path else "err.doctor"
        print(i18n.t(key, i18n.lang(), error=error), file=sys.stderr)
        return 1
    language = i18n.lang()
    free_vram, free_ram = free_budgets(profile)
    selected = None if profile_path else load_plan()
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
        if profile_path:
            data["simulated"] = True
        _print_json(data)
        return 0
    if profile_path:
        _console().print(f"[yellow]{i18n.t('warn.simulated_profile', language)}[/yellow]")
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
            f"{i18n.t('label.free', language)} "
            f"({i18n.t('label.vram_source', language)}: {gpu.vram_source})",
        )
    table.add_row("Free budget VRAM", _bytes(free_vram))
    table.add_row("Free budget RAM", _bytes(free_ram))
    _console().print(table)
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
    _console().print(backend)
    for warning in localized_warnings:
        _console().print(f"[yellow]- {warning}[/yellow]")
    if selected_models:
        models = Table(title=i18n.t("label.selected_models", language))
        models.add_column(i18n.t("label.service", language))
        models.add_column(i18n.t("label.model", language))
        models.add_column(i18n.t("label.languages", language))
        for item in selected_models:
            models.add_row(item["service"], item["model"], ",".join(item["languages"]))
        _console().print(models)
    return 0


def _make_plan(args: argparse.Namespace) -> object:
    profile = _load_profile(args.profile) if getattr(args, "profile", None) else detect_hardware()
    args._simulated = bool(getattr(args, "profile", None))
    roles = [role.strip() for role in args.roles.split(",") if role.strip()]
    policy = Policy(roles=roles or ["chat", "code", "embed"], prefer=args.prefer,
                    max_context=args.context, budget_source=getattr(args, "budget", "total"),
                    parallel_slots=getattr(args, "parallel_slots", None),
                    lang=i18n.lang(),
                    languages=_parse_languages(getattr(args, "lang", None)))
    live, skipped = overlay_report()
    args._telemetry_keys = len(live)
    args._telemetry_under_load = skipped
    cache = {**load_cache(), **live}
    return build_plan(profile, load_catalog(), policy, cache, _eval_rates())


def _eval_rates() -> dict[tuple[str, str, str], EvalSummary]:
    latest: dict[tuple[str, str, str], tuple[float, EvalSummary]] = {}
    for record in load_eval_cache().values():
        key = (record.model_id, record.quant, record.backend)
        previous = latest.get(key)
        if previous is None or record.at > previous[0]:
            latest[key] = (
                record.at,
                EvalSummary(
                    record.pass_rate,
                    record.passed,
                    record.n_tasks,
                    record.task_results,
                ),
            )
    return {key: value for key, (_, value) in latest.items()}


def _eval_divergence(
    result: EvalRun, records: Mapping[str, EvalRecord],
) -> list[dict[str, object]]:
    current = {outcome.id: outcome.passed for outcome in result.outcomes}
    divergence: list[dict[str, object]] = []
    for record in records.values():
        if (
            record.model_id != result.model_id
            or (record.quant, record.backend) == (result.quant, result.backend)
            or not record.task_results
        ):
            continue
        comparable = sorted(set(record.task_results) & set(current))
        disagreeing = sorted(
            task_id for task_id in comparable
            if record.task_results[task_id] != current[task_id]
        )
        if not comparable:
            continue
        divergence.append({
            "config": f"{record.quant}|{record.backend}",
            "artifact": record.artifact or None,
            "pass_rate": record.pass_rate,
            "compared": len(comparable),
            "disagreeing": disagreeing,
        })
    return divergence


def _plan(args: argparse.Namespace) -> int:
    try:
        result = _make_plan(args)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        key = "err.profile_load" if getattr(args, "profile", None) else "err.plan"
        print(i18n.t(key, i18n.lang(), error=error), file=sys.stderr)
        return 1
    if not result.services or not result.runnable:
        print(i18n.t("err.plan_empty", i18n.lang()), file=sys.stderr)
        return 1
    path = None
    if not getattr(args, "_simulated", False):
        try:
            path = save_plan(result)
        except OSError as error:
            print(i18n.t("err.plan_save", i18n.lang(), error=error), file=sys.stderr)
            return 1
    if args.json:
        data = asdict(result)
        data["profile"]["warnings"] = _profile_warnings(result.profile, result.policy.lang)
        if getattr(args, "_simulated", False):
            data["simulated"] = True
        _print_json(data)
        return 0
    language = result.policy.lang
    if getattr(args, "_simulated", False):
        _console().print(f"[yellow]{i18n.t('warn.simulated_profile', language)}[/yellow]")
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
                      "-" if service.n_gpu_layers is None else str(service.n_gpu_layers),
                      ",".join(service.languages),
                      f"{service.decode_tps:.1f}")
    _console().print(table)
    if path is not None:
        _console().print(i18n.t("label.saved_to", language, path=path))
    if result.policy.budget_source == "free":
        _console().print(i18n.t("label.free_budgets", language))
    if getattr(args, "_telemetry_keys", 0):
        _console().print(
            i18n.t("label.telemetry_overlay", language, count=args._telemetry_keys)
        )
    if getattr(args, "_telemetry_under_load", 0):
        _console().print(
            i18n.t(
                "label.telemetry_under_load",
                language,
                count=getattr(args, "_telemetry_under_load", 0),
            )
        )
    for hint in result.install_hints:
        _console().print(f"[yellow]{i18n.t('label.install', language, hint=hint)}[/yellow]")
    for warning in result.warnings:
        _console().print(f"[yellow]{i18n.t('label.warning', language, warning=warning)}[/yellow]")
    if not getattr(args, "lang", None) and language != "en":
        _console().print(i18n.t("hint.language", language, language=language))
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
        _console().print(memory)
    return 0


def _runtime(args: argparse.Namespace) -> int:
    exit_code = 0
    if args.command == "serve":
        try:
            process, _ = _launch_gateway(args.port, detach=False)
        except OSError as error:
            print(i18n.t("err.gateway_start", i18n.lang(), error=error), file=sys.stderr)
            return 1
        try:
            exit_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            exit_code = 1
        finally:
            clear_gateway(process.pid)
        return 0 if exit_code == 0 else 1
    if args.command == "up":
        gateway_log: Path | None = None
        plan = load_plan()
        if plan is None:
            if _plan(argparse.Namespace(
                roles="chat,code,embed", prefer="balanced", context=None,
                budget="total", parallel_slots=None, json=False, explain=False,
                lang=None,
            )) != 0:
                return 1
            plan = load_plan()
        if plan is None or not plan.services or not plan.runnable:
            return 1
        if getattr(args, "lang", None):
            plan = build_plan(
                detect_hardware(), load_catalog(),
                replace(plan.policy, lang=i18n.lang(),
                        languages=_parse_languages(args.lang)),
                {**load_cache(), **bench_overlay()},
                _eval_rates(),
            )
            save_plan(plan)
        cache = {**load_cache(), **bench_overlay()}
        try:
            result = runtime_up(
                plan, no_download=args.no_download, dry_run=args.dry_run,
                admit=not args.ignore_free_memory, bench_cache=cache,
            )
        except (OSError, RuntimeError) as error:
            print(i18n.t("err.up", i18n.lang(), error=error), file=sys.stderr)
            return 1
        exit_code = 0
        if not args.dry_run:
            try:
                process, log_path = _launch_gateway(args.port, detach=args.detach)
            except OSError as error:
                runtime_down()
                print(i18n.t("err.gateway_start", i18n.lang(), error=error), file=sys.stderr)
                return 1
            if args.detach:
                gateway_log = log_path
                disarm_atexit()
                if not _wait_gateway(args.port, process):
                    clear_gateway(process.pid)
                    process.terminate()
                    print(
                        i18n.t("err.gateway_not_ready", i18n.lang(), path=log_path),
                        file=sys.stderr,
                    )
                    runtime_down()
                    return 1
                result = runtime_status()
            else:
                try:
                    exit_code = process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    runtime_down()
                    exit_code = 1
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
    language = i18n.lang()
    if args.command == "status":
        status_data["telemetry"] = telemetry_summary()
    if args.json:
        _print_json(status_data)
    elif args.command == "up" and args.dry_run:
        for item in result.services:
            _console().print(
                i18n.t("label.backend_detail", language, service=item["service"],
                       backend=item["backend"], model_ref=item["model_ref"],
                       port=item["port"], context=item["context"],
                       slots=item["parallel_slots"], layers=item["n_gpu_layers"],
                       argv=" ".join(str(x) for x in item["argv"]))
            )
    else:
        _console().print(result)
        if args.command == "up" and args.detach and gateway_log is not None:
            _console().print(
                i18n.t("label.gateway_log", language, path=gateway_log)
            )
        if args.command == "status":
            for item in result.services:
                if item.get("note"):
                    _console().print(
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
            _console().print(table)
    return 0 if exit_code == 0 else 1


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
        log_path = nmesh_home() / "gateway.log"
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
    _console().print(table)
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
        measurement = measure(
            service, base_url, decode_tokens=args.tokens, runs=args.runs
        )
    except (OSError, RuntimeError) as error:
        print(i18n.t("err.bench_measure", i18n.lang(), error=error), file=sys.stderr)
        return 1
    cache = load_cache()
    key = benchmark_key(service.model_id, service.quant, service.backend,
                        plan.profile.gpus[0].name if plan.profile.gpus else "cpu",
                        service.n_gpu_layers)
    cache[key] = measurement.decode_tps
    try:
        save_cache(cache)
    except OSError as error:
        print(i18n.t("err.bench_save", i18n.lang(), error=error), file=sys.stderr)
        return 1
    decode_spread = (
        (measurement.decode_tps_max - measurement.decode_tps_min)
        / measurement.decode_tps
        if measurement.decode_tps else 0.0
    )
    result = {"key": key, "prefill_tokens": 512, "decode_tokens": args.tokens,
              "median_tps": cache[key], "prefill_tps": measurement.prefill_tps,
              "ttft_s": measurement.ttft_s, "approximate": measurement.approximate,
              "prompt_tokens": measurement.prompt_tokens,
              "prefill_source": measurement.prefill_source,
              "cached_prompt_tokens": measurement.cached_prompt_tokens,
              "runs": measurement.runs,
              "decode_tps_min": measurement.decode_tps_min,
              "decode_tps_max": measurement.decode_tps_max,
              "decode_spread": decode_spread}
    if args.json:
        _print_json(result)
    else:
        marker = "~" if measurement.approximate else ""
        prefill_marker = "~" if measurement.prefill_source != "timings" else ""
        language = i18n.lang()
        _console().print("\n".join((
            i18n.t("label.median_decode", language, marker=marker,
                   value=measurement.decode_tps),
            i18n.t("label.decode_range", language,
                   minimum=measurement.decode_tps_min,
                   maximum=measurement.decode_tps_max,
                   spread=decode_spread),
            i18n.t("label.prefill", language, marker=prefill_marker,
                   value=measurement.prefill_tps),
            i18n.t("label.ttft", language, marker=marker, value=measurement.ttft_s),
        )))
        if decode_spread > 0.25:
            _console().print(i18n.t(
                "warn.bench_reproducibility",
                language,
                minimum=measurement.decode_tps_min,
                maximum=measurement.decode_tps_max,
                spread=decode_spread,
            ))
    return 0


def _eval(args: argparse.Namespace) -> int:
    plan = load_plan()
    if plan is None or not plan.services:
        return 1
    service = next((item for item in plan.services if item.name == args.service), None)
    if service is None:
        print(i18n.t("err.unknown_service", i18n.lang(), service=args.service),
              file=sys.stderr)
        return 1
    if not _service_running(service, runtime_status()):
        print(i18n.t("err.eval_up", i18n.lang()), file=sys.stderr)
        return 1
    requested = {item.strip() for item in args.categories.split(",") if item.strip()}
    base_tasks = SUITES[args.suite]
    tasks = (
        tuple(task for task in base_tasks if task.category in requested)
        if args.categories.strip()
        else ()
    )
    if not tasks:
        print(i18n.t("err.eval_categories", i18n.lang()), file=sys.stderr)
        return 1
    base_url = "http://127.0.0.1:11434" if service.backend == "ollama" else (
        f"http://127.0.0.1:{service.port}"
    )
    try:
        result = eval_run(tasks, base_url, service.model_ref)
    except RuntimeError as error:
        print(i18n.t("err.eval_run", i18n.lang(), error=error), file=sys.stderr)
        return 1
    result = replace(
        result,
        model_id=service.model_id,
        quant=service.quant,
        backend=service.backend,
        artifact=service_fingerprint(service.backend, service.model_ref) or "",
    )
    cached = load_eval_cache()
    previous = cached.get(f"{result.model_id}|{result.quant}|{result.backend}")
    artifact_warning = None
    if (
        previous is not None
        and previous.artifact
        and result.artifact
        and previous.artifact != result.artifact
    ):
        artifact_warning = i18n.t(
            "warn.eval_artifact_changed",
            i18n.lang(),
            model=result.model_id,
            quant=result.quant,
            backend=result.backend,
            previous=previous.artifact,
            current=result.artifact,
        )
    try:
        save_eval(result)
    except OSError as error:
        print(i18n.t("err.eval_save", i18n.lang(), error=error), file=sys.stderr)
        return 1
    divergence = _eval_divergence(result, cached)
    key = f"{result.model_id}|{result.quant}|{result.backend}"
    failed = [{"id": outcome.id, "output": outcome.output}
              for outcome in result.outcomes if not outcome.passed]
    language = i18n.lang()
    note = i18n.t("note.eval_scope", language, tasks=result.n_tasks)
    pass_rate_ci = wilson_interval(result.passed, result.n_tasks)
    minimum_difference = min_resolvable_difference(result.n_tasks)
    uncertainty_note = i18n.t(
        "note.eval_uncertainty",
        language,
        lo=pass_rate_ci[0],
        hi=pass_rate_ci[1],
        tasks=result.n_tasks,
        minimum=minimum_difference,
    )
    suite_upgrade_note = None
    if args.suite != "extended" and result.n_tasks < len(EXTENDED_TASKS):
        suite_upgrade_note = i18n.t(
            "note.eval_suite_upgrade",
            language,
            tasks=len(EXTENDED_TASKS),
            minimum=min_resolvable_difference(len(EXTENDED_TASKS)),
        )
    config_note = i18n.t(
        "note.eval_config",
        language,
        model=result.model_id,
        quant=result.quant,
        backend=result.backend,
    )
    output = {
        "key": key,
        "model_id": result.model_id,
        "quant": result.quant,
        "backend": result.backend,
        "suite": args.suite,
        "artifact": result.artifact or None,
        "n_tasks": result.n_tasks,
        "passed": result.passed,
        "pass_rate": result.pass_rate,
        "pass_rate_ci": list(pass_rate_ci),
        "min_resolvable_difference": minimum_difference,
        "by_category": result.by_category,
        "failed": failed,
        "note": note,
        "uncertainty_note": uncertainty_note,
        "suite_upgrade_note": suite_upgrade_note,
        "config_note": config_note,
        "divergence": divergence,
        "artifact_warning": artifact_warning,
    }
    if args.json:
        _print_json(output)
        return 0
    table = Table(title=i18n.t("label.eval_title", language))
    for column in (
        i18n.t("label.eval_category", language),
        i18n.t("label.eval_passed", language),
        i18n.t("label.eval_total", language),
        i18n.t("label.eval_pass_rate", language),
    ):
        table.add_column(column)
    for category, rate in result.by_category.items():
        category_outcomes = [item for item in result.outcomes if item.category == category]
        category_passed = sum(item.passed for item in category_outcomes)
        table.add_row(category, str(category_passed), str(len(category_outcomes)), f"{rate:.1%}")
    _console().print(table)
    _console().print(note)
    _console().print(uncertainty_note)
    if suite_upgrade_note is not None:
        _console().print(suite_upgrade_note)
    _console().print(config_note)
    if artifact_warning is not None:
        _console().print(artifact_warning)
    for item in divergence:
        ids = ", ".join(item["disagreeing"]) or "-"
        _console().print(i18n.t(
            "note.eval_divergence",
            language,
            config=item["config"],
            other_rate=f"{item['pass_rate']:.1%}",
            rate=f"{result.pass_rate:.1%}",
            count=len(item["disagreeing"]),
            compared=item["compared"],
            ids=ids,
        ))
    _console().print(i18n.t(
        "label.eval_overall", language, passed=result.passed, total=result.n_tasks,
        rate=result.pass_rate,
    ))
    failed_ids = ", ".join(item["id"] for item in failed) or "-"
    _console().print(i18n.t("label.eval_failed", language, ids=failed_ids))
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
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    _configure_output()
    parser = argparse.ArgumentParser(prog="nmesh")
    parser.add_argument("--dry-run", action="store_true", dest="global_dry_run")
    parser.add_argument("--json", action="store_true", dest="global_json")
    sub = parser.add_subparsers(dest="command")
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--profile")
    plan = sub.add_parser("plan")
    plan.add_argument("--json", action="store_true")
    plan.add_argument("--explain", action="store_true")
    plan.add_argument("--prefer", choices=("quality", "speed", "balanced"), default="balanced")
    plan.add_argument("--roles", default="chat,code,embed")
    plan.add_argument("--context", type=int)
    plan.add_argument("--budget", choices=("total", "free"), default="total")
    plan.add_argument("--parallel-slots", type=int)
    plan.add_argument("--lang")
    plan.add_argument("--profile")
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
    bench_parser.add_argument("--runs", type=_positive_int, default=3)
    bench_parser.add_argument("--json", action="store_true")
    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("--service", default="chat")
    eval_parser.add_argument("--json", action="store_true")
    eval_parser.add_argument("--categories", default=",".join(CATEGORIES))
    eval_parser.add_argument("--suite", choices=("core", "extended"), default="core")
    auto = sub.add_parser("autotune")
    auto.add_argument("--json", action="store_true")
    autostart = sub.add_parser("autostart")
    autostart.add_argument("--port", type=int, default=18000)
    autostart.add_argument("--install", action="store_true")
    autostart.add_argument("--json", action="store_true")
    models = sub.add_parser("models")
    models.add_argument("--role")
    models.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.global_json and hasattr(args, "json"):
        args.json = True
    if args.command == "doctor":
        return _doctor(args.json, args.profile)
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
    if args.command == "eval":
        return _eval(args)
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
                    print(i18n.t("err.autotune_measure", i18n.lang(), error=error),
                          file=sys.stderr)
                    runtime_down()
                    try:
                        runtime_up(plan, no_download=True)
                    except (OSError, RuntimeError) as restore_error:
                        print(
                            i18n.t("err.autotune_restore", i18n.lang(), error=restore_error),
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
                    i18n.t("err.autotune_restore", i18n.lang(), error=restore_error),
                    file=sys.stderr,
                )
            print(
                i18n.t("err.autotune_winning", i18n.lang(), error=error),
                file=sys.stderr,
            )
            return 1
        result = {"service": service.name, "context": best[0], "n_gpu_layers": best[1],
                  "decode_tps": best[2]}
        _print_json(result) if args.json else _console().print(result)
        return 0
    if args.command == "autostart":
        filename, text, install_command = service_unit(args.port)
        launcher_filename, launcher_text = launcher_script(args.port)
        home = nmesh_home()
        launcher_path = home / launcher_filename
        env_path = home / "gateway.env"
        installed = False
        if args.install:
            home.mkdir(parents=True, exist_ok=True)
            launcher_path.write_text(launcher_text, encoding="utf-8", newline="")
            if os.name != "nt":
                launcher_path.chmod(0o700)
            if not env_path.exists():
                api_key = os.environ.get("NMESH_API_KEY", "")
                env_path.write_text(
                    f"NMESH_API_KEY={api_key}\n"
                    "# NMESH_HOME is set by the launcher to its own directory.\n",
                    encoding="utf-8",
                    newline="",
                )
                if os.name != "nt":
                    env_path.chmod(0o600)
            installed = True
        limitations = []
        if os.name == "nt":
            limitations.append(i18n.t("autostart.windows_limitations", i18n.lang()))
        data = {
            "filename": filename,
            "text": text,
            "install_command": install_command,
            "launcher_path": str(launcher_path),
            "env_path": str(env_path),
            "installed": installed,
            "limitations": limitations,
        }
        if args.json:
            _print_json(data)
        else:
            print(f"Filename: {filename}\n\n{text}\nInstall with:\n{install_command}")
            if args.install:
                print(i18n.t("label.launcher_written", i18n.lang(), path=launcher_path))
                print(i18n.t("label.gateway_env", i18n.lang(), path=env_path))
            for limitation in limitations:
                print(i18n.t("label.autostart_limitation", i18n.lang(), text=limitation))
        return 0
    if args.command == "run":
        return _run_prompt(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
