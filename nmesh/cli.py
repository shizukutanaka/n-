from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError
from urllib.parse import quote

import httpx
import psutil
from rich.console import Console
from rich.table import Table

from nmesh import __version__, i18n
from nmesh.artifact import gguf_info, service_fingerprint
from nmesh.artifacts import load_cache as load_artifact_cache
from nmesh.bench import (
    BENCH_HARNESS_VERSION,
    EMBED_HARNESS_VERSION,
    EPOCH_HISTORY,
    MIN_DECODE_TOKENS,
    RETRIEVAL_HARNESS_VERSION,
    BenchRecord,
    EmbedRecord,
    EpochSample,
    RetrievalLimit,
    RetrievalRecord,
    baseline,
    benchmark_key,
    choose_reference_model,
    classify,
    demote_stale,
    find_reference_binary,
    load_cache,
    load_embed_cache,
    load_history,
    load_records,
    load_retrieval_cache,
    measure,
    measure_controlled,
    measure_embedding,
    measure_reference,
    measure_retrieval,
    measure_retrieval_chunk_arm,
    measure_retrieval_estimate,
    merge_measurement,
    prune_degraded,
    reference_id,
    retrieval_digest,
    save_embed,
    save_history,
    save_records,
    save_retrieval,
)
from nmesh.catalog import load_catalog
from nmesh.eval import (
    EXTENDED_CATEGORIES,
    EXTENDED_TASKS,
    SUITES,
    EvalRun,
    EvalSummary,
    load_eval_cache,
    needle_tasks,
    save_eval,
    suite_digest,
)
from nmesh.eval import run as eval_run
from nmesh.eval import select as eval_select
from nmesh.eval.cache import EvalRecord, eval_key
from nmesh.eval.context import (
    ContextRecord,
    FamilyResult,
    load_context_cache,
    save_context,
)
from nmesh.eval.select import (
    context_depth_evidence,
    effective_context_records,
    planner_eval_records,
    valid_eval_records,
)
from nmesh.eval.stats import (
    min_discordant_for_significance,
    min_resolvable_difference,
    wilson_interval,
)
from nmesh.evidence_inventory import collect_evidence
from nmesh.inventory import (
    FILE_TYPE_QUANT,
    default_stores,
)
from nmesh.inventory import (
    Artifact as InventoryArtifact,
)
from nmesh.inventory import (
    duplicates as inventory_duplicates,
)
from nmesh.inventory import (
    label_mismatch as inventory_label_mismatch,
)
from nmesh.inventory import (
    scan as scan_inventory,
)
from nmesh.inventory import (
    variants as inventory_variants,
)
from nmesh.orchestrate import (
    Endpoint,
    RoleIdentity,
    combine,
    decide,
    decide_cost,
    from_run,
)
from nmesh.orchestrate import (
    demote_stale as demote_delegation_stale,
)
from nmesh.orchestrate import (
    load_cache as load_delegation_cache,
)
from nmesh.orchestrate import (
    measure as orchestrate_measure,
)
from nmesh.orchestrate import (
    save as save_delegation,
)
from nmesh.orchestrate import (
    save_all as save_all_delegation,
)
from nmesh.paths import is_windows, nmesh_home
from nmesh.planner import (
    Plan,
    PlannedService,
    Policy,
    build_plan,
    free_budgets,
    load_plan,
    save_plan,
)
from nmesh.probe import HardwareProfile, detect_hardware, profile_from_dict
from nmesh.runtime import (
    RuntimeStatus,
    clear_gateway,
    disarm_atexit,
    gateway_health,
    gateway_listener_pid,
    record_gateway,
)
from nmesh.runtime import down as runtime_down
from nmesh.runtime import engine as engine_runtime
from nmesh.runtime import status as runtime_status
from nmesh.runtime import up as runtime_up
from nmesh.runtime.acquisition import parse_label
from nmesh.runtime.logs import available as available_logs
from nmesh.runtime.logs import log_path
from nmesh.runtime.logs import rotate as rotate_log
from nmesh.runtime.logs import tail as tail_log
from nmesh.runtime.service_unit import (
    launcher_script,
    service_unit,
    unit_install_path,
    watch_unit,
)
from nmesh.runtime.supervisor import Supervisor
from nmesh.spec import (
    KIND_DRAFT,
    KIND_NGRAM,
    WORKLOADS,
    SpecConfig,
    engine_identity,
    from_arms,
    run_arm,
)
from nmesh.spec.record import decide as decide_spec
from nmesh.spec.record import demote_stale as demote_spec_stale
from nmesh.spec.record import load_cache as load_spec_cache
from nmesh.spec.record import save as save_spec
from nmesh.spec.record import save_all as save_all_spec
from nmesh.telemetry import (
    COMPARABLE_PROMPT_TOKENS,
    bench_overlay,
    overlay_report,
)
from nmesh.telemetry import summary as telemetry_summary
from nmesh.watch import extract as extract_mentions
from nmesh.watch import fetch_qiita, fetch_x, fetch_zenn
from nmesh.watch.draft import write_draft
from nmesh.watch.sources import SourceItem, SourceStatus
from nmesh.watch.state import WatchState, load_state, now_iso, save_state
from nmesh.watch.verify import Finding, caps_available, verify

_CONTEXT_CATEGORIES = (
    "context.literal",
    "context.latent",
    "context.multi",
    "context.update",
)


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


def _parse_model_ids(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    result: list[str] = []
    for item in value.split(","):
        model_id = item.strip()
        if model_id and model_id.casefold() not in {entry.casefold() for entry in result}:
            result.append(model_id)
    return tuple(result)


def _candidate_fit(finding: Finding, budget_bytes: float) -> str:
    verified = finding.verified
    weight_sets = verified.get("weight_sets")
    if not isinstance(weight_sets, dict) or not weight_sets:
        return "no_weights"
    if verified.get("gated"):
        return "gated"
    pipeline_tag = verified.get("pipeline_tag")
    if not isinstance(pipeline_tag, str) or not pipeline_tag:
        return "role_unknown"
    if pipeline_tag not in {"text-generation", "text2text-generation"}:
        return "not_text"
    smallest = verified.get("smallest_weight_bytes")
    if (
        isinstance(smallest, (int, float))
        and not isinstance(smallest, bool)
        and smallest > budget_bytes
    ):
        return "too_large"
    return "fits"


def _watch_budget() -> tuple[float, str]:
    profile = detect_hardware()
    vram, ram = free_budgets(profile)
    return (
        (ram, "planner.free_budgets.ram_bytes")
        if not profile.gpus
        else (vram, "planner.free_budgets.vram_bytes")
    )


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


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive number") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer of at least 0") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
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
    selected_models: list[dict[str, str | list[str]]] = [
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
            languages = item["languages"]
            models.add_row(
                str(item["service"]),
                str(item["model"]),
                ",".join(languages) if isinstance(languages, list) else str(languages),
            )
        _console().print(models)
    return 0


def _argv_text(value: object) -> str:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"invalid argv: {value!r}")
    return " ".join(str(item) for item in value)


def _bench_cache(live: Mapping[str, float]) -> dict[object, float]:
    """Stored measurements with live telemetry taking precedence."""
    merged: dict[object, float] = {}
    merged.update(load_cache())
    merged.update(live)
    return merged


def _make_plan(args: argparse.Namespace) -> Plan:
    profile = _load_profile(args.profile) if getattr(args, "profile", None) else detect_hardware()
    simulated = bool(getattr(args, "profile", None))
    args._simulated = simulated
    roles_arg = getattr(args, "roles", None)
    roles = (
        [role.strip() for role in roles_arg.split(",") if role.strip()]
        if roles_arg is not None
        else []
    )
    policy = Policy(
        roles=roles or ["chat", "code", "embed"],
        roles_explicit=roles_arg is not None,
        prefer=args.prefer,
        max_context=args.context,
        budget_source=getattr(args, "budget", "total"),
        kv_quant=getattr(args, "kv_quant", "f16"),
        parallel_slots=getattr(args, "parallel_slots", None),
        lang=i18n.lang(),
        languages=_parse_languages(getattr(args, "lang", None)),
        model_ids=_parse_model_ids(getattr(args, "model", None)),
        eval_evidence=not getattr(args, "ignore_eval_evidence", False),
        spec=getattr(args, "spec", "none"),
        spec_draft=getattr(args, "spec_draft", ""),
        spec_n_max=getattr(args, "spec_n_max", 3),
        ignore_spec_evidence=getattr(args, "ignore_spec_evidence", False),
        sleep_idle_seconds=getattr(args, "sleep_idle_seconds", 0),
        cache_reuse=getattr(args, "cache_reuse", 0),
        context_shift=getattr(args, "context_shift", False),
    )
    # Throughput and capacity measurements describe the machine that ran them:
    # their cache key carries the GPU name and layer count but not the CPU, so
    # a simulated CPU placement would otherwise inherit this machine's tok/s
    # and its llama.cpp build's embedding limits. A simulated machine has no
    # measurements, so a simulated plan uses none. Model-level evidence (eval
    # quality, usable context depth, artifact sizes) holds on any machine and
    # stays in.
    live: dict[str, float] = {}
    cache: dict[object, float] = {}
    records: dict[str, BenchRecord] = {}
    embed_input_caps: dict[tuple[str, str, str], int] = {}
    embed_retrieval_limits: dict[tuple[str, str, str], RetrievalLimit] = {}
    embed_measured: set[tuple[str, str, str]] = set()
    args._telemetry_keys = 0
    args._telemetry_under_load = 0
    args._telemetry_off_reference = 0
    args._telemetry_unknown_depth = 0
    if not simulated:
        telemetry_report = overlay_report()
        live = telemetry_report.values
        args._telemetry_keys = len(live)
        args._telemetry_under_load = telemetry_report.under_load
        args._telemetry_off_reference = telemetry_report.off_reference
        args._telemetry_unknown_depth = telemetry_report.unknown_depth
        cache = _bench_cache(live)
        records = {
            key: value for key, value in load_records().items()
            if key not in live
        }
        embed_input_caps = _embed_context_caps()
        embed_retrieval_limits = _embed_retrieval_limits()
        embed_measured = _embed_measured_keys()
    eval_records = load_eval_cache()
    eval_depth_coverage, eval_depth_lost = _context_depth_maps()
    return build_plan(
        profile,
        load_catalog(),
        policy,
        cache,
        _eval_rates(eval_records),
        load_artifact_cache(),
        records,
        eval_depth_coverage=eval_depth_coverage,
        eval_depth_lost=eval_depth_lost,
        embed_input_caps=embed_input_caps,
        embed_retrieval_limits=embed_retrieval_limits,
        embed_measured=embed_measured,
    )


_eval_records = valid_eval_records
_context_records = effective_context_records
_context_evidence = context_depth_evidence
DepthEvidence = eval_select.DepthEvidence


def _eval_rates(
    records: Mapping[str, EvalRecord] | None = None,
) -> dict[tuple[str, str, str], EvalSummary]:
    latest = _eval_planner_records(records)
    return {
        key: EvalSummary(
            record.pass_rate,
            record.passed,
            record.n_tasks,
            record.task_results,
            record.suite,
            record.digest,
            record.reasoning_allowance,
            record.cache_prompt,
        )
        for key, record in latest.items()
    }


def _eval_planner_records(
    records: Mapping[str, EvalRecord] | None = None,
) -> dict[tuple[str, str, str], EvalRecord]:
    return planner_eval_records(
        records if records is not None else load_eval_cache()
    )


def _context_depth_maps() -> tuple[
    dict[tuple[str, str, str], int],
    dict[tuple[str, str, str], int],
]:
    evidence = _context_evidence(load_context_cache())
    return (
        {
            key: value.verified
            for key, value in evidence.items()
            if value.verified > 0
        },
        {
            key: value.lost
            for key, value in evidence.items()
            if value.lost > 0
        },
    )


def _embed_refused_tokens(record: EmbedRecord) -> int | None:
    """Return the smallest probe size the backend refused, if any."""
    refused = [
        tokens for tokens, was_refused in (
            (record.probe_tokens_small, record.refused_small),
            (record.probe_tokens_large, record.refused_large),
        )
        if was_refused
    ]
    return min(refused) if refused else None


def _embed_measured_keys() -> set[tuple[str, str, str]]:
    return {
        (
            record.model_id.casefold(),
            record.quant.casefold(),
            record.backend.casefold(),
        )
        for record in load_embed_cache().values()
        if record.harness == EMBED_HARNESS_VERSION
    }


def _embed_context_caps() -> dict[tuple[str, str, str], int]:
    latest: dict[tuple[str, str, str], EmbedRecord] = {}
    for record in load_embed_cache().values():
        if record.harness != EMBED_HARNESS_VERSION or record.cap is None:
            continue
        key = (
            record.model_id.casefold(),
            record.quant.casefold(),
            record.backend.casefold(),
        )
        previous = latest.get(key)
        if previous is None or record.at > previous.at:
            latest[key] = record
    return {
        key: record.cap
        for key, record in latest.items()
        if record.cap is not None
    }


def _embed_retrieval_limits() -> dict[tuple[str, str, str], RetrievalLimit]:
    latest: dict[tuple[str, str, str], RetrievalRecord] = {}
    digest = retrieval_digest()
    for record in load_retrieval_cache().values():
        if (
            record.harness != RETRIEVAL_HARNESS_VERSION
            or record.digest != digest
            or record.degraded_tokens is None
        ):
            continue
        key = (
            record.model_id.casefold(),
            record.quant.casefold(),
            record.backend.casefold(),
        )
        previous = latest.get(key)
        if previous is None or record.at > previous.at:
            latest[key] = record
    return {
        key: RetrievalLimit(
            degraded_tokens=record.degraded_tokens,
            chunk_tokens=(
                record.chunk.chunk_tokens
                if record.chunk is not None else None
            ),
            chunk_recovers=record.chunk_recovers,
            chunk_hits=record.chunk.hits if record.chunk is not None else None,
            chunk_trials=(
                record.chunk.trials if record.chunk is not None else None
            ),
            pool_recovers=record.pool_recovers,
            pool_hits=(
                record.chunk.pool_hits if record.chunk is not None else None
            ),
            pool_trials=(
                record.chunk.pool_trials if record.chunk is not None else None
            ),
        )
        for key, record in latest.items()
        if record.degraded_tokens is not None
    }


def _stale_grader_notes(records: Mapping[str, EvalRecord]) -> list[str]:
    _, stale = _eval_records(records)
    language = i18n.lang()
    return [
        i18n.t(
            "warn.eval_stale_grader",
            language,
            model=record.model_id,
            quant=record.quant,
            backend=record.backend,
            suite=record.suite,
        )
        for record in stale
    ]


@dataclass(frozen=True)
class _Divergence:
    """A stored run of the same model that disagrees with the current one."""

    config: str
    artifact: str | None
    pass_rate: float
    compared: int
    disagreeing: list[str]
    discordant_here: int
    discordant_there: int
    zero_power_families: list[str]


def _eval_divergence(
    result: EvalRun, records: Mapping[str, EvalRecord],
) -> list[_Divergence]:
    current = {outcome.id: outcome.passed for outcome in result.outcomes}
    divergence: list[_Divergence] = []
    for record in records.values():
        if (
            record.model_id != result.model_id
            or (record.quant, record.backend) == (result.quant, result.backend)
            or record.digest != result.digest
            or record.reasoning_allowance != result.reasoning_allowance
            or record.cache_prompt != result.cache_prompt
            or record.unscorable
            or result.unscorable
            or record.transport_errors
            or result.transport_errors
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
        discordant_here = sorted(
            task_id for task_id in comparable
            if current[task_id] and not record.task_results[task_id]
        )
        discordant_there = sorted(
            task_id for task_id in comparable
            if record.task_results[task_id] and not current[task_id]
        )
        disagreeing_families = {
            task_id.split(".")[0] for task_id in disagreeing
        }
        compared_families = {
            task_id.split(".")[0] for task_id in comparable
        }
        divergence.append(_Divergence(
            config=f"{record.quant}|{record.backend}",
            artifact=record.artifact or None,
            pass_rate=record.pass_rate,
            compared=len(comparable),
            disagreeing=disagreeing,
            discordant_here=len(discordant_here),
            discordant_there=len(discordant_there),
            zero_power_families=sorted(
                compared_families - disagreeing_families
            ),
        ))
    return divergence


def _plan(args: argparse.Namespace) -> int:
    try:
        result = _make_plan(args)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        key = "err.profile_load" if getattr(args, "profile", None) else "err.plan"
        print(i18n.t(key, i18n.lang(), error=error), file=sys.stderr)
        return 1
    if not result.services or not result.runnable:
        _print_plan_failure(result, json_output=args.json)
        return 1
    path = None
    if not getattr(args, "_simulated", False):
        try:
            path = save_plan(result)
        except OSError as error:
            print(i18n.t("err.plan_save", i18n.lang(), error=error), file=sys.stderr)
            return 1
    if args.json:
        data = _plan_json_data(result)
        if getattr(args, "_simulated", False):
            data["simulated"] = True
        _print_json(data)
        return 0
    language = result.policy.lang
    if getattr(args, "_simulated", False):
        _console().print(f"[yellow]{i18n.t('warn.simulated_profile', language)}[/yellow]")
    _render_plan(result)
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
    if getattr(args, "_telemetry_off_reference", 0):
        _console().print(
            i18n.t(
                "label.telemetry_off_reference",
                language,
                count=getattr(args, "_telemetry_off_reference", 0),
                tokens=COMPARABLE_PROMPT_TOKENS,
            )
        )
    if getattr(args, "_telemetry_unknown_depth", 0):
        _console().print(
            i18n.t(
                "label.telemetry_unknown_depth",
                language,
                count=getattr(args, "_telemetry_unknown_depth", 0),
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
            memory.add_row(
                service.name,
                _bytes(item.weight_bytes),
                f"{_bytes(item.kv_cache_bytes)} ({service.kv_quant})",
                f"{_bytes(item.gpu_bytes)} / {_bytes(item.cpu_bytes)}",
            )
        _console().print(memory)
    return 0


def _render_plan(result: Plan) -> None:
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
                      "-" if service.n_gpu_layers is None else str(service.n_gpu_layers),
                      ",".join(service.languages),
                      "—" if service.decode_tps is None else f"{service.decode_tps:.1f}")
    _console().print(table)


def _up_plan_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        roles=getattr(args, "roles", None),
        prefer="balanced",
        context=None,
        budget="total",
        kv_quant=getattr(args, "kv_quant", "f16"),
        parallel_slots=None,
        json=False,
        explain=False,
        lang=getattr(args, "lang", None),
        model=getattr(args, "model", None),
        ignore_eval_evidence=getattr(args, "ignore_eval_evidence", False),
        spec=getattr(args, "spec", "none"),
        spec_draft=getattr(args, "spec_draft", ""),
        spec_n_max=getattr(args, "spec_n_max", 3),
        ignore_spec_evidence=getattr(args, "ignore_spec_evidence", False),
        sleep_idle_seconds=getattr(args, "sleep_idle_seconds", 0),
        cache_reuse=getattr(args, "cache_reuse", 0),
        context_shift=getattr(args, "context_shift", False),
        profile=getattr(args, "profile", None),
    )


def _plan_json_data(plan: Plan) -> dict[str, object]:
    data = asdict(plan)
    data["profile"]["warnings"] = _profile_warnings(plan.profile, plan.policy.lang)
    return data


def _print_plan_failure(plan: Plan, *, json_output: bool = False) -> None:
    print(i18n.t("err.plan_empty", i18n.lang()), file=sys.stderr)
    for warning in plan.warnings:
        print(warning, file=sys.stderr)
    for hint in plan.install_hints:
        print(hint, file=sys.stderr)
    if json_output:
        _print_json(_plan_json_data(plan))


def _ensure_runnable_plan(args: argparse.Namespace) -> Plan | None:
    plan = load_plan()
    if plan is not None and plan.services and plan.runnable:
        return plan
    plan_args = _up_plan_args(args)
    try:
        plan = _make_plan(plan_args)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        key = "err.profile_load" if getattr(plan_args, "profile", None) else "err.plan"
        print(i18n.t(key, i18n.lang(), error=error), file=sys.stderr)
        return None
    simulated = bool(
        getattr(args, "profile", None)
        or getattr(args, "_simulated", False)
        or getattr(plan_args, "_simulated", False)
    )
    if (
        plan.missing_backends == ["llamacpp"]
        and not getattr(args, "no_download", False)
        and not getattr(args, "dry_run", False)
        and not simulated
    ):
        try:
            installed, _ = engine_runtime.install(variant="auto")
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            tarfile.TarError,
            zipfile.BadZipFile,
        ) as error:
            print(i18n.t("err.up", i18n.lang(), error=error), file=sys.stderr)
            _print_plan_failure(plan, json_output=getattr(args, "json", False))
            return None
        _console().print(
            i18n.t(
                "info.engine_autoinstall",
                i18n.lang(),
                tag=installed.tag,
                variant=installed.variant,
            )
        )
        try:
            plan = _make_plan(plan_args)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            key = "err.profile_load" if getattr(plan_args, "profile", None) else "err.plan"
            print(i18n.t(key, i18n.lang(), error=error), file=sys.stderr)
            return None
    if not plan.services or not plan.runnable:
        _print_plan_failure(plan, json_output=getattr(args, "json", False))
        return None
    if not getattr(args, "json", False):
        _render_plan(plan)
    if not simulated:
        try:
            save_plan(plan)
        except OSError as error:
            print(i18n.t("err.plan_save", i18n.lang(), error=error), file=sys.stderr)
            return None
    return plan


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
        plan = _ensure_runnable_plan(args)
        if plan is None:
            return 1
        requested_model_ids = _parse_model_ids(getattr(args, "model", None))
        ignore_eval_evidence = bool(getattr(args, "ignore_eval_evidence", False))
        if getattr(args, "lang", None) or requested_model_ids or ignore_eval_evidence:
            policy = plan.policy
            if getattr(args, "lang", None):
                policy = replace(
                    policy,
                    lang=i18n.lang(),
                    languages=_parse_languages(args.lang),
                )
            if requested_model_ids:
                policy = replace(policy, model_ids=requested_model_ids)
            if ignore_eval_evidence:
                policy = replace(policy, eval_evidence=False)
            live = bench_overlay()
            records = {
                key: value for key, value in load_records().items()
                if key not in live
            }
            eval_records = load_eval_cache()
            eval_depth_coverage, eval_depth_lost = _context_depth_maps()
            embed_input_caps = _embed_context_caps()
            embed_retrieval_limits = _embed_retrieval_limits()
            plan = build_plan(
                detect_hardware(), load_catalog(),
                policy,
                _bench_cache(live),
                _eval_rates(eval_records),
                load_artifact_cache(),
                records,
                eval_depth_coverage=eval_depth_coverage,
                eval_depth_lost=eval_depth_lost,
                embed_input_caps=embed_input_caps,
                embed_retrieval_limits=embed_retrieval_limits,
                embed_measured=_embed_measured_keys(),
            )
            save_plan(plan)
        cache = _bench_cache(bench_overlay())
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
        result = runtime_down(foreign=True, gateway_port=args.port)
    else:
        result = runtime_status()
        gateway = next(
            (item for item in result.services if item.get("service") == "gateway"),
            None,
        )
        recorded_port = gateway.get("port") if gateway else None
        try:
            port = (
                int(recorded_port)
                if isinstance(recorded_port, (int, float, str))
                else args.port
            )
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
                result.services.append(
                    {"service": "gateway", "port": args.port, "running": False}
                )
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
                       argv=_argv_text(item["argv"]))
            )
    else:
        _console().print(result)
        for warning in result.warnings:
            _console().print(warning)
        if args.command == "status":
            for item in result.services:
                if item.get("idle"):
                    _console().print(
                        i18n.t("label.service_idle", language,
                               service=item.get("service"))
                    )
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


def _logs(args: argparse.Namespace) -> int:
    language = i18n.lang()
    if args.service is None:
        services = available_logs()
        if args.json:
            _print_json({"services": services})
        else:
            for service in services:
                _console().print(service)
        return 0
    try:
        path = log_path(args.service)
    except ValueError:
        path = None
    if path is None or not path.is_file():
        print(
            i18n.t("err.no_log", language, service=args.service),
            file=sys.stderr,
        )
        return 1
    lines = tail_log(args.service, args.lines)
    if args.json:
        _print_json({
            "service": args.service,
            "path": str(path),
            "lines": lines,
        })
    else:
        for line in lines:
            _console().print(line)
    return 0


def _unload(args: argparse.Namespace) -> int:
    path = "/admin/unload"
    if args.service is not None:
        path += f"/{quote(args.service, safe='')}"
    request = urllib.request.Request(
        f"http://127.0.0.1:{args.port}{path}",
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode())
    except HTTPError as error:
        if error.code == 404 and args.service is not None:
            print(i18n.t("err.unknown_service", i18n.lang(), service=args.service),
                  file=sys.stderr)
        else:
            print(i18n.t("err.gateway_unload", i18n.lang(), error=error), file=sys.stderr)
        return 1
    except (OSError, json.JSONDecodeError) as error:
        print(i18n.t("err.gateway_unload", i18n.lang(), error=error), file=sys.stderr)
        return 1
    unloaded = data.get("unloaded", [])
    if args.service is not None and not unloaded:
        result = next(
            (
                item for item in data.get("results", [])
                if isinstance(item, dict) and item.get("service") == args.service
            ),
            {},
        )
        reason = result.get("reason")
        reason_key = {
            "not_running": "err.unload_not_running",
            "idle": "err.unload_idle",
            "shared": "err.unload_shared",
            "external": "err.unload_external",
            "not_owned": "err.unload_not_owned",
        }.get(str(reason), "err.unload_unknown")
        print(i18n.t(reason_key, i18n.lang(), service=args.service), file=sys.stderr)
        return 1
    if args.json:
        _print_json(data)
    else:
        print(i18n.t("label.unloaded", i18n.lang(),
                     services=", ".join(unloaded) if unloaded else
                     i18n.t("label.none", i18n.lang())))
    return 0


class _GatewayProcess(Protocol):
    """The parts of a gateway handle the CLI drives, adopted or spawned."""

    @property
    def pid(self) -> int: ...

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...


class _AdoptedGateway:
    """Process-like handle for a gateway that was already serving the port."""

    def __init__(self, pid: int) -> None:
        self.pid = pid

    def poll(self) -> int | None:
        try:
            return None if psutil.Process(self.pid).is_running() else 0
        except psutil.Error:
            return 0

    def wait(self, timeout: float | None = None) -> int:
        try:
            psutil.Process(self.pid).wait(timeout=timeout)
        except psutil.TimeoutExpired:
            raise subprocess.TimeoutExpired(
                str(self.pid), timeout if timeout is not None else 0.0,
            ) from None
        except psutil.Error:
            pass
        return 0

    def terminate(self) -> None:
        try:
            psutil.Process(self.pid).terminate()
        except psutil.Error:
            pass


def _launch_gateway(
    port: int, detach: bool,
) -> tuple[subprocess.Popen[bytes] | _AdoptedGateway, Path | None]:
    """Serve on ``port``, adopting a gateway that already answers there."""
    if gateway_health(port):
        adopted = gateway_listener_pid(port)
        if adopted is None:
            raise OSError(
                f"port {port} already answers /health but no nmesh gateway owns it"
            )
        record_gateway(adopted, port)
        return _AdoptedGateway(adopted), None
    command = [sys.executable, "-m", "nmesh.gateway.server", "--port", str(port)]
    if not detach:
        process = subprocess.Popen(command)
        record_gateway(process.pid, port)
        return process, None
    log_path = nmesh_home() / "gateway.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rotate_log(log_path)
    with log_path.open("ab") as log:
        if is_windows():
            # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
            process = subprocess.Popen(
                command, stdout=log, stderr=log, close_fds=True,
                creationflags=0x00000008 | 0x00000200,
            )
        else:
            process = subprocess.Popen(
                command, stdout=log, stderr=log, close_fds=True,
                start_new_session=True,
            )
    record_gateway(process.pid, port)
    return process, log_path


def _wait_gateway(port: int, process: _GatewayProcess, timeout: float = 20.0) -> bool:
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


def _inventory_artifact_payload(
    artifact: InventoryArtifact,
    planned: bool,
    group: str | None,
) -> dict[str, object]:
    return {
        "store": artifact.store,
        "path": str(artifact.path),
        "bytes": artifact.bytes,
        "arch": artifact.arch,
        "name": artifact.name,
        "tensors": artifact.tensors,
        "elements": artifact.elements,
        "file_type": artifact.file_type,
        "quant": artifact.quant,
        "label": artifact.label,
        "label_mismatch": artifact.label_mismatch,
        "tags": list(artifact.tags),
        "identity": artifact.identity,
        "planned": planned,
        "group": group,
    }


def _models_scan(args: argparse.Namespace) -> int:
    stores = default_stores()
    extra_indexes = [
        int(name.split(":", 1)[1])
        for name in stores
        if name.startswith("extra:") and name.split(":", 1)[1].isdigit()
    ]
    next_extra = max(extra_indexes, default=-1) + 1
    for root in getattr(args, "root", []) or []:
        root_path = Path(root).expanduser()
        if not root_path.is_dir():
            print(
                f"model root is not an existing directory: {root_path}",
                file=sys.stderr,
            )
            return 1
        stores[f"extra:{next_extra}"] = root_path
        next_extra += 1
    artifacts = scan_inventory(stores)
    plan = load_plan()
    planned = {
        str(Path(service.model_ref).resolve()).casefold()
        for service in (plan.services if plan is not None else [])
        if service.model_ref
    }
    duplicate_groups = inventory_duplicates(artifacts)
    variant_groups = inventory_variants(artifacts)
    resolved_paths = {
        artifact.path: str(artifact.path.resolve()).casefold()
        for artifact in artifacts
    }
    groups: dict[str, str] = {}
    for index, duplicate_group in enumerate(duplicate_groups, 1):
        for artifact in duplicate_group.artifacts:
            groups[resolved_paths[artifact.path]] = f"dup:{index}"
    for index, variant_group in enumerate(variant_groups, 1):
        for artifact in variant_group.artifacts:
            groups.setdefault(
                resolved_paths[artifact.path], f"var:{index}"
            )
    artifact_payloads = [
        _inventory_artifact_payload(
            artifact,
            resolved_paths[artifact.path] in planned,
            groups.get(resolved_paths[artifact.path]),
        )
        for artifact in artifacts
    ]
    store_totals: dict[str, dict[str, object]] = {}
    for store, root in stores.items():
        store_artifacts = [artifact for artifact in artifacts if artifact.store == store]
        total_bytes = sum(artifact.bytes for artifact in store_artifacts)
        store_totals[store] = {
            "path": str(root),
            "files": len(store_artifacts),
            "bytes": total_bytes,
            "gib": total_bytes / 1024**3,
        }
    reclaimable_bytes = sum(group.reclaimable_bytes for group in duplicate_groups)
    total_bytes = sum(artifact.bytes for artifact in artifacts)
    totals = {
        "files": len(artifacts),
        "bytes": total_bytes,
        "gib": total_bytes / 1024**3,
        "reclaimable_bytes": reclaimable_bytes,
        "reclaimable_gib": reclaimable_bytes / 1024**3,
        "variant_groups": len(variant_groups),
    }
    payload = {
        "stores": store_totals,
        "artifacts": artifact_payloads,
        "duplicates": [
            {
                "identity": group.identity,
                "artifacts": [str(item.path) for item in group.artifacts],
                "reclaimable_bytes": group.reclaimable_bytes,
            }
            for group in duplicate_groups
        ],
        "variants": [
            {
                "arch": group.arch,
                "quant": group.quant,
                "name": group.name,
                "artifacts": [str(item.path) for item in group.artifacts],
            }
            for group in variant_groups
        ],
        "totals": totals,
    }
    if args.json:
        _print_json(payload)
        return 0
    language = i18n.lang()
    table = Table(title=i18n.t("models.scan", language))
    for column in ("Store", "Model", "Quant", "Label", "GiB", "Tags", "Planned", "Group"):
        table.add_column(column)
    for artifact, item in zip(artifacts, artifact_payloads, strict=True):
        label = artifact.label or "-"
        if artifact.label_mismatch:
            label = f"{label} != {artifact.quant}"
        table.add_row(
            artifact.store,
            str(artifact.path),
            artifact.quant or "-",
            label,
            f"{artifact.bytes / 1024**3:.2f}",
            ",".join(artifact.tags) or "-",
            str(item["planned"]),
            str(item["group"] or "-"),
        )
    _console().print(table)
    for store, summary in store_totals.items():
        _console().print(i18n.t(
            "models.scan_store",
            language,
            store=store,
            files=summary["files"],
            gib=summary["gib"],
        ))
    _console().print(i18n.t(
        "models.scan_total",
        language,
        files=totals["files"],
        gib=totals["gib"],
    ))
    _console().print(i18n.t(
        "models.scan_reclaimable",
        language,
        gib=totals["reclaimable_gib"],
    ))
    _console().print(i18n.t(
        "models.scan_variants",
        language,
        count=totals["variant_groups"],
    ))
    return 0


def _models(args: argparse.Namespace) -> int:
    if getattr(args, "models_command", None) == "scan":
        return _models_scan(args)
    if getattr(args, "models_command", None) in {"local", "rm"}:
        model_root = nmesh_home() / "models"
        if args.models_command == "rm":
            candidate = Path(args.name)
            if not candidate.is_absolute():
                candidate = model_root / candidate
            candidate = candidate.resolve()
            if not candidate.is_file() or candidate.suffix.lower() != ".gguf":
                print(f"model not found: {args.name}", file=sys.stderr)
                return 1
            plan = load_plan()
            planned = {
                str(Path(service.model_ref).resolve()).casefold()
                for service in (plan.services if plan is not None else [])
                if service.model_ref
            }
            if str(candidate).casefold() in planned and not args.force:
                print(
                    i18n.t(
                        "models.planned_refusal",
                        i18n.lang(),
                        path=candidate,
                    ),
                    file=sys.stderr,
                )
                return 1
            try:
                candidate.unlink()
            except OSError as error:
                print(f"unable to remove model: {error}", file=sys.stderr)
                return 1
            if args.json:
                _print_json({"removed": str(candidate), "forced": bool(args.force)})
            else:
                print(i18n.t("models.removed", i18n.lang(), path=candidate))
            return 0
        plan = load_plan()
        planned = {
            str(Path(service.model_ref).resolve()).casefold()
            for service in (plan.services if plan is not None else [])
            if service.model_ref
        }
        items: list[dict[str, object]] = []
        for path in sorted(model_root.rglob("*.gguf")) if model_root.exists() else []:
            info = gguf_info(path)
            label = parse_label(path.name)
            quant = (
                FILE_TYPE_QUANT.get(info.file_type) if info.file_type is not None else None
            ) if info is not None else label
            items.append({
                "path": str(path),
                "bytes": path.stat().st_size,
                "quant": quant,
                "label": label,
                "label_mismatch": inventory_label_mismatch(quant, label),
                "quant_source": "header" if info is not None else "label",
                "planned": str(path.resolve()).casefold() in planned,
            })
        if args.json:
            _print_json(items)
        else:
            table = Table(title=i18n.t("models.local", i18n.lang()))
            for column in ("Path", "Bytes", "Quant", "Label", "Planned"):
                table.add_column(column)
            for item in items:
                table.add_row(
                    str(item["path"]), str(item["bytes"]), str(item["quant"] or "-"),
                    str(item["label"] or "-"), str(item["planned"]),
                )
            _console().print(table)
        return 0
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


def _engine(args: argparse.Namespace) -> int:
    command = getattr(args, "engine_command", None)
    try:
        if command == "list":
            entries = engine_runtime.installed()
            active = engine_runtime.active()
            available = engine_runtime.build_tags() if args.available else []
            payload: dict[str, object] = {
                "installed": [asdict(item) for item in entries],
                "active": asdict(active) if active is not None else None,
            }
            if args.available:
                payload["available"] = available
            if args.json:
                _print_json(payload)
                return 0
            table = Table(title=i18n.t("engine.installed", i18n.lang()))
            for column in (
                i18n.t("engine.tag", i18n.lang()),
                i18n.t("engine.variant", i18n.lang()),
                i18n.t("label.version", i18n.lang()),
                i18n.t("engine.active_column", i18n.lang()),
                i18n.t("engine.path", i18n.lang()),
            ):
                table.add_column(column)
            for item in entries:
                table.add_row(
                    item.tag, item.variant, item.version_line or "-",
                    "yes" if active is not None and active.tag == item.tag else "",
                    str(item.exe),
                )
            _console().print(table)
            if active is not None:
                _console().print(
                    f"{i18n.t('engine.active', i18n.lang())}: {active.tag}"
                )
            if args.available:
                _console().print(
                    f"{i18n.t('engine.available', i18n.lang())}: "
                    + ", ".join(available)
                )
            return 0
        if command == "install":
            item, warnings = engine_runtime.install(
                tag=args.version,
                variant=args.variant,
            )
            payload = {"installed": asdict(item), "warnings": warnings}
            if args.json:
                _print_json(payload)
            else:
                _console().print(i18n.t(
                    "engine.install", i18n.lang(), tag=item.tag, variant=item.variant,
                ))
                for warning in warnings:
                    _console().print(f"[yellow]- {warning}[/yellow]")
            return 0
        if command == "use":
            item = engine_runtime.use(args.tag)
            payload = {"active": asdict(item)}
            if args.json:
                _print_json(payload)
            else:
                _console().print(i18n.t(
                    "engine.use", i18n.lang(), tag=item.tag, variant=item.variant,
                ))
            return 0
        if command == "remove":
            was_active = engine_runtime.remove(args.tag)
            payload = {"removed": args.tag, "active_cleared": was_active}
            if args.json:
                _print_json(payload)
            else:
                _console().print(i18n.t("engine.remove", i18n.lang(), tag=args.tag))
                if was_active:
                    _console().print(i18n.t("engine.active_cleared", i18n.lang()))
            return 0
    except (OSError, RuntimeError, ValueError, tarfile.TarError, zipfile.BadZipFile) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 1


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


def _reference_context(
    service: PlannedService,
) -> tuple[Path, Path, str, int] | None:
    active = engine_runtime.active()
    server = (
        active.exe
        if active is not None and active.exe.is_file()
        else Path(service.launch.argv[0])
        if service.launch.argv
        else None
    )
    if server is None:
        return None
    binary = find_reference_binary(server)
    model_ref = Path(service.model_ref)
    model = choose_reference_model(model_ref.parent)
    if model is None and model_ref.is_file() and model_ref.suffix.casefold() == ".gguf":
        model = model_ref
    if binary is None or model is None:
        return None
    try:
        _, free_ram = free_budgets(detect_hardware())
        if free_ram < 1.5 * model.stat().st_size:
            return None
    except (OSError, RuntimeError):
        return None
    threads = max(1, min(os.cpu_count() or 4, 8))
    engine_build = active.tag if active is not None else server.name
    return binary, model, reference_id(engine_build, model, threads, 32), threads


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
    if service.roles == ["embed"]:
        try:
            with httpx.Client(timeout=300.0) as client:
                embed_measurement = measure_embedding(
                    client,
                    base_url,
                    service.model_ref,
                    requested_context=service.context,
                    runs=args.runs,
                )
        except httpx.HTTPError as error:
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", "unknown")
            print(
                i18n.t(
                    "err.bench_http",
                    i18n.lang(),
                    service=service.name,
                    url=f"{base_url}/v1/embeddings",
                    status=status,
                ),
                file=sys.stderr,
            )
            return 1
        except (OSError, RuntimeError) as error:
            print(
                i18n.t("err.bench_measure", i18n.lang(), error=error),
                file=sys.stderr,
            )
            return 1
        gpu_name = plan.profile.gpus[0].name if plan.profile.gpus else "cpu"
        embed_record = EmbedRecord(
            model_id=service.model_id,
            quant=service.quant,
            backend=service.backend,
            gpu_name=gpu_name,
            n_gpu_layers=service.n_gpu_layers or 0,
            requested_context=service.context,
            probe_tokens_small=embed_measurement.probe_tokens_small,
            served_small=embed_measurement.served_small,
            probe_tokens_large=embed_measurement.probe_tokens_large,
            served_large=embed_measurement.served_large,
            encode_tps=embed_measurement.encode_tps,
            encode_tps_min=embed_measurement.encode_tps_min,
            encode_tps_max=embed_measurement.encode_tps_max,
            encode_input_tokens=embed_measurement.encode_input_tokens,
            runs=embed_measurement.runs,
            harness=EMBED_HARNESS_VERSION,
            at=time.time(),
            refused_small=embed_measurement.refused_small,
            refused_large=embed_measurement.refused_large,
        )
        try:
            save_embed(embed_record)
        except OSError as error:
            print(i18n.t("err.bench_save", i18n.lang(), error=error), file=sys.stderr)
            return 1
        retrieval_record: RetrievalRecord | None = None
        if getattr(args, "retrieval", False):
            try:
                with httpx.Client(timeout=300.0) as client:
                    requests, seconds = measure_retrieval_estimate(
                        client, base_url, service.model_ref,
                        encode_tps=embed_record.encode_tps,
                    )
                    print(i18n.t(
                        "label.retrieval_estimate",
                        i18n.lang(),
                        requests=requests,
                        minutes=max(1, round(seconds / 60)),
                    ), file=(
                        sys.stderr
                        if getattr(args, "json", False)
                        else sys.stdout
                    ))
                    rungs = measure_retrieval(
                        client,
                        base_url,
                        service.model_ref,
                        cap=embed_record.cap,
                    )
            except httpx.HTTPError as error:
                response = getattr(error, "response", None)
                status = getattr(response, "status_code", "unknown")
                print(
                    i18n.t(
                        "err.bench_http",
                        i18n.lang(),
                        service=service.name,
                        url=f"{base_url}/v1/embeddings",
                        status=status,
                    ),
                    file=sys.stderr,
                )
                return 1
            except (OSError, RuntimeError) as error:
                print(
                    i18n.t("err.bench_measure", i18n.lang(), error=error),
                    file=sys.stderr,
                )
                return 1
            retrieval_record = RetrievalRecord(
                model_id=service.model_id,
                quant=service.quant,
                backend=service.backend,
                gpu_name=gpu_name,
                n_gpu_layers=service.n_gpu_layers or 0,
                rungs=rungs,
                digest=retrieval_digest(),
                harness=RETRIEVAL_HARNESS_VERSION,
                at=time.time(),
            )
            if (
                retrieval_record.usable_tokens is not None
                and retrieval_record.degraded_tokens is not None
            ):
                usable_rung = next(
                    (
                        rung for rung in retrieval_record.rungs
                        if rung.served_tokens == retrieval_record.usable_tokens
                    ),
                    None,
                )
                degraded_rung = next(
                    (
                        rung for rung in retrieval_record.rungs
                        if rung.served_tokens == retrieval_record.degraded_tokens
                    ),
                    None,
                )
                if usable_rung is not None and degraded_rung is not None:
                    try:
                        with httpx.Client(timeout=300.0) as client:
                            chunk_arm = measure_retrieval_chunk_arm(
                                client,
                                base_url,
                                service.model_ref,
                                doc_words=degraded_rung.words,
                                chunk_words=usable_rung.words,
                                chunk_tokens=usable_rung.served_tokens,
                            )
                    except httpx.HTTPError as error:
                        response = getattr(error, "response", None)
                        status = getattr(response, "status_code", "unknown")
                        print(
                            i18n.t(
                                "err.bench_http",
                                i18n.lang(),
                                service=service.name,
                                url=f"{base_url}/v1/embeddings",
                                status=status,
                            ),
                            file=sys.stderr,
                        )
                        return 1
                    except (OSError, RuntimeError) as error:
                        print(
                            i18n.t("err.bench_measure", i18n.lang(), error=error),
                            file=sys.stderr,
                        )
                        return 1
                    retrieval_record = replace(retrieval_record, chunk=chunk_arm)
            try:
                save_retrieval(retrieval_record)
            except OSError as error:
                print(
                    i18n.t("err.bench_save", i18n.lang(), error=error),
                    file=sys.stderr,
                )
                return 1
        if args.json:
            output: dict[str, object] = {**asdict(embed_record), "cap": embed_record.cap}
            if retrieval_record is not None:
                output["retrieval"] = {
                    **asdict(retrieval_record),
                    "control_passed": retrieval_record.control_passed,
                    "usable_tokens": retrieval_record.usable_tokens,
                    "degraded_tokens": retrieval_record.degraded_tokens,
                    "chunk_recovers": retrieval_record.chunk_recovers,
                    "pool_recovers": retrieval_record.pool_recovers,
                    "pool_hits": (
                        retrieval_record.chunk.pool_hits
                        if retrieval_record.chunk is not None else None
                    ),
                    "pool_trials": (
                        retrieval_record.chunk.pool_trials
                        if retrieval_record.chunk is not None else None
                    ),
                }
            _print_json(output)
        else:
            language = i18n.lang()
            cap = embed_record.cap if embed_record.cap is not None else "unproven"
            _console().print(
                i18n.t(
                    "label.embed_measurement",
                    language,
                    cap=cap,
                    tps=embed_record.encode_tps,
                )
            )
            if embed_record.cap is not None and embed_record.cap < service.context:
                _console().print(
                    i18n.t(
                        "warn.embed_truncated",
                        language,
                        cap=embed_record.cap,
                    )
                )
            refused_tokens = _embed_refused_tokens(embed_record)
            if refused_tokens is not None:
                _console().print(
                    i18n.t(
                        "label.embed_refused",
                        language,
                        tokens=refused_tokens,
                    )
                )
            if retrieval_record is not None:
                usable = (
                    retrieval_record.usable_tokens
                    if retrieval_record.usable_tokens is not None
                    else "unproven"
                )
                degraded = (
                    retrieval_record.degraded_tokens
                    if retrieval_record.degraded_tokens is not None
                    else "unproven"
                )
                rung_lines = "\n".join(
                    f"{rung.words} words: served={rung.served_tokens} "
                    f"rank1={rung.hits}/{rung.trials}"
                    for rung in retrieval_record.rungs
                )
                chunk = "unmeasured"
                if retrieval_record.chunk is not None:
                    outcome = (
                        "recovered"
                        if retrieval_record.chunk_recovers
                        else "not recovered"
                    )
                    chunk = (
                        f"{outcome} {retrieval_record.chunk.hits}/"
                        f"{retrieval_record.chunk.trials} at ~"
                        f"{retrieval_record.chunk.chunk_tokens} tokens"
                    )
                pool = "unmeasured"
                if retrieval_record.chunk is not None:
                    pool_outcome = (
                        "recovered"
                        if retrieval_record.pool_recovers
                        else "not recovered"
                    )
                    pool = (
                        f"{pool_outcome} {retrieval_record.chunk.pool_hits}/"
                        f"{retrieval_record.chunk.pool_trials}"
                    )
                _console().print(
                    i18n.t(
                        "label.retrieval_measurement",
                        language,
                        usable=usable,
                        degraded=degraded,
                        rungs=rung_lines,
                        chunk=chunk,
                        pool=pool,
                    )
                )
                if not retrieval_record.control_passed:
                    _console().print(
                        i18n.t("warn.retrieval_control", language)
                    )
                if (
                    retrieval_record.degraded_tokens is not None
                    and service.context > retrieval_record.degraded_tokens
                ):
                    _console().print(
                        i18n.t(
                            "warn.retrieval_degraded",
                            language,
                            degraded=retrieval_record.degraded_tokens,
                            context=service.context,
                        )
                    )
        return 0
    context = None if args.no_reference else _reference_context(service)
    history = load_history()
    reference_baseline = (
        baseline(history, context[2])
        if context is not None else None
    )
    before_reference = None
    if context is not None:
        try:
            before_reference = measure_reference(context[0], context[1], context[3])
        except (OSError, RuntimeError):
            before_reference = None
    try:
        controlled = measure_controlled(
            service,
            base_url,
            decode_tokens=args.tokens,
            runs=args.runs,
            passes=args.passes,
            cache_prompt=False if service.backend == "llamacpp" else None,
        )
    except httpx.HTTPError as error:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", "unknown")
        print(
            i18n.t(
                "err.bench_http",
                i18n.lang(),
                service=service.name,
                url=f"{base_url}/v1/chat/completions",
                status=status,
            ),
            file=sys.stderr,
        )
        return 1
    except (OSError, RuntimeError) as error:
        print(i18n.t("err.bench_measure", i18n.lang(), error=error), file=sys.stderr)
        return 1
    after_reference = None
    if context is not None:
        try:
            after_reference = measure_reference(context[0], context[1], context[3])
        except (OSError, RuntimeError):
            after_reference = None
    reference_tps = (
        statistics.mean((before_reference, after_reference))
        if before_reference is not None and after_reference is not None
        else None
    )
    reference_key = context[2] if context is not None else ""
    epoch = (
        classify(reference_tps, reference_baseline)
        if reference_tps is not None else "unknown"
    )
    pruned = 0
    if (
        reference_tps is not None
        and reference_key
        and epoch in {"healthy", "unknown"}
    ):
        samples = history.get(reference_key, ())
        retained = prune_degraded(samples, reference_tps)
        pruned = len(samples) - len(retained)
        history[reference_key] = (
            EpochSample(
                reference_id=reference_key,
                tps=reference_tps,
                measured_at=datetime.now(timezone.utc).isoformat(),
            ),
            *retained,
        )[:EPOCH_HISTORY]
        try:
            save_history(history)
        except OSError as error:
            print(i18n.t("err.bench_save", i18n.lang(), error=error), file=sys.stderr)
    measurement = controlled.result
    key = benchmark_key(service.model_id, service.quant, service.backend,
                        plan.profile.gpus[0].name if plan.profile.gpus else "cpu",
                        service.n_gpu_layers, service.kv_quant, service.spec)
    demoted: tuple[str, ...] = ()
    spec_demoted: tuple[str, ...] = ()
    delegation_demoted: tuple[str, ...] = ()
    stored = controlled.stable and epoch != "degraded"
    language = i18n.lang()
    warnings: list[str] = []
    records = load_records()
    if (
        reference_tps is not None
        and reference_key
        and epoch in {"healthy", "unknown"}
    ):
        demoted = demote_stale(records, reference_key, reference_tps)
        spec_records = load_spec_cache()
        spec_demoted = demote_spec_stale(
            spec_records, reference_key, reference_tps,
        )
        if spec_demoted:
            try:
                save_all_spec(spec_records)
            except OSError as error:
                print(
                    i18n.t("err.bench_save", language, error=error),
                    file=sys.stderr,
                )
        delegation_records = load_delegation_cache()
        delegation_demoted = demote_delegation_stale(
            delegation_records, reference_key, reference_tps,
        )
        if delegation_demoted:
            try:
                save_all_delegation(delegation_records)
            except OSError as error:
                print(
                    i18n.t("err.bench_save", language, error=error),
                    file=sys.stderr,
                )
    if measurement.decode_tokens_served < MIN_DECODE_TOKENS:
        warnings.append(i18n.t(
            "warn.bench_decode_unmeasurable",
            language,
            requested=args.tokens,
            served=measurement.decode_tokens_served,
            minimum=MIN_DECODE_TOKENS,
        ))
        stored = False
        record = None
        decode_spread = None
        if (
            reference_tps is not None
            and reference_key
            and epoch in {"healthy", "unknown"}
        ):
            try:
                save_records(records)
            except OSError as error:
                print(i18n.t("err.bench_save", language, error=error), file=sys.stderr)
                return 1
    else:
        if measurement.decode_tokens_served < args.tokens:
            warnings.append(i18n.t(
                "warn.bench_decode_short",
                language,
                requested=args.tokens,
                served=measurement.decode_tokens_served,
            ))
        record = merge_measurement(
            records,
            key,
            tps=measurement.decode_tps,
            decode_tps_min=measurement.decode_tps_min,
            decode_tps_max=measurement.decode_tps_max,
            runs=measurement.runs,
            passes=args.passes,
            control_ratio=controlled.control_ratio,
            reference_tps=reference_tps,
            reference_id=reference_key,
            epoch=epoch,
        )
        try:
            save_records(records)
        except OSError as error:
            print(i18n.t("err.bench_save", language, error=error), file=sys.stderr)
            return 1
        decode_spread = (
            (measurement.decode_tps_max - measurement.decode_tps_min)
            / measurement.decode_tps
            if measurement.decode_tps else 0.0
        )
    result = {
              "key": key, "prefill_tokens": 512,
              "decode_tokens_requested": args.tokens,
              "decode_tokens_served": measurement.decode_tokens_served,
              "median_tps": record.tps if record is not None else None,
              "session_tps": measurement.decode_tps if record is not None else None,
              "reference_tps": reference_tps,
              "reference_baseline": reference_baseline,
              "reference_id": reference_key,
              "epoch": epoch,
              "demoted": list(demoted),
              "spec_demoted": list(spec_demoted),
              "delegation_demoted": list(delegation_demoted),
              "pruned": pruned,
              "prefill_tps": measurement.prefill_tps,
              "ttft_s": measurement.ttft_s, "approximate": measurement.approximate,
              "prompt_tokens": measurement.prompt_tokens,
              "prefill_source": measurement.prefill_source,
              "cached_prompt_tokens": measurement.cached_prompt_tokens,
              "runs": measurement.runs,
              "decode_tps_min": (
                  measurement.decode_tps_min if record is not None else None
              ),
              "decode_tps_max": (
                  measurement.decode_tps_max if record is not None else None
              ),
              "decode_spread": decode_spread,
              "passes": args.passes,
              "pass_tps": list(controlled.pass_tps),
              "control_ratio": controlled.control_ratio,
              "stable": controlled.stable,
              "stored": stored,
              "confirmations": record.confirmations if record is not None else 0,
              "warnings": warnings}
    if args.json:
        _print_json(result)
    else:
        marker = "~" if measurement.approximate else ""
        prefill_marker = "~" if measurement.prefill_source != "timings" else ""
        lines: list[str] = []
        if record is not None:
            lines.extend((
                i18n.t("label.median_decode", language, marker=marker,
                       value=measurement.decode_tps),
                i18n.t("label.decode_range", language,
                       minimum=measurement.decode_tps_min,
                       maximum=measurement.decode_tps_max,
                       spread=decode_spread),
            ))
        lines.extend((
            i18n.t("label.prefill", language, marker=prefill_marker,
                   value=measurement.prefill_tps),
            i18n.t("label.ttft", language, marker=marker, value=measurement.ttft_s),
        ))
        if record is not None:
            lines.append(i18n.t(
                "label.bench_passes",
                language,
                passes=args.passes,
                values=", ".join(f"{value:.2f}" for value in controlled.pass_tps),
            ))
        lines.append(i18n.t(
            "label.bench_control",
            language,
            ratio=(
                f"{controlled.control_ratio:.1%}"
                if controlled.control_ratio is not None else "n/a"
            ),
        ))
        _console().print("\n".join(lines))
        for warning in warnings:
            _console().print(warning)
        if args.passes == 1:
            _console().print(i18n.t("warn.bench_no_control", language))
        elif not controlled.stable:
            _console().print(i18n.t(
                "warn.bench_control",
                language,
                ratio=controlled.control_ratio or 0.0,
                kept=i18n.t(
                    "label.bench_kept" if record is not None and record.stable
                    else "label.bench_nothing_stored",
                    language,
                ),
            ))
        if epoch == "degraded":
            _console().print(i18n.t(
                "warn.bench_epoch",
                language,
                ratio=(
                    reference_tps / reference_baseline
                    if reference_tps is not None and reference_baseline else 0.0
                ),
                kept=i18n.t(
                    "label.bench_kept" if record is not None and record.stable
                    else "label.bench_nothing_stored",
                    language,
                ),
            ))
        elif reference_tps is None:
            _console().print(i18n.t("warn.bench_no_reference", language))
        if demoted:
            _console().print(i18n.t(
                "warn.bench_demoted",
                language,
                count=len(demoted),
            ))
        if spec_demoted:
            _console().print(i18n.t(
                "warn.bench_spec_demoted",
                language,
                count=len(spec_demoted),
            ))
        if delegation_demoted:
            _console().print(i18n.t(
                "warn.bench_orchestrate_demoted",
                language,
                count=len(delegation_demoted),
            ))
        if decode_spread is not None and decode_spread > 0.25:
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
    requested = (
        None if args.categories is None
        else {item.strip() for item in args.categories.split(",") if item.strip()}
    )
    base_tasks = SUITES[args.suite]
    tasks = (
        base_tasks if requested is None
        else tuple(task for task in base_tasks if task.category in requested)
    )
    if not tasks:
        print(i18n.t("err.eval_categories", i18n.lang()), file=sys.stderr)
        return 1
    base_url = "http://127.0.0.1:11434" if service.backend == "ollama" else (
        f"http://127.0.0.1:{service.port}"
    )
    allowance = max(0, getattr(args, "reasoning_allowance", 0) or 0)
    timeout = getattr(args, "timeout", None)
    depth = max(0, getattr(args, "depth", 0) or 0)
    try:
        result = eval_run(
            tasks,
            base_url,
            service.model_ref,
            timeout=timeout,
            reasoning_allowance=allowance,
            cache_prompt=False if service.backend == "llamacpp" else None,
            depth=depth,
        )
    except RuntimeError as error:
        print(i18n.t("err.eval_run", i18n.lang(), error=error), file=sys.stderr)
        return 1
    result = replace(
        result,
        model_id=service.model_id,
        quant=service.quant,
        backend=service.backend,
        artifact=service_fingerprint(service.backend, service.model_ref) or "",
        suite=args.suite,
        digest=suite_digest(tasks),
        reasoning_allowance=allowance,
        transport_errors=result.transport_errors,
        depth=depth,
    )
    context_probe = None
    context_probe_families: dict[str, dict[str, int | bool]] = {}
    context_depth_lost: list[tuple[str, int, int, int]] = []
    context_uncontrolled: list[str] = []
    context_record: ContextRecord | None = None
    context_path: Path | None = None
    if depth > 0:
        probes = needle_tasks(depth, seed=args.suite)
        try:
            control_before = eval_run(
                needle_tasks(0, seed=args.suite),
                base_url,
                service.model_ref,
                timeout=timeout,
                reasoning_allowance=allowance,
                cache_prompt=False if service.backend == "llamacpp" else None,
                depth=0,
            )
        except RuntimeError as error:
            print(i18n.t("err.eval_run", i18n.lang(), error=error), file=sys.stderr)
            return 1
        try:
            probe_result = eval_run(
                probes,
                base_url,
                service.model_ref,
                timeout=timeout,
                reasoning_allowance=allowance,
                cache_prompt=False if service.backend == "llamacpp" else None,
                depth=depth,
            )
        except RuntimeError as error:
            print(i18n.t("err.eval_run", i18n.lang(), error=error), file=sys.stderr)
            return 1
        try:
            control_after = eval_run(
                needle_tasks(0, seed=args.suite),
                base_url,
                service.model_ref,
                timeout=timeout,
                reasoning_allowance=allowance,
                cache_prompt=False if service.backend == "llamacpp" else None,
                depth=0,
            )
        except RuntimeError as error:
            print(i18n.t("err.eval_run", i18n.lang(), error=error), file=sys.stderr)
            return 1
        controls = (control_before, control_after)
        control_pass: dict[str, bool] = {}
        for control in controls:
            for item in control.outcomes:
                control_pass[item.id] = control_pass.get(item.id, True) and item.passed
        families: dict[str, dict[str, int | bool]] = {}
        for category in _CONTEXT_CATEGORIES:
            category_outcomes = [
                item for item in probe_result.outcomes if item.category == category
            ]
            paired = [
                item for item in category_outcomes
                if control_pass.get(item.id) is True
            ]
            control_outcomes = [
                item
                for control in controls
                for item in control.outcomes
                if item.category == category
            ]
            passed = sum(item.passed for item in paired)
            total = len(paired)
            control_passed = sum(item.passed for item in control_outcomes)
            control_total = len(control_outcomes)
            attributable = total > 0
            family = category.rsplit(".", 1)[-1]
            families[family] = {
                "passed": passed,
                "of": total,
                "control_passed": control_passed,
                "control_of": control_total,
                "attributable": attributable,
            }
            if not attributable:
                context_uncontrolled.append(family)
            elif passed < total:
                context_depth_lost.append((
                    family,
                    passed,
                    total,
                    probe_result.prompt_tokens_max or depth,
                ))
        control_passed = sum(
            int(values["control_passed"]) for values in families.values()
        )
        control_of = sum(int(values["control_of"]) for values in families.values())
        context_probe = {
            "passed": probe_result.passed,
            "of": probe_result.n_tasks,
            "families": families,
            "control_passed": control_passed,
            "control_of": control_of,
            "attributable": all(
                bool(values["attributable"]) for values in families.values()
            ),
        }
        context_probe_families = families
        context_record = ContextRecord(
            result.model_id,
            result.quant,
            result.backend,
            args.suite,
            depth,
            probe_result.prompt_tokens_max,
            suite_digest(probes),
            tuple(
                FamilyResult(
                    category,
                    int(families[category.rsplit(".", 1)[-1]]["passed"]),
                    int(families[category.rsplit(".", 1)[-1]]["of"]),
                    int(families[category.rsplit(".", 1)[-1]]["control_passed"]),
                    int(families[category.rsplit(".", 1)[-1]]["control_of"]),
                )
                for category in _CONTEXT_CATEGORIES
            ),
            probe_result.at,
            result.artifact,
            result.cache_prompt,
        )
    cached = load_eval_cache()
    key = eval_key(
        result.model_id, result.quant, result.backend, result.suite, result.digest,
        result.reasoning_allowance,
        result.cache_prompt,
        result.depth,
    )
    previous = cached.get(key)
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
        if context_record is not None:
            context_path = save_context(context_record)
    except OSError as error:
        print(i18n.t("err.eval_save", i18n.lang(), error=error), file=sys.stderr)
        return 1
    divergence = _eval_divergence(result, cached)
    stale_grader_notes = _stale_grader_notes(cached)
    failed_outcomes = [outcome for outcome in result.outcomes if not outcome.passed]
    failed = [
        {
            "id": outcome.id,
            "output": outcome.output,
            "unscorable": outcome.unscorable,
            "value_passed": outcome.value_passed,
            "failure_kind": outcome.failure_kind,
        }
        for outcome in failed_outcomes
    ]
    value_only_failures = sum(
        outcome.value_passed is True for outcome in failed_outcomes
    )
    value_checked_failures = sum(
        outcome.value_passed is not None for outcome in failed_outcomes
    )
    failures_by_kind = {
        "value": sum(outcome.failure_kind == "value" for outcome in failed_outcomes),
        "form": sum(outcome.failure_kind == "form" for outcome in failed_outcomes),
    }
    language = i18n.lang()
    context_depth_warnings = [
        i18n.t(
            "warn.context_depth_lost",
            language,
            model=result.model_id,
            quant=result.quant,
            backend=result.backend,
            family=family,
            passed=passed,
            of=total,
            depth=served_depth,
        )
        for family, passed, total, served_depth in context_depth_lost
    ]
    context_probe_note = (
        i18n.t(
            "note.context_probe_uncontrolled",
            language,
            families=", ".join(context_uncontrolled),
        )
        if context_uncontrolled
        else None
    )
    if context_probe is not None:
        context_probe["depth_warnings"] = context_depth_warnings
        context_probe["uncontrolled_families"] = context_uncontrolled
        context_probe["uncontrolled_note"] = context_probe_note
        if context_path is not None:
            context_probe["saved"] = str(context_path)
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
    unscorable_note = None
    if result.unscorable:
        unscorable_note = i18n.t(
            "warn.eval_unscorable",
            language,
            count=result.unscorable,
            tasks=result.n_tasks,
            allowance=allowance,
        )
    transport_note = None
    if result.transport_errors:
        transport_note = i18n.t(
            "warn.eval_transport",
            language,
            count=result.transport_errors,
        )
    value_note = None
    if value_checked_failures:
        value_note = i18n.t(
            "note.eval_value_vs_discipline",
            language,
            checked=value_checked_failures,
            value_only=value_only_failures,
        )
    failure_kinds_note = None
    if failed_outcomes:
        failure_kinds_note = i18n.t(
            "note.eval_failure_kinds",
            language,
            failures=len(failed_outcomes),
            value=failures_by_kind["value"],
            form=failures_by_kind["form"],
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
        "digest": result.digest,
        "artifact": result.artifact or None,
        "n_tasks": result.n_tasks,
        "passed": result.passed,
        "unscorable": result.unscorable,
        "transport_errors": result.transport_errors,
        "reasoning_allowance": result.reasoning_allowance,
        "requested_depth": result.depth,
        "served_depth": result.prompt_tokens_max or None,
        "served_depth_known": bool(result.prompt_tokens_max),
        "unscorable_note": unscorable_note,
        "transport_note": transport_note,
        "pass_rate": result.pass_rate,
        "pass_rate_ci": list(pass_rate_ci),
        "min_resolvable_difference": minimum_difference,
        "by_category": result.by_category,
        "failed": failed,
        "failures_by_kind": failures_by_kind,
        "value_only_failures": value_only_failures,
        "value_checked_failures": value_checked_failures,
        "value_note": value_note,
        "failure_kinds_note": failure_kinds_note,
        "note": note,
        "uncertainty_note": uncertainty_note,
        "suite_upgrade_note": suite_upgrade_note,
        "stale_grader_notes": stale_grader_notes,
        "config_note": config_note,
        "divergence": [asdict(other) for other in divergence],
        "artifact_warning": artifact_warning,
        "context_probe": context_probe,
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
    _console().print(i18n.t(
        "label.eval_depth",
        language,
        requested=result.depth,
        served=(
            str(result.prompt_tokens_max)
            if result.prompt_tokens_max else
            i18n.t("label.unknown", language)
        ),
    ))
    if context_probe is not None:
        _console().print(i18n.t(
            "label.eval_context_probe",
            language,
            passed=context_probe["passed"],
            total=context_probe["of"],
            literal_passed=context_probe_families["literal"]["passed"],
            literal_total=context_probe_families["literal"]["of"],
            latent_passed=context_probe_families["latent"]["passed"],
            latent_total=context_probe_families["latent"]["of"],
            multi_passed=context_probe_families["multi"]["passed"],
            multi_total=context_probe_families["multi"]["of"],
        ))
        _console().print(i18n.t(
            "label.eval_context_control",
            language,
            passed=context_probe["control_passed"],
            total=context_probe["control_of"],
        ))
        for warning in context_depth_warnings:
            _console().print(warning)
        if context_probe_note is not None:
            _console().print(context_probe_note)
    for stale_note in stale_grader_notes:
        _console().print(stale_note)
    _console().print(uncertainty_note)
    if suite_upgrade_note is not None:
        _console().print(suite_upgrade_note)
    _console().print(config_note)
    if unscorable_note is not None:
        _console().print(unscorable_note)
    if transport_note is not None:
        _console().print(transport_note)
    if value_note is not None:
        _console().print(value_note)
    if failure_kinds_note is not None:
        _console().print(failure_kinds_note)
    if artifact_warning is not None:
        _console().print(artifact_warning)
    for other in divergence:
        ids = ", ".join(other.disagreeing) or "-"
        _console().print(i18n.t(
            "note.eval_divergence",
            language,
            config=other.config,
            other_rate=f"{other.pass_rate:.1%}",
            rate=f"{result.pass_rate:.1%}",
            count=len(other.disagreeing),
            compared=other.compared,
            ids=ids,
        ))
        _console().print(i18n.t(
            "note.eval_paired_power",
            language,
            compared=other.compared,
            discordant=other.discordant_here + other.discordant_there,
            here=other.discordant_here,
            there=other.discordant_there,
            required=min_discordant_for_significance(),
            families=", ".join(other.zero_power_families) or "-",
        ))
    _console().print(i18n.t(
        "label.eval_overall", language, passed=result.passed, total=result.n_tasks,
        rate=result.pass_rate,
    ))
    failed_ids = ", ".join(outcome.id for outcome in failed_outcomes) or "-"
    _console().print(i18n.t("label.eval_failed", language, ids=failed_ids))
    return 0


def _orchestration_service(
    plan: Plan, selector: str,
) -> PlannedService | None:
    selector = plan.routing.role_to_service.get(selector, selector)
    service = next(
        (item for item in plan.services if item.name == selector),
        None,
    )
    if service is not None:
        return service
    return next((item for item in plan.services if item.model_id == selector), None)


def _orchestration_identity(service: PlannedService) -> RoleIdentity:
    return RoleIdentity(
        model_id=service.model_id,
        quant=service.quant,
        backend=service.backend,
        artifact=service_fingerprint(service.backend, service.model_ref) or "",
    )


def _orchestration_generative(service: PlannedService) -> bool:
    return bool(set(service.roles) & {"chat", "code", "worker"})


def _orchestration_url(service: PlannedService) -> str:
    return "http://127.0.0.1:11434" if service.backend == "ollama" else (
        f"http://127.0.0.1:{service.port}"
    )


def _orchestrate_measure_command(args: argparse.Namespace) -> int:
    language = i18n.lang()
    plan = load_plan()
    if plan is None or not plan.services:
        print(i18n.t("err.orchestrate_plan", i18n.lang()), file=sys.stderr)
        return 1
    lead = _orchestration_service(plan, args.lead)
    worker = _orchestration_service(plan, args.worker)
    if worker is None and args.worker == "worker":
        worker = next(
            (
                item for item in plan.services
                if item != lead and _orchestration_generative(item)
            ),
            None,
        )
    if lead is None:
        print(
            i18n.t(
                "err.orchestrate_service",
                i18n.lang(),
                lead=args.lead,
                worker=args.worker,
            ),
            file=sys.stderr,
        )
        return 1
    if worker is None:
        print(
            i18n.t(
                "err.orchestrate_no_worker",
                i18n.lang(),
                lead=lead.name,
            ),
            file=sys.stderr,
        )
        return 1
    if not args.lead_url and not _orchestration_generative(lead):
        print(
            i18n.t(
                "err.orchestrate_nongenerative",
                i18n.lang(),
                role="lead",
                service=lead.name,
            ),
            file=sys.stderr,
        )
        return 1
    if not args.worker_url and not _orchestration_generative(worker):
        print(
            i18n.t(
                "err.orchestrate_nongenerative",
                i18n.lang(),
                role="worker",
                service=worker.name,
            ),
            file=sys.stderr,
        )
        return 1
    tasks = SUITES[args.suite]
    if args.limit is not None:
        tasks = tasks[:args.limit]
    lead_url = args.lead_url or _orchestration_url(lead)
    worker_url = args.worker_url or _orchestration_url(worker)
    if not args.lead_url and not _service_running(lead, runtime_status()):
        print(i18n.t("err.orchestrate_up", i18n.lang()), file=sys.stderr)
        return 1
    if not args.worker_url and not _service_running(worker, runtime_status()):
        print(i18n.t("err.orchestrate_up", language), file=sys.stderr)
        return 1
    lead_identity = _orchestration_identity(lead)
    worker_identity = _orchestration_identity(worker)
    reference_context = (
        None if args.no_reference else _reference_context(lead)
    )
    history = load_history()
    reference_key = (
        reference_context[2] if reference_context is not None else ""
    )
    reference_baseline = (
        baseline(history, reference_key)
        if reference_context is not None else None
    )
    before_reference = None
    if reference_context is not None:
        try:
            before_reference = measure_reference(
                reference_context[0],
                reference_context[1],
                reference_context[3],
            )
        except (OSError, RuntimeError):
            before_reference = None
    try:
        runs = [
            orchestrate_measure(
                tasks,
                lead=Endpoint(
                    lead_url,
                    lead.model_ref,
                    cache_prompt=False if lead.backend == "llamacpp" else None,
                ),
                worker=Endpoint(
                    worker_url,
                    worker.model_ref,
                    cache_prompt=(
                        False if worker.backend == "llamacpp" else None
                    ),
                ),
                lead_identity=lead_identity,
                worker_identity=worker_identity,
                suite=args.suite,
                reasoning_allowance=max(0, args.reasoning_allowance),
            )
            for _ in range(args.repeats)
        ]
        run = combine(runs)
        after_reference = None
        if reference_context is not None:
            try:
                after_reference = measure_reference(
                    reference_context[0],
                    reference_context[1],
                    reference_context[3],
                )
            except (OSError, RuntimeError):
                after_reference = None
        reference_tps = (
            statistics.mean((before_reference, after_reference))
            if before_reference is not None and after_reference is not None
            else None
        )
        epoch = (
            classify(reference_tps, reference_baseline)
            if reference_tps is not None else "unknown"
        )
        pruned = 0
        if (
            reference_tps is not None
            and reference_key
            and epoch in {"healthy", "unknown"}
        ):
            samples = history.get(reference_key, ())
            retained = prune_degraded(samples, reference_tps)
            pruned = len(samples) - len(retained)
            history[reference_key] = (
                EpochSample(
                    reference_id=reference_key,
                    tps=reference_tps,
                    measured_at=datetime.now(timezone.utc).isoformat(),
                ),
                *retained,
            )[:EPOCH_HISTORY]
            try:
                save_history(history)
            except OSError as error:
                print(
                    i18n.t("err.bench_save", language, error=error),
                    file=sys.stderr,
                )
        save_delegation(
            run,
            reference_id=reference_key,
            reference_tps=reference_tps or 0.0,
            epoch=epoch,
        )
        delegation_records = load_delegation_cache()
        demoted: tuple[str, ...] = ()
        if (
            reference_tps is not None
            and reference_key
            and epoch in {"healthy", "unknown"}
        ):
            demoted = demote_delegation_stale(
                delegation_records, reference_key, reference_tps,
            )
            if demoted:
                try:
                    save_all_delegation(delegation_records)
                except OSError as error:
                    print(
                        i18n.t("err.bench_save", language, error=error),
                        file=sys.stderr,
                    )
    except (OSError, RuntimeError, ValueError, httpx.HTTPError) as error:
        print(
            i18n.t("err.orchestrate_measure", language, error=error),
            file=sys.stderr,
        )
        return 1
    record = from_run(
        run,
        reference_id=reference_key,
        reference_tps=reference_tps or 0.0,
        epoch=epoch,
    )
    decision, reason = decide(record)
    cost, cost_reason = decide_cost(record)
    output = {
        "n": run.n_tasks,
        "worker_passed": run.worker_passed,
        "lead_passed": run.lead_passed,
        "delegated_passed": run.delegated_passed,
        "ceiling_passed": run.ceiling_passed,
        "delegated_gained": run.delegated_vs_lead.gained,
        "delegated_lost": run.delegated_vs_lead.lost,
        "delegated_p": run.delegated_vs_lead.p,
        "ceiling_gained": run.ceiling_vs_lead.gained,
        "ceiling_lost": run.ceiling_vs_lead.lost,
        "ceiling_p": run.ceiling_vs_lead.p,
        "verifier_accuracy": run.verifier.accuracy,
        "verifier_accepted": run.verifier.accepted,
        "accepted_but_wrong": run.verifier.accepted_but_wrong,
        "rejected_but_right": run.verifier.rejected_but_right,
        "verifier_unparsed": run.verifier.unparsed,
        "lead_tokens_solo": run.lead_tokens_solo,
        "lead_tokens_delegated": run.lead_tokens_delegated,
        "verify_overhead": run.verify_overhead,
        "seconds_solo": run.seconds_solo,
        "seconds_delegated": run.seconds_delegated,
        "reference_tps": reference_tps,
        "reference_id": reference_key,
        "epoch": epoch,
        "demoted": len(demoted),
        "pruned": pruned,
        "cost": cost,
        "cost_reason": cost_reason,
        "seconds_ratio": record.seconds_ratio,
        "token_ratio": record.token_ratio,
        "repeats": run.repeats,
        "unstable_tasks": run.unstable_tasks,
        "gate": decision,
        "reason": reason,
        "digest": run.digest,
        "protocol": run.protocol,
    }
    if args.json:
        _print_json(output)
        return 0
    _console().print("\n".join((
        i18n.t("label.orchestrate_summary", language, n=run.n_tasks),
        i18n.t(
            "label.orchestrate_passed",
            language,
            worker=run.worker_passed,
            lead=run.lead_passed,
            delegated=run.delegated_passed,
            ceiling=run.ceiling_passed,
        ),
        i18n.t(
            "label.orchestrate_comparison",
            language,
            name="delegated",
            gained=run.delegated_vs_lead.gained,
            lost=run.delegated_vs_lead.lost,
            p=run.delegated_vs_lead.p,
        ),
        i18n.t(
            "label.orchestrate_comparison",
            language,
            name="ceiling",
            gained=run.ceiling_vs_lead.gained,
            lost=run.ceiling_vs_lead.lost,
            p=run.ceiling_vs_lead.p,
        ),
        i18n.t(
            "label.orchestrate_verifier",
            language,
            accuracy=run.verifier.accuracy,
            accepted=run.verifier.accepted,
            wrong=run.verifier.accepted_but_wrong,
            right=run.verifier.rejected_but_right,
            unparsed=run.verifier.unparsed,
        ),
        i18n.t(
            "label.orchestrate_cost",
            language,
            solo=run.lead_tokens_solo,
            delegated=run.lead_tokens_delegated,
            overhead=run.verify_overhead,
            solo_seconds=run.seconds_solo,
            delegated_seconds=run.seconds_delegated,
        ),
        i18n.t("label.orchestrate_gate", language, decision=decision, reason=reason),
        i18n.t(
            "label.orchestrate_repeats",
            language,
            repeats=run.repeats,
            unstable=run.unstable_tasks,
        ),
    )))
    if epoch == "degraded":
        _console().print(i18n.t("warn.orchestrate_degraded", language))
    elif reference_tps is None and not args.no_reference:
        _console().print(i18n.t("warn.bench_no_reference", language))
    if demoted:
        _console().print(i18n.t(
            "warn.orchestrate_demoted",
            language,
            count=len(demoted),
        ))
    return 0


def _orchestrate_show(args: argparse.Namespace) -> int:
    records = load_delegation_cache()
    language = i18n.lang()
    if args.json:
        _print_json([
            {
                **asdict(record),
                "gate": decide(record)[0],
                "reason": decide(record)[1],
                "cost": decide_cost(record)[0],
                "cost_reason": decide_cost(record)[1],
            }
            for record in records.values()
        ])
        return 0
    if not records:
        _console().print(i18n.t("label.orchestrate_empty", language))
        return 0
    table = Table(title=i18n.t("label.orchestrate_title", language))
    for column in (
        "lead",
        "worker",
        "suite",
        "n",
        "delegated_passed",
        "lead_passed",
        "gate",
        "epoch",
        "cost",
        "repeats",
        "unstable",
    ):
        table.add_column(column)
    for record in sorted(records.values(), key=lambda item: item.at, reverse=True):
        table.add_row(
            record.lead.model_id,
            record.worker.model_id,
            record.suite,
            str(record.n_tasks),
            str(record.delegated_passed),
            str(record.lead_passed),
            decide(record)[0],
            record.epoch,
            decide_cost(record)[0],
            str(record.repeats),
            str(record.unstable_tasks),
        )
    _console().print(table)
    return 0


def _spec_measure_argv(service: PlannedService, kind: str, draft: str,
                       n_max: int, port: int, enabled: bool) -> list[str]:
    argv = list(service.launch.argv)
    cleaned: list[str] = []
    skip = 0
    for item in argv:
        if skip:
            skip -= 1
            continue
        if item in {"--spec-type", "--spec-draft-model", "--spec-draft-n-max"}:
            skip = 1
            continue
        if item == "--port":
            cleaned.extend([item, str(port)])
            skip = 1
            continue
        cleaned.append(item)
    if enabled:
        if kind == KIND_NGRAM:
            cleaned.extend(["--spec-type", "ngram-simple"])
        else:
            cleaned.extend([
                "--spec-type", "draft-simple",
                "--spec-draft-model", draft,
                "--spec-draft-n-max", str(n_max),
            ])
    return cleaned


def _spec_measure_command(args: argparse.Namespace) -> int:
    language = i18n.lang()
    if args.kind == KIND_DRAFT and not args.draft:
        print(i18n.t("err.spec_draft_required", language), file=sys.stderr)
        return 2
    if args.repeats < 3:
        print(
            i18n.t("err.spec_repeats", language),
            file=sys.stderr,
        )
        return 2
    plan = load_plan()
    if plan is None or not plan.services:
        print(i18n.t("err.no_active_plan", language), file=sys.stderr)
        return 1
    service = None
    if args.service:
        service = next((item for item in plan.services if item.name == args.service), None)
        if service is None:
            print(i18n.t("err.unknown_service", language, service=args.service), file=sys.stderr)
            return 1
    else:
        service = next(
            (
                item for item in plan.services
                if set(item.roles) & {"chat", "code", "worker"}
            ),
            None,
        )
        if service is None:
            print(
                i18n.t("err.spec_no_generative_service", language),
                file=sys.stderr,
            )
            return 1
    target = RoleIdentity(
        model_id=service.model_id,
        quant=service.quant,
        backend=service.backend,
        artifact=service_fingerprint(service.backend, service.model_ref) or "",
    )
    draft = str(Path(args.draft).expanduser().resolve()) if args.draft else ""
    draft_identity = (
        RoleIdentity(Path(draft).stem, "", "llamacpp",
                     service_fingerprint("llamacpp", draft) or "")
        if draft else None
    )
    reference_context = (
        None if args.no_reference else _reference_context(service)
    )
    history = load_history()
    reference_key = (
        reference_context[2] if reference_context is not None else ""
    )
    reference_baseline = (
        baseline(history, reference_key)
        if reference_context is not None else None
    )
    before_reference = None
    if reference_context is not None:
        try:
            before_reference = measure_reference(
                reference_context[0],
                reference_context[1],
                reference_context[3],
            )
        except (OSError, RuntimeError):
            before_reference = None
    engine_id = engine_identity(plan.profile)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    with tempfile.TemporaryDirectory(prefix="nmesh-spec-") as temp:
        state_path = Path(temp) / "state.json"
        supervisor = Supervisor(
            state_path=state_path, plan_path=Path(temp) / "plan.json"
        )
        try:
            def arm(enabled: bool):
                launch = replace(
                    service.launch,
                    argv=_spec_measure_argv(
                        service, args.kind, draft, args.n_max, port, enabled
                    ),
                    health_url=f"http://127.0.0.1:{port}/health",
                )
                active_engine = engine_runtime.active()
                binary = (
                    str(active_engine.exe)
                    if active_engine is not None and active_engine.exe.is_file()
                    else plan.profile.backend_paths.get(service.backend)
                )
                if binary:
                    launch = replace(
                        launch,
                        argv=[binary, *launch.argv[1:]],
                    )
                item = replace(
                    service,
                    port=port,
                    launch=launch,
                    spec=args.kind if enabled else "none",
                    spec_draft=draft if enabled else "",
                )
                arm_plan = replace(
                    plan,
                    services=[item],
                    swap_group=[],
                )
                supervisor.up(arm_plan, no_download=True, admit=False)
                base_url = f"http://127.0.0.1:{port}"
                request_timeout = 30.0 + max(
                    workload.max_tokens for workload in WORKLOADS
                ) / 2.0
                with httpx.Client(timeout=request_timeout) as client:
                    return run_arm(
                        client, base_url, item.model_ref,
                        target=target,
                        spec=(
                            SpecConfig(
                                kind=args.kind,
                                draft=draft_identity,
                                n_max=args.n_max,
                            )
                            if enabled else SpecConfig()
                        ),
                        repeats=args.repeats,
                    )

            reference = arm(False)
            supervisor.down()
            candidate = arm(True)
            supervisor.down()
            control_arm = arm(False)
            after_reference = None
            if reference_context is not None:
                try:
                    after_reference = measure_reference(
                        reference_context[0],
                        reference_context[1],
                        reference_context[3],
                    )
                except (OSError, RuntimeError):
                    after_reference = None
            reference_tps = (
                statistics.mean((before_reference, after_reference))
                if before_reference is not None and after_reference is not None
                else None
            )
            epoch = (
                classify(reference_tps, reference_baseline)
                if reference_tps is not None else "unknown"
            )
            pruned = 0
            if (
                reference_tps is not None
                and reference_key
                and epoch in {"healthy", "unknown"}
            ):
                samples = history.get(reference_key, ())
                retained = prune_degraded(samples, reference_tps)
                pruned = len(samples) - len(retained)
                history[reference_key] = (
                    EpochSample(
                        reference_id=reference_key,
                        tps=reference_tps,
                        measured_at=datetime.now(timezone.utc).isoformat(),
                    ),
                    *retained,
                )[:EPOCH_HISTORY]
                try:
                    save_history(history)
                except OSError as error:
                    print(
                        i18n.t("err.bench_save", language, error=error),
                        file=sys.stderr,
                    )
            spec_records = load_spec_cache()
            demoted: tuple[str, ...] = ()
            if (
                reference_tps is not None
                and reference_key
                and epoch in {"healthy", "unknown"}
            ):
                demoted = demote_spec_stale(
                    spec_records, reference_key, reference_tps,
                )
                if demoted:
                    save_all_spec(spec_records)
            record = from_arms(
                reference,
                candidate,
                engine=engine_id,
                control_arm=control_arm,
                reference_id=reference_key,
                reference_tps=reference_tps or 0.0,
                epoch=epoch,
            )
            save_spec(record)
        except httpx.HTTPError:
            print(i18n.t("err.spec_transport", language), file=sys.stderr)
            return 1
        except (OSError, RuntimeError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 1
        finally:
            supervisor.down()
    decision, reason = decide_spec(record)
    if args.json:
        _print_json({
            "target": asdict(record.target),
            "spec": asdict(record.spec),
            "engine": record.engine,
            "classes": [asdict(item) for item in record.classes],
            "control": [asdict(item) for item in record.control],
            "reference_id": record.reference_id,
            "reference_tps": record.reference_tps or None,
            "epoch": record.epoch,
            "demoted": len(demoted),
            "pruned": pruned,
            "decision": decision,
            "reason": reason,
        })
    else:
        table = Table(title="nmesh spec measure")
        for column in (
            "class", "reference", "candidate", "speedup", "identical",
            "acceptance", i18n.t("label.spec_ref_spread", language),
            i18n.t("label.spec_cand_spread", language),
        ):
            table.add_column(column)
        for item in record.classes:
            table.add_row(
                item.name, f"{item.reference_tps:.2f}", f"{item.candidate_tps:.2f}",
                f"{item.speedup:.2f}", str(item.identical), f"{item.acceptance:.2f}",
                f"{item.reference_spread:.1%}", f"{item.candidate_spread:.1%}",
            )
        _console().print(table)
        worst = min((item.ratio for item in record.control), default=0.0)
        identical = bool(record.control) and all(
            item.identical for item in record.control
        )
        print(i18n.t(
            "label.spec_control",
            i18n.lang(),
            ratio=f"{worst:.2f}",
            identical=identical,
        ))
        if record.epoch == "degraded":
            _console().print(i18n.t("warn.spec_degraded", language))
        elif reference_tps is None and not args.no_reference:
            _console().print(i18n.t("warn.bench_no_reference", language))
        if demoted:
            _console().print(i18n.t(
                "warn.spec_demoted",
                language,
                count=len(demoted),
            ))
        print(f"decision: {decision} ({reason})")
    return 0


def _spec_show(args: argparse.Namespace) -> int:
    records = load_spec_cache()
    if args.json:
        _print_json({
            key: {
                **asdict(record),
                "decision": decide_spec(record)[0],
                "reason": decide_spec(record)[1],
            }
            for key, record in records.items()
        })
        return 0
    table = Table(title="nmesh spec")
    for column in (
        "target", "kind", "engine", "control", "epoch", "decision",
    ):
        table.add_column(column)
    for record in records.values():
        control_text = ", ".join(
            f"{item.name}:{item.ratio:.2f}/{item.identical}"
            for item in record.control
        ) or "-"
        table.add_row(
            record.target.model_id, record.spec.kind, record.engine,
            control_text,
            record.epoch,
            decide_spec(record)[0],
        )
    _console().print(table)
    return 0


def _offline_items(path: str, sources: Sequence[str]) -> tuple[
    tuple[SourceStatus, ...], tuple[SourceItem, ...]
]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(raw_items, list):
        raise TypeError("offline items must be a JSON list or an object with items")
    items: list[SourceItem] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        source = raw.get("source")
        url = raw.get("url")
        if not isinstance(source, str) or not isinstance(url, str):
            continue
        items.append(SourceItem(
            source,
            url,
            raw.get("title", "") if isinstance(raw.get("title", ""), str) else "",
            raw.get("body", "") if isinstance(raw.get("body", ""), str) else "",
            raw.get("published", "") if isinstance(raw.get("published", ""), str) else "",
        ))
    statuses = tuple(
        SourceStatus(
            source,
            True,
            sum(item.source == source for item in items),
            all(item.body for item in items if item.source == source),
            False,
            "offline",
        )
        for source in sources
    )
    return statuses, tuple(items)


def _watch(args: argparse.Namespace) -> int:
    language = i18n.lang()
    requested = tuple(
        item.strip().lower() for item in args.sources.split(",") if item.strip()
    )
    if not requested:
        requested = ("zenn", "qiita")
    if args.unit:
        filename, text, command = watch_unit(args.interval_hours)
        payload = {
            "filename": filename,
            "text": text,
            "install_command": command,
        }
        if args.json:
            _print_json(payload)
        else:
            print(
                f"{i18n.t('label.watch_filename', language)}: {filename}\n\n"
                f"{text}\n{i18n.t('label.watch_install', language)}:\n{command}"
            )
        return 0
    try:
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            if args.offline:
                statuses, items = _offline_items(args.offline, requested)
            else:
                results: dict[str, tuple[SourceStatus, tuple[SourceItem, ...]]] = {}
                for source in requested:
                    if source == "zenn":
                        results[source] = fetch_zenn(limit=args.limit, client=client)
                    elif source == "qiita":
                        results[source] = fetch_qiita(limit=args.limit, client=client)
                    elif source == "x":
                        results[source] = fetch_x(
                            "llm OR ollama OR llama.cpp OR vllm OR gguf",
                            args.limit,
                            client,
                        )
                statuses = tuple(result[0] for result in results.values())
                items = tuple(item for result in results.values() for item in result[1])
            mentions = extract_mentions(items)
            mentioned_repo_ids = {
                mention.value.casefold()
                for mention in mentions
                if mention.kind == "model_repo"
            }
            catalog_stats = {
                "mentioned_repo_ids": len(mentioned_repo_ids),
                "in_catalog": 0,
                "resolved_repo_ids": 0,
            }
            findings = verify(mentions, client, catalog_stats)
    except (
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        httpx.HTTPError,
    ) as error:
        print(i18n.t("err.watch_run", language, error=error), file=sys.stderr)
        return 1
    state = load_state()
    stamp = now_iso()
    finding_keys = {f"{item.kind}|{item.value}" for item in findings}
    new_keys = (
        finding_keys
        if args.all
        else {key for key in finding_keys if key not in state.seen_findings}
    )
    next_state = WatchState(
        stamp,
        {**state.seen_items, **{item.url: stamp for item in items}},
        {**state.seen_findings, **{key: stamp for key in finding_keys}},
    )
    try:
        save_state(next_state)
    except OSError as error:
        print(i18n.t("warn.watch_state", language, error=error))
    drafts: list[str] = []
    if args.write_drafts:
        for finding in findings:
            if finding.kind == "catalog_gap":
                try:
                    drafts.append(str(write_draft(finding, args.write_drafts)))
                except OSError as error:
                    print(i18n.t("warn.watch_draft", language, error=error))
    notes = [
        i18n.t("note.watch_external_claim", language),
        i18n.t("note.watch_route", language),
    ]
    if (
        any(mention.kind == "flag" for mention in mentions)
        and not caps_available()
    ):
        notes.append(i18n.t("note.watch_caps_unavailable", language))
    if any(finding.kind == "flag_unknown" for finding in findings):
        notes.append(i18n.t("note.watch_unknown_flag", language))
    if any(status.name == "zenn" and status.reachable for status in statuses):
        notes.append(i18n.t("note.watch_zenn_body", language))
    for status in statuses:
        if not status.reachable:
            notes.append(i18n.t(
                "warn.watch_unreachable",
                language,
                source=status.name,
                detail=status.detail,
            ))
    catalog_models = load_catalog()
    catalog_metrics = {
        "entries": len(catalog_models),
        "repo_ids": len({
            value.casefold()
            for model in catalog_models
            for value in model.sources.values()
        }),
        "mentioned_repo_ids": catalog_stats["mentioned_repo_ids"],
        "in_catalog": catalog_stats["in_catalog"],
        "resolved_repo_ids": catalog_stats["resolved_repo_ids"],
        "absent_repo_ids": len({
            finding.value.casefold()
            for finding in findings
            if finding.kind == "catalog_gap"
        }),
    }
    budget_bytes, budget_source = _watch_budget()
    candidate_findings = [finding for finding in findings if finding.kind == "catalog_gap"]
    fit_counts = {
        fit: sum(_candidate_fit(finding, budget_bytes) == fit for finding in candidate_findings)
        for fit in ("no_weights", "gated", "role_unknown", "not_text", "too_large", "fits")
    }
    candidates = {
        "total": len(candidate_findings),
        "counts": fit_counts,
        "fits": [
            finding.value
            for finding in candidate_findings
            if _candidate_fit(finding, budget_bytes) == "fits"
        ],
        "budget_bytes": budget_bytes,
        "budget_source": budget_source,
    }
    output = {
        "sources": [asdict(status) for status in statuses],
        "items": len(items),
        "mentions": len(mentions),
        "findings": [asdict(finding) for finding in findings],
        "new_findings": len(new_keys),
        "drafts": drafts,
        "notes": notes,
        "catalog": catalog_metrics,
        "candidates": candidates,
    }
    if args.json:
        _print_json(output)
    else:
        table = Table(title=i18n.t("label.watch_title", language))
        table.add_column(i18n.t("label.watch_source", language))
        table.add_column(i18n.t("label.watch_reachable", language))
        table.add_column(i18n.t("label.watch_items", language))
        table.add_column(i18n.t("label.watch_detail", language))
        for status in statuses:
            table.add_row(
                status.name,
                str(status.reachable),
                str(status.items),
                status.detail,
            )
        _console().print(table)
        catalog_table = Table(title=i18n.t("label.watch_catalog_title", language))
        catalog_table.add_column(i18n.t("label.watch_metric", language))
        catalog_table.add_column(i18n.t("label.watch_value", language))
        catalog_table.add_row(
            i18n.t("label.watch_catalog_entries", language),
            str(catalog_metrics["entries"]),
        )
        catalog_table.add_row(
            i18n.t("label.watch_catalog_repo_ids", language),
            str(catalog_metrics["repo_ids"]),
        )
        catalog_table.add_row(
            i18n.t("label.watch_catalog_mentioned", language),
            str(catalog_metrics["mentioned_repo_ids"]),
        )
        catalog_table.add_row(
            i18n.t("label.watch_catalog_in_catalog", language),
            str(catalog_metrics["in_catalog"]),
        )
        catalog_table.add_row(
            i18n.t("label.watch_catalog_resolved", language),
            str(catalog_metrics["resolved_repo_ids"]),
        )
        catalog_table.add_row(
            i18n.t("label.watch_catalog_absent", language),
            str(catalog_metrics["absent_repo_ids"]),
        )
        _console().print(catalog_table)
        candidate_table = Table(title=i18n.t("label.watch_candidates_title", language))
        candidate_table.add_column(i18n.t("label.watch_fit_class", language))
        candidate_table.add_column(i18n.t("label.watch_value", language))
        for fit, count in fit_counts.items():
            candidate_table.add_row(
                i18n.t(f"label.watch_fit_{fit}", language),
                str(count),
            )
        candidate_table.add_row(
            i18n.t("label.watch_candidate_budget", language),
            f"{budget_bytes:.0f} ({budget_source})",
        )
        _console().print(candidate_table)
        for finding in findings:
            _console().print(i18n.t(
                "label.watch_finding",
                language,
                kind=finding.kind,
                value=finding.value,
                mentions=finding.mentions,
            ))
        for note in notes:
            _console().print(note)
    return 1 if statuses and all(not status.reachable for status in statuses) else 0


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


def _evidence(args: argparse.Namespace) -> int:
    payload = collect_evidence()
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    language = i18n.lang()
    records = payload["records"]
    assert isinstance(records, list)
    for kind, title_key in (
        ("bench", "evidence.bench_title"),
        ("eval", "evidence.eval_title"),
        ("depth", "evidence.depth_title"),
        ("embed", "evidence.embed_title"),
        ("retrieval", "evidence.retrieval_title"),
        ("spec", "evidence.spec_title"),
    ):
        # Bound value and key-text columns so evidence strings fit at 80 columns.
        width_options = {
            "value": {"max_width": 10},
            "reasons": {"max_width": 16},
            "remeasure": {"max_width": 11},
        }
        fold_columns = {"model", "reasons", "remeasure"}
        if kind == "depth":
            # Keep the numeric depth scope and verdict intact at 80 columns.
            width_options.update({
                "scope": {"max_width": 21},
                "value": {"max_width": 8},
            })
            fold_columns.update({"scope", "value"})
        table = Table(title=i18n.t(title_key, language), padding=(0, 0))
        for column in (
            "model", "quant", "backend", "scope", "value", "usable",
            "reasons", "remeasure",
        ):
            options = (
                {
                    **width_options.get(column, {}),
                    "overflow": "fold",
                    "no_wrap": False,
                }
                if column in fold_columns
                else width_options.get(column, {})
            )
            table.add_column(
                i18n.t(f"evidence.column.{column}", language),
                **options,
            )
        seen_reasons: set[str] = set()
        for row in records:
            if row["kind"] != kind:
                continue
            scope = (
                row.get("suite", "")
                if kind == "eval"
                else (
                    f"req {row['requested_depth']} / served {row['served_depth']}"
                    if kind == "depth"
                    else str(row.get("spec", "")) if kind == "spec" else ""
                )
            )
            reasons = row["reasons"]
            assert isinstance(reasons, list)
            seen_reasons.update(str(reason) for reason in reasons)
            value = str(row["value"])
            if kind == "bench":
                value = f"{float(row['value']):.1f} tok/s"
            elif kind == "depth":
                value = str(row["verdict"]) or "-"
            table.add_row(
                str(row["model_id"]),
                str(row["quant"]),
                str(row["backend"]),
                str(scope),
                value,
                str(row["usable"]),
                "\n".join(str(reason) for reason in reasons),
                str(row["remeasure"]),
            )
        _console().print(table)
        for reason in sorted(seen_reasons):
            _console().print(
                f"{reason}: {i18n.t(f'evidence.reason.{reason}', language)}"
            )
        if kind == "depth":
            _console().print(i18n.t("evidence.depth_json_hint", language))
    if not records:
        _console().print(i18n.t("evidence.empty", language))
    decisions = payload.get("recommendations")
    if isinstance(decisions, list) and decisions:
        actions = Table(title=i18n.t("evidence.actions_title", language))
        for column in ("priority", "action", "reason", "confidence"):
            actions.add_column(i18n.t(f"evidence.column.{column}", language))
        action_reasons: set[str] = set()
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            reason = str(decision.get("reason", ""))
            action_reasons.update(reason.split("+"))
            confidence = decision.get("confidence", 0)
            actions.add_row(
                str(decision.get("priority", "")),
                str(decision.get("action", "")),
                reason,
                f"{float(confidence):.2f}" if isinstance(confidence, (int, float)) else "",
            )
        _console().print(actions)
        for reason in sorted(action_reasons):
            _console().print(
                f"{reason}: {i18n.t(f'evidence.reason.{reason}', language)}"
            )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    _configure_output()
    parser = argparse.ArgumentParser(prog="nmesh")
    parser.add_argument(
        "--version",
        action="version",
        version=(
            f"nmesh {__version__} "
            f"(bench={BENCH_HARNESS_VERSION}, "
            f"probe_rules={suite_digest(needle_tasks(0, 'core'))})"
        ),
    )
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
    plan.add_argument("--roles", default=None)
    plan.add_argument("--context", type=int)
    plan.add_argument("--budget", choices=("total", "free"), default="total")
    plan.add_argument("--kv-quant", choices=("f16", "q8_0"), default="f16")
    plan.add_argument("--parallel-slots", type=int)
    plan.add_argument("--model", help="comma-separated model IDs")
    plan.add_argument("--ignore-eval-evidence", action="store_true")
    plan.add_argument("--spec", choices=("none", "ngram", "draft"), default="none")
    plan.add_argument("--spec-draft", default="")
    plan.add_argument("--spec-n-max", type=int, default=3)
    plan.add_argument("--sleep-idle-seconds", type=int, default=0)
    plan.add_argument("--cache-reuse", type=int, default=0)
    plan.add_argument("--context-shift", action="store_true")
    plan.add_argument("--ignore-spec-evidence", action="store_true")
    plan.add_argument("--lang")
    plan.add_argument("--profile")
    up_parser = sub.add_parser("up")
    up_parser.add_argument("--json", action="store_true")
    up_parser.add_argument("--dry-run", action="store_true")
    up_parser.add_argument("--no-download", action="store_true")
    up_parser.add_argument("--kv-quant", choices=("f16", "q8_0"), default="f16")
    up_parser.add_argument("--detach", action="store_true")
    up_parser.add_argument("--port", type=int, default=18000)
    up_parser.add_argument("--ignore-free-memory", action="store_true")
    up_parser.add_argument("--lang")
    up_parser.add_argument("--roles", default=None)
    up_parser.add_argument("--model", help="comma-separated model IDs")
    up_parser.add_argument("--ignore-eval-evidence", action="store_true")
    up_parser.add_argument("--spec", choices=("none", "ngram", "draft"), default="none")
    up_parser.add_argument("--spec-draft", default="")
    up_parser.add_argument("--spec-n-max", type=int, default=3)
    up_parser.add_argument("--sleep-idle-seconds", type=int, default=0)
    up_parser.add_argument("--cache-reuse", type=int, default=0)
    up_parser.add_argument("--context-shift", action="store_true")
    up_parser.add_argument("--ignore-spec-evidence", action="store_true")
    serve_parser = sub.add_parser("serve")
    serve_parser.add_argument("--port", type=int, default=18000)
    serve_parser.add_argument("--roles", default=None)
    reload_parser = sub.add_parser("reload")
    reload_parser.add_argument("--port", type=int, default=18000)
    reload_parser.add_argument("--json", action="store_true")
    unload_parser = sub.add_parser("unload")
    unload_parser.add_argument("service", nargs="?")
    unload_parser.add_argument("--port", type=int, default=18000)
    unload_parser.add_argument("--json", action="store_true")
    for name in ("status", "down"):
        item = sub.add_parser(name)
        item.add_argument("--json", action="store_true")
        item.add_argument("--port", type=int, default=18000)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("prompt")
    run_parser.add_argument("--role", default="chat")
    run_parser.add_argument("--json", action="store_true")
    bench_parser = sub.add_parser("bench")
    bench_parser.add_argument("--service", default="chat")
    bench_parser.add_argument("--tokens", type=int, default=128)
    bench_parser.add_argument("--runs", type=_positive_int, default=3)
    bench_parser.add_argument("--passes", type=_positive_int, default=2)
    bench_parser.add_argument("--no-reference", action="store_true")
    bench_parser.add_argument("--json", action="store_true")
    bench_parser.add_argument(
        "--retrieval",
        action="store_true",
        help="measure retrieval usability (432 ladder requests plus about 72 "
        "chunk-arm requests; wall time scales with this host's measured "
        "encode throughput and is estimated from a calibration request "
        "before the ladder runs)",
    )
    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("--service", default="chat")
    eval_parser.add_argument("--json", action="store_true")
    eval_parser.add_argument(
        "--categories",
        default=None,
        help="comma-separated categories "
        f"({','.join(EXTENDED_CATEGORIES)})",
    )
    orchestrate_parser = sub.add_parser("orchestrate")
    orchestrate_commands = orchestrate_parser.add_subparsers(
        dest="orchestrate_command",
        required=True,
    )
    measure_parser = orchestrate_commands.add_parser("measure")
    measure_parser.add_argument("--suite", choices=("core", "extended", "hard"), default="hard")
    measure_parser.add_argument("--lead", default="chat")
    measure_parser.add_argument("--worker", default="worker")
    measure_parser.add_argument("--lead-url")
    measure_parser.add_argument("--worker-url")
    measure_parser.add_argument("--reasoning-allowance", type=int, default=0)
    measure_parser.add_argument("--repeats", type=_positive_int, default=2)
    measure_parser.add_argument("--limit", type=_positive_int)
    measure_parser.add_argument("--no-reference", action="store_true")
    measure_parser.add_argument("--json", action="store_true")
    show_parser = orchestrate_commands.add_parser("show")
    show_parser.add_argument("--json", action="store_true")
    spec_parser = sub.add_parser("spec")
    spec_commands = spec_parser.add_subparsers(dest="spec_command", required=True)
    spec_measure = spec_commands.add_parser("measure")
    spec_measure.add_argument("--kind", choices=("ngram", "draft"), required=True)
    spec_measure.add_argument("--draft")
    spec_measure.add_argument("--repeats", type=int, default=3)
    spec_measure.add_argument("--n-max", type=int, default=3)
    spec_measure.add_argument("--service")
    spec_measure.add_argument("--no-reference", action="store_true")
    spec_measure.add_argument("--json", action="store_true")
    spec_show = spec_commands.add_parser("show")
    spec_show.add_argument("--json", action="store_true")
    eval_parser.add_argument(
        "--suite", choices=("core", "extended", "hard"), default="core",
    )
    eval_parser.add_argument(
        "--depth",
        type=_non_negative_int,
        default=0,
        help="requested real prompt depth for evaluation evidence",
    )
    eval_parser.add_argument(
        "--reasoning-allowance",
        type=_non_negative_int,
        default=0,
        dest="reasoning_allowance",
        help="extra output tokens per task for models that emit reasoning "
        "before the answer (recorded with the result)",
    )
    eval_parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=None,
        dest="timeout",
        help="per-task request timeout in seconds; default is derived from "
        "the task token budget",
    )
    evidence_parser = sub.add_parser("evidence")
    evidence_parser.add_argument("--json", action="store_true")
    logs_parser = sub.add_parser("logs")
    logs_parser.add_argument("service", nargs="?")
    logs_parser.add_argument("--lines", type=_positive_int, default=50)
    logs_parser.add_argument("--json", action="store_true")
    watch_parser = sub.add_parser("watch")
    watch_parser.add_argument("--sources", default="zenn,qiita")
    watch_parser.add_argument(
        "--limit",
        type=_positive_int,
        default=20,
        help="items per source tag/topic",
    )
    watch_parser.add_argument("--all", action="store_true", dest="all")
    watch_parser.add_argument("--json", action="store_true")
    watch_parser.add_argument("--write-drafts")
    watch_parser.add_argument("--offline")
    watch_parser.add_argument("--unit", action="store_true")
    watch_parser.add_argument("--interval-hours", type=_positive_int, default=24)
    auto = sub.add_parser("autotune")
    auto.add_argument("--json", action="store_true")
    autostart = sub.add_parser("autostart")
    autostart.add_argument("--port", type=int, default=18000)
    autostart.add_argument("--install", action="store_true")
    autostart.add_argument("--json", action="store_true")
    models = sub.add_parser("models")
    models.add_argument("--role")
    models.add_argument("--json", action="store_true")
    model_commands = models.add_subparsers(dest="models_command")
    model_list = model_commands.add_parser("local")
    model_list.add_argument("--json", action="store_true")
    model_scan = model_commands.add_parser("scan")
    model_scan.add_argument("--json", action="store_true")
    model_scan.add_argument("--root", action="append", default=[])
    model_rm = model_commands.add_parser("rm")
    model_rm.add_argument("name")
    model_rm.add_argument("--force", action="store_true")
    model_rm.add_argument("--json", action="store_true")
    engine = sub.add_parser("engine")
    engine_commands = engine.add_subparsers(dest="engine_command", required=True)
    engine_list = engine_commands.add_parser("list")
    engine_list.add_argument("--available", action="store_true")
    engine_list.add_argument("--json", action="store_true")
    engine_install = engine_commands.add_parser("install")
    engine_install.add_argument("--version")
    engine_install.add_argument("--variant", default="auto")
    engine_install.add_argument("--json", action="store_true")
    engine_use = engine_commands.add_parser("use")
    engine_use.add_argument("tag")
    engine_use.add_argument("--json", action="store_true")
    engine_remove = engine_commands.add_parser("remove")
    engine_remove.add_argument("tag")
    engine_remove.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.global_json and hasattr(args, "json"):
        args.json = True
    if args.command == "doctor":
        return _doctor(args.json, args.profile)
    if args.command == "plan":
        return _plan(args)
    if args.command == "reload":
        return _reload(args)
    if args.command == "unload":
        return _unload(args)
    if args.command in {"up", "down", "status", "serve"}:
        if args.command == "up":
            args.dry_run = args.dry_run or args.global_dry_run
        return _runtime(args)
    if args.command == "models":
        return _models(args)
    if args.command == "engine":
        return _engine(args)
    if args.command == "bench":
        return _bench(args)
    if args.command == "logs":
        return _logs(args)
    if args.command == "eval":
        return _eval(args)
    if args.command == "evidence":
        return _evidence(args)
    if args.command == "orchestrate":
        if args.orchestrate_command == "measure":
            return _orchestrate_measure_command(args)
        return _orchestrate_show(args)
    if args.command == "spec":
        if args.spec_command == "measure":
            return _spec_measure_command(args)
        return _spec_show(args)
    if args.command == "watch":
        return _watch(args)
    if args.command == "autotune":
        saved_plan = load_plan()
        if saved_plan is None or not saved_plan.services:
            return 1
        service = saved_plan.services[0]
        if service.roles == ["embed"]:
            print(
                i18n.t(
                    "err.bench_embedding",
                    i18n.lang(),
                    service=service.name,
                ),
                file=sys.stderr,
            )
            return 2
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
                tuned_plan = replace(saved_plan, services=[
                    tuned if item.name == service.name else item for item in saved_plan.services
                ])
                runtime_down()
                try:
                    runtime_up(tuned_plan, no_download=True)
                    tuned_result = measure(tuned, base_url)
                except (OSError, RuntimeError) as error:
                    print(i18n.t("err.autotune_measure", i18n.lang(), error=error),
                          file=sys.stderr)
                    runtime_down()
                    try:
                        runtime_up(saved_plan, no_download=True)
                    except (OSError, RuntimeError) as restore_error:
                        print(
                            i18n.t("err.autotune_restore", i18n.lang(), error=restore_error),
                            file=sys.stderr,
                        )
                    return 1
                save_plan(saved_plan)
                if best is None or tuned_result.decode_tps > best[2]:
                    best = (context, layers, tuned_result.decode_tps)
                    best_plan = tuned_plan
        if best is None:
            runtime_down()
            try:
                runtime_up(saved_plan, no_download=True)
            except (OSError, RuntimeError):
                pass
            return 1
        runtime_down()
        assert best_plan is not None
        save_plan(best_plan)
        try:
            runtime_up(best_plan, no_download=True)
        except (OSError, RuntimeError) as error:
            save_plan(saved_plan)
            runtime_down()
            try:
                runtime_up(saved_plan, no_download=True)
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
        summary = {"service": service.name, "context": best[0], "n_gpu_layers": best[1],
                   "decode_tps": best[2]}
        _print_json(summary) if args.json else _console().print(summary)
        return 0
    if args.command == "autostart":
        if is_windows():
            os_name = "nt"
        elif sys.platform.startswith("darwin"):
            os_name = "darwin"
        else:
            os_name = "posix"
        filename, text, install_command = service_unit(args.port, os_name=os_name)
        launcher_filename, launcher_text = launcher_script(args.port, os_name=os_name)
        home = nmesh_home()
        launcher_path = home / launcher_filename
        env_path = home / "gateway.env"
        installed = False
        unit_path: Path | None = None
        if args.install:
            home.mkdir(parents=True, exist_ok=True)
            launcher_path.write_text(launcher_text, encoding="utf-8", newline="")
            if not is_windows():
                launcher_path.chmod(0o700)
            if not env_path.exists():
                api_key = os.environ.get("NMESH_API_KEY", "")
                env_path.write_text(
                    f"NMESH_API_KEY={api_key}\n"
                    "# NMESH_HOME is set by the launcher to its own directory.\n",
                    encoding="utf-8",
                    newline="",
                )
                if not is_windows():
                    env_path.chmod(0o600)
            unit_path = unit_install_path(filename, os_name)
            if unit_path is not None:
                unit_path.parent.mkdir(parents=True, exist_ok=True)
                unit_path.write_text(text, encoding="utf-8", newline="")
            installed = True
        limitations = []
        if is_windows():
            limitations.append(i18n.t("autostart.windows_limitations", i18n.lang()))
        data = {
            "filename": filename,
            "text": text,
            "install_command": install_command,
            "launcher_path": str(launcher_path),
            "env_path": str(env_path),
            "unit_path": str(unit_path) if unit_path is not None else None,
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
                if unit_path is not None:
                    print(i18n.t("label.unit_written", i18n.lang(), path=unit_path))
            for limitation in limitations:
                print(i18n.t("label.autostart_limitation", i18n.lang(), text=limitation))
        return 0
    if args.command == "run":
        return _run_prompt(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
