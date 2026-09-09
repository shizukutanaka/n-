from __future__ import annotations

import json
from dataclasses import asdict

from rich.console import Console

from nmesh import cli
from nmesh.bench.cache import (
    BENCH_HARNESS_VERSION,
    BenchRecord,
    benchmark_key,
    load_cache,
    save_records,
)
from nmesh.eval import SUITES, needle_tasks, suite_digest
from nmesh.eval.cache import EvalRecord
from nmesh.eval.context import ContextRecord, FamilyResult
from nmesh.evidence_inventory import collect_evidence


def _bench(
    harness: str = BENCH_HARNESS_VERSION,
    *,
    stable: bool = True,
    epoch: str = "healthy",
    sessions: tuple[float, ...] = (20.0, 20.0),
) -> BenchRecord:
    return BenchRecord(
        20.0, 19.0, 21.0, 2, 2, 0.98, stable, "now", harness, sessions,
        epoch=epoch,
    )


def _eval(
    *,
    suite: str = "core",
    digest: str | None = None,
    transport_errors: int = 0,
    depth: int = 0,
    at: float | None = None,
) -> EvalRecord:
    return EvalRecord(
        model_id="model",
        quant="f16",
        backend="llamacpp",
        n_tasks=16,
        passed=12,
        pass_rate=0.75,
        by_category={},
        at=(
            float(depth + transport_errors + 1)
            if at is None else at
        ),
        suite=suite,
        digest=(
            suite_digest(SUITES["core"])
            if digest is None else digest
        ),
        transport_errors=transport_errors,
        depth=depth,
    )


def _write_records(home, name: str, records: dict[str, object]) -> None:
    (home / name).write_text(
        json.dumps({"results": {key: asdict(value) for key, value in records.items()}}),
        encoding="utf-8",
    )


def _rows(kind: str) -> list[dict[str, object]]:
    return [
        row for row in collect_evidence()["records"]
        if row["kind"] == kind
    ]


def test_bench_reasons_and_load_cache_consistency(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    legacy_key = benchmark_key("legacy-model", "f16", "llamacpp", "cpu", 0)
    current_key = benchmark_key("current-model", "f16", "llamacpp", "cpu", 0)
    unstable_key = benchmark_key("unstable-model", "f16", "llamacpp", "cpu", 0)
    degraded_key = benchmark_key("degraded-model", "f16", "llamacpp", "cpu", 0)
    save_records({
        legacy_key: _bench("bench-v1"),
        current_key: _bench(sessions=(20.0,)),
        unstable_key: _bench(stable=False),
        degraded_key: _bench(epoch="degraded"),
    })

    rows = {row["key"]: row for row in _rows("bench")}
    assert rows[legacy_key]["usable"] is False
    assert "harness_mismatch" in rows[legacy_key]["reasons"]
    assert rows[legacy_key]["remeasure"] == "nmesh bench"
    assert rows[current_key]["usable"] is True
    assert rows[current_key]["reasons"] == ["unconfirmed"]
    assert "unstable" in rows[unstable_key]["reasons"]
    assert "epoch_degraded" in rows[degraded_key]["reasons"]

    cached = set(load_cache())
    visible = {
        key for key, row in rows.items()
        if row["usable"] and set(row["reasons"]) <= {"unconfirmed"}
    }
    assert cached == visible


def test_eval_reasons_and_depth_probe_mismatch(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    digest = suite_digest(SUITES["core"])
    _write_records(tmp_path, "eval.json", {
        "stale": _eval(digest="stale"),
        "transport": _eval(transport_errors=1),
        "deep": _eval(depth=4096),
        "unknown": _eval(suite="missing", digest=digest),
    })
    probe = ContextRecord(
        "model", "f16", "llamacpp", "core", 8, 8,
        "stale-probe",
        (FamilyResult("literal", 8, 8, 8, 8),),
        1.0,
    )
    _write_records(tmp_path, "context.json", {"probe": probe})

    eval_rows = {row["key"]: row for row in _rows("eval")}
    assert "grader_digest_mismatch" in eval_rows["stale"]["reasons"]
    assert "transport_errors" in eval_rows["transport"]["reasons"]
    assert "depth_scoped" in eval_rows["deep"]["reasons"]
    assert "suite_unknown" in eval_rows["unknown"]["reasons"]
    depth_row = _rows("depth")[0]
    assert "probe_digest_mismatch" in depth_row["reasons"]
    assert "lost" not in depth_row["value"]
    assert depth_row["remeasure"] == "nmesh eval --depth 8"


def test_evidence_json_and_empty_home_are_successful(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    assert cli.main(["evidence", "--json"]) == 0
    empty = json.loads(capsys.readouterr().out)
    assert empty["records"] == []
    assert empty["counts"]["total"] == 0

    save_records({
        benchmark_key("model", "f16", "llamacpp", "cpu", 0): _bench(),
    })
    assert cli.main(["evidence", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {"records", "counts"} <= set(payload)
    assert payload["counts"]["bench"] == 1


def test_evidence_table_folds_reasons_and_remeasure_at_narrow_width(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    save_records({
        benchmark_key("legacy-model", "f16", "llamacpp", "cpu", 0): _bench(
            "bench-v1",
        ),
    })
    probe_digest = suite_digest(needle_tasks(8, "core"))
    _write_records(tmp_path, "context.json", {
        "depth": ContextRecord(
            "depth-model", "f16", "llamacpp", "core", 8, 8, probe_digest,
            (FamilyResult("literal", 8, 8, 8, 8),), 1.0,
        ),
    })
    console = Console(width=80, record=True, color_system=None)
    monkeypatch.setattr(cli, "_console", lambda: console)

    assert cli.main(["evidence"]) == 0

    output = console.export_text()
    assert "Kind" not in output
    assert "harness_mismatch" in output
    assert "nmesh bench" in output
    assert "20.0 tok/s" in output
    assert "req 8 / served 8" in output
    assert "verified" in output


def test_superseded_records_are_explained_and_all_unusable_rows_have_reasons(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("NMESH_HOME", str(tmp_path))
    save_records({
        benchmark_key("model", "f16", "llamacpp", "cpu", 0): _bench(),
    })
    _write_records(tmp_path, "eval.json", {
        "old": _eval(at=1.0),
        "new": _eval(at=2.0),
    })
    probe_digest = suite_digest(needle_tasks(8, "core"))
    _write_records(tmp_path, "context.json", {
        "old": ContextRecord(
            "model", "f16", "llamacpp", "core", 8, 8, probe_digest,
            (FamilyResult("literal", 8, 8, 8, 8),), 1.0,
        ),
        "new": ContextRecord(
            "model", "f16", "llamacpp", "core", 8, 8, probe_digest,
            (FamilyResult("literal", 8, 8, 8, 8),), 2.0,
        ),
    })

    payload = collect_evidence()
    rows = payload["records"]
    superseded = {
        row["key"]: row for row in rows if row["key"] == "old"
    }
    assert superseded["old"]["reasons"] == ["superseded"]
    assert superseded["old"]["usable"] is False
    assert superseded["old"]["remeasure"] == ""
    assert "lost" not in superseded["old"]["value"]
    assert all(
        row["usable"] or row["reasons"]
        for row in rows
    )
