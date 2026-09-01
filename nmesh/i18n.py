"""Small, failure-tolerant translation layer for nmesh output."""

from __future__ import annotations

import locale
import os
import re

SUPPORTED = ("en", "ja")

MESSAGES = {
    "en": {
        "install.ollama": "Install Ollama: https://ollama.com/download",
        "install.llamacpp": "Install llama.cpp: winget install llama.cpp / brew install llama.cpp / build from source",
        "install.vllm": "Install vLLM: pip install vllm",
        "install.mlx": "Install MLX-LM: pip install mlx-lm",
        "warn.parallel_unsupported": "llamacpp: --parallel unsupported; using one slot and plain context",
        "warn.gpu_layers_unsupported": "llamacpp: GPU-layer flags unsupported; using CPU placement",
        "warn.tensor_split_unsupported": "llamacpp: --tensor-split unsupported; omitting tensor split",
        "warn.gpu_over_budget": "{service}: GPU placement needs {committed:.0f} bytes, over the {budget:.0f}-byte budget",
        "warn.backend_placement_estimate": "{service} uses {backend}, which manages its own placement; this RAM fallback is nmesh's estimate, not a controllable setting.",
        "warn.gpu_layers_cpu_fallback": "{service}: GPU layers did not fit on any card; it will run on the CPU.",
        "warn.layers_reduced": "{service}: reduced GPU layers to {layers}; {previous} layers did not fit in RAM",
        "warn.slots_clamped": "{service}: parallel_slots {requested} exceeds available memory; using {slots}",
        "warn.no_source": "{model}: no Hugging Face or Ollama source is configured",
        "warn.no_candidate": "No runnable model found for role {role}",
        "warn.nvidia_unavailable": "NVIDIA detection unavailable",
        "warn.parallel_clamped": "llama.cpp: --parallel unsupported; using one slot instead of {requested}",
        "warn.rocm_parse": "Unable to parse rocm-smi output",
        "warn.mlx_check": "Unable to check mlx_lm",
        "warn.system_probe": "System probe failed: {error}",
        "warn.backend_no_gpu": "llama.cpp binary reports no GPU backend; using CPU placement. Install a Vulkan, CUDA, HIP, or SYCL build to use the detected GPU",
        "warn.low_vram": "{gpu} has less than 2 GiB dedicated VRAM; it remains visible but is not used for GPU placement",
        "warn.download_budget": "Planned downloads exceed policy.allow_download_gb.",
        "warn.language_coverage": "{model} does not claim coverage for {languages}; language coverage is a publisher/vendor claim, not a measured benchmark.",
        "warn.simulated_profile": "Simulated profile: planning decisions only; no runtime or measurements are performed.",
        "hint.language": "Locale is {language}; use --lang {language} to prioritize models claiming {language}.",
        "warn.free_admission": "Free-memory admission: planned usage {need_gpu:.2f} GiB VRAM / {need_cpu:.2f} GiB RAM exceeds available {vram:.2f} GiB VRAM / {ram:.2f} GiB RAM.",
        "warn.free_admission_fallback": "Free-memory admission failed; using the original plan. Reason: {error}",
        "warn.runtime_fallback": "Runtime fallback attempt {attempt}",
        "warn.admission_skipped": "Free-memory admission skipped: {error}",
        "warn.health_failed": "Service failed health check: {service}",
        "info.resolved_gguf": "resolved GGUF: {name}",
        "err.no_plan": "No plan found",
        "err.no_active_plan": "No active plan",
        "err.unknown_service": "Unknown service: {service}",
        "err.restart_budget": "Restart budget exhausted: {service}",
        "err.service_unhealthy": "Service did not become healthy: {service}",
        "err.runtime_start": "Runtime startup failed",
        "err.gateway_reload": "gateway reload failed: {error}",
        "err.gateway_unavailable": "gateway unavailable: {error}",
        "err.doctor": "doctor failed: {error}",
        "err.plan": "plan failed: {error}",
        "err.profile_load": "failed to load profile: {error}",
        "err.plan_empty": "plan produced no runnable services",
        "err.plan_save": "plan failed to save: {error}",
        "err.gateway_start": "gateway failed to start: {error}",
        "err.up": "up failed: {error}",
        "err.gateway_not_ready": "gateway did not become ready; see {path}",
        "err.bench_up": "Service is not running; run nmesh up first.",
        "err.bench_measure": "Benchmark failed: {error}",
        "err.bench_save": "benchmark failed to save: {error}",
        "err.autotune_measure": "Autotune failed: {error}",
        "err.autotune_restore": "failed to restore original autotune configuration: {error}",
        "err.autotune_winning": "failed to restore winning autotune configuration: {error}",
        "label.item": "Item",
        "label.value": "Value",
        "label.free": "free",
        "label.vram_source": "VRAM source",
        "label.gateway_log": "Gateway log: {path}",
        "label.none": "none",
        "label.not_found": "not found",
        "label.unknown": "unknown",
        "label.backends": "Backends",
        "label.backend": "Backend",
        "label.binary": "Binary",
        "label.version": "Version",
        "label.flags": "Flags",
        "label.models": "Models",
        "label.selected_models": "Selected models",
        "label.service": "Service",
        "label.model": "Model",
        "label.languages": "Languages",
        "label.roles": "Roles",
        "label.context": "Context",
        "label.slots": "Slots",
        "label.gpu_layers": "GPU layers",
        "label.tps": "tok/s",
        "label.saved_to": "Saved to: {path}",
        "label.free_budgets": "Budgets use currently-free memory.",
        "label.telemetry_overlay": "Live telemetry overlay: {count} benchmark key(s)",
        "label.install": "Install: {hint}",
        "label.warning": "Warning: {warning}",
        "label.memory": "Memory",
        "label.weights": "Weights",
        "label.kv": "KV",
        "warn.ollama_quant_estimate": "{service} uses the Ollama tag's own quantization; nmesh cannot control it, so this service's memory estimate is an estimate.",
        "warn.ollama_context_default": "Ollama context could not be set for {service}; it will run at Ollama's default context, not the planned {context}.",
        "label.gpu_cpu": "GPU / CPU",
        "label.acquisition_note": "{service}: acquisition note: {note}",
        "label.telemetry": "Telemetry",
        "label.samples": "Samples",
        "label.decode_median": "Decode median",
        "label.ttft_median": "TTFT median",
        "label.ttft_p95": "TTFT p95",
        "label.total_median": "Total median",
        "label.reloaded": "Reloaded: {services} (created_at={created_at})",
        "label.median_decode": "median decode: {marker}{value} tok/s",
        "label.prefill": "prefill:       {marker}{value} tok/s",
        "label.ttft": "TTFT:          {marker}{value} s",
        "label.backend_detail": "{service}: backend={backend} model_ref={model_ref} port={port} context={context} slots={slots} n_gpu_layers={layers} argv={argv}",
    },
    "ja": {
        "install.ollama": "Ollamaをインストールしてください: https://ollama.com/download",
        "install.llamacpp": "llama.cppをインストールしてください: winget install llama.cpp / brew install llama.cpp / ソースからビルド",
        "install.vllm": "vLLMをインストールしてください: pip install vllm",
        "install.mlx": "MLX-LMをインストールしてください: pip install mlx-lm",
        "warn.parallel_unsupported": "llama.cpp: --parallelは未対応です。1スロットと通常のコンテキストを使用します",
        "warn.gpu_layers_unsupported": "llama.cpp: GPUレイヤー指定は未対応です。CPU配置を使用します",
        "warn.tensor_split_unsupported": "llama.cpp: --tensor-splitは未対応です。テンソル分割を省略します",
        "warn.gpu_over_budget": "{service}: GPU配置に{committed:.0f}バイト必要ですが、予算{budget:.0f}バイトを超えています",
        "warn.backend_placement_estimate": "{service} は {backend} を使用し、配置を独自に管理します。この RAM フォールバックは nmesh の推定であり、制御可能な設定ではありません。",
        "warn.layers_reduced": "{service}: GPUレイヤーを{layers}に減らしました。{previous}レイヤーはRAMに収まりません",
        "warn.slots_clamped": "{service}: parallel_slots {requested}は空きメモリを超えるため、{slots}にします",
        "warn.no_source": "{model}: Hugging FaceまたはOllamaのソースが設定されていません",
        "warn.no_candidate": "ロール{role}に実行可能なモデルがありません",
        "warn.nvidia_unavailable": "NVIDIA の検出を利用できません。",
        "warn.parallel_clamped": "llama.cpp: --parallel は未対応です。{requested} スロットではなく1スロットを使用します。",
        "warn.rocm_parse": "rocm-smi の出力を解析できません。",
        "warn.mlx_check": "mlx_lm を確認できません。",
        "warn.system_probe": "システム検出に失敗しました: {error}",
        "warn.download_budget": "予定ダウンロードがpolicy.allow_download_gbを超えています。",
        "warn.language_coverage": "{model}は{languages}をカバーすると主張していません。言語対応は公開元/ベンダーの主張であり、測定ベンチマークではありません。",
        "hint.language": "ロケールは{language}です。{language}対応を主張するモデルを優先するには --lang {language} を使用してください。",
        "warn.free_admission": "空きメモリ判定: 使用予定はVRAM {need_gpu:.2f} GiB / RAM {need_cpu:.2f} GiBで、空きはVRAM {vram:.2f} GiB / RAM {ram:.2f} GiBです。",
        "warn.free_admission_fallback": "空きメモリ判定に失敗したため元のプランを使用します。理由: {error}",
        "warn.runtime_fallback": "ランタイムのフォールバック試行 {attempt}",
        "warn.admission_skipped": "空きメモリ判定をスキップしました: {error}",
        "warn.health_failed": "サービスのヘルスチェックに失敗しました: {service}",
        "info.resolved_gguf": "解決したGGUF: {name}",
        "err.no_plan": "プランが見つかりません",
        "err.no_active_plan": "アクティブなプランがありません",
        "err.unknown_service": "不明なサービス: {service}",
        "err.restart_budget": "再起動上限を使い切りました: {service}",
        "err.service_unhealthy": "サービスが正常になりませんでした: {service}",
        "err.runtime_start": "ランタイムの起動に失敗しました",
        "err.gateway_reload": "ゲートウェイの再読み込みに失敗しました: {error}",
        "err.gateway_unavailable": "ゲートウェイを利用できません: {error}",
        "err.bench_up": "サービスが起動していません。先にnmesh upを実行してください。",
        "err.bench_measure": "ベンチマークに失敗しました: {error}",
        "err.autotune_measure": "自動調整に失敗しました: {error}",
        "label.item": "項目",
        "label.value": "値",
        "label.free": "空き",
        "label.none": "なし",
        "label.not_found": "見つかりません",
        "label.unknown": "不明",
        "label.backends": "バックエンド",
        "label.backend": "バックエンド",
        "label.binary": "バイナリ",
        "label.version": "バージョン",
        "label.flags": "フラグ",
        "label.models": "モデル",
        "label.selected_models": "選択されたモデル",
        "label.service": "サービス",
        "label.model": "モデル",
        "label.languages": "言語",
        "label.roles": "ロール",
        "label.context": "コンテキスト",
        "label.slots": "スロット",
        "label.gpu_layers": "GPUレイヤー",
        "label.tps": "トークン/秒",
        "label.saved_to": "保存先: {path}",
        "label.free_budgets": "予算には現在空いているメモリを使用します。",
        "label.telemetry_overlay": "ライブテレメトリの上書き: ベンチマークキー{count}件",
        "label.install": "インストール: {hint}",
        "label.warning": "警告: {warning}",
        "label.memory": "メモリ",
        "label.weights": "重み",
        "label.kv": "KV",
        "label.gpu_cpu": "GPU / CPU",
        "warn.gpu_layers_cpu_fallback": "{service}: GPU レイヤーがどのカードにも収まらないため、CPU で実行します。",
        "label.acquisition_note": "{service}: 取得メモ: {note}",
        "label.telemetry": "テレメトリ",
        "label.samples": "サンプル数",
        "label.decode_median": "デコード中央値",
        "label.ttft_median": "TTFT中央値",
        "label.ttft_p95": "TTFT p95",
        "label.total_median": "合計中央値",
        "label.reloaded": "再読み込み完了: {services} (created_at={created_at})",
        "label.median_decode": "デコード中央値: {marker}{value} トークン/秒",
        "label.prefill": "プレフィル:     {marker}{value} トークン/秒",
        "label.ttft": "TTFT:          {marker}{value} 秒",
        "label.backend_detail": "{service}: backend={backend} model_ref={model_ref} port={port} context={context} slots={slots} n_gpu_layers={layers} argv={argv}",
        "warn.backend_no_gpu": "llama.cpp バイナリは GPU バックエンドを報告しません。CPU 配置を使用します。検出された GPU を使うには Vulkan、CUDA、HIP、または SYCL ビルドをインストールしてください。",
        "warn.ollama_quant_estimate": "{service} は Ollama タグ固有の量子化を使用します。nmesh は制御できないため、このサービスのメモリ見積もりは推定値です。",
        "warn.ollama_context_default": "{service} の Ollama コンテキストを設定できないため、計画した {context} ではなく Ollama のデフォルトコンテキストで実行します。",
        "warn.low_vram": "{gpu} の専用 VRAM は 2 GiB 未満です。表示は維持しますが、GPU 配置には使用しません。",
        "err.doctor": "doctor に失敗しました: {error}",
        "err.plan": "plan に失敗しました: {error}",
        "err.profile_load": "プロファイルの読み込みに失敗しました: {error}",
        "warn.simulated_profile": "シミュレーションプロファイル: 計画のみです。実行時の動作や測定は行いません。",
        "err.plan_empty": "実行可能なサービスがないプランです",
        "err.plan_save": "plan の保存に失敗しました: {error}",
        "err.gateway_start": "gateway の起動に失敗しました: {error}",
        "err.up": "up に失敗しました: {error}",
        "err.gateway_not_ready": "gateway の準備が完了しませんでした。{path} を確認してください",
        "err.bench_save": "ベンチマーク結果の保存に失敗しました: {error}",
        "err.autotune_restore": "自動調整前の設定の復元に失敗しました: {error}",
        "err.autotune_winning": "最適設定の復元に失敗しました: {error}",
        "label.vram_source": "VRAM の情報源",
        "label.gateway_log": "ゲートウェイログ: {path}",
    },
}

_PRIMARY_SUBTAG = re.compile(r"^[A-Za-z]+")


def _primary(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    match = _PRIMARY_SUBTAG.match(value.strip())
    return match.group(0).lower() if match else None


def lang() -> str:
    """Resolve a supported language without changing process locale state."""
    values: list[object] = [
        os.environ.get("NMESH_LANG"),
        os.environ.get("LC_ALL"),
        os.environ.get("LC_MESSAGES"),
        os.environ.get("LANG"),
    ]
    try:
        values.append(locale.getlocale()[0])
    except (AttributeError, ValueError):
        values.append(None)
    for value in values:
        primary = _primary(value)
        if primary:
            return primary if primary in SUPPORTED else "en"
    return "en"


def t(key: str, lang_code: str = "en", **params: object) -> str:
    """Translate and format a message; missing translations never raise."""
    language = lang_code if lang_code in MESSAGES else "en"
    template = MESSAGES[language].get(key)
    if template is None:
        return key
    try:
        return template.format(**params)
    except (KeyError, IndexError, ValueError):
        return template
